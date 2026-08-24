#ifndef MMAP_BACKEND_H
#define MMAP_BACKEND_H

/**
 * @file mmap_backend.h
 * @brief Disk-backed storage that leans on the OS page cache.
 *
 * Writes every point as a bare `Point` into one flat file, then maps the whole
 * file read-only. Blocks are delimited by point offsets into that flat array.
 *
 * Records carry no payload padding: what lands on disk is exactly the
 * coordinates. Modelling a larger tuple is the paged backend's job, via its
 * runtime record width.
 *
 * Note what this backend can and cannot tell you: residency is entirely the
 * kernel's business, so there is no memory budget to set and no miss to count.
 * `Stats()` stays zeroed. That is the motivation for the buffer-pool backend,
 * not an oversight here.
 */

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstring>
#include <cstdio>
#include <fstream>
#include <numeric>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include"storage_backend.h"
#include"sort_tools.h"


/**
 * True when the mmap backend should be materialized at all. Off by default.
 *
 * The evaluator no longer times an mmap pass: the buffer-pool backend answers
 * the same question with a memory budget attached, which the page cache cannot
 * give. Building the mapping anyway cost a full copy of the point data on disk
 * -- 128 MB per index at 8M points -- written inside every index's construction
 * window and charged to build_time, for a file nothing read.
 *
 * The backend itself is untouched and still selectable; VERIFY_BACKENDS turns it
 * back on so the three-way fingerprint check has an mmap column to compare.
 */
inline bool MmapBackendEnabled(){
    const char* flag = std::getenv("ENABLE_MMAP_BACKEND");
    return flag != nullptr && std::string(flag) == "1";
}


class MmapBackend: public PointStorageBackend{
    public:
        MmapBackend() = default;

        // Owns a mapping and a temp file; copying would double-free both.
        MmapBackend(const MmapBackend&) = delete;
        MmapBackend& operator=(const MmapBackend&) = delete;

        ~MmapBackend() override {
            if(memory_mapped_data_created_){
                if (flattened_block_list_ && munmap(flattened_block_list_, file_store_bytes_))
                    std::cerr << "munmap error " << std::string(strerror(errno));
                std::cout<<"Deleteing blockstorefile "<<blockstore_filename_<<std::endl;
                std::remove(blockstore_filename_.c_str());
            }
        }

        /** Write every block to a temp file and map it read-only. */
        void Build(const std::vector<Block>& blocks, const std::vector<size_t>& counts) override {
            //1. open file write object.
            std::string blockstore_dir = ResolveBlockstoreDir();

            file_store_size_ = std::accumulate(counts.begin(), counts.end(), size_t{0});
            if(file_store_size_ == 0)
                throw std::runtime_error("MmapBackend::Build: refusing to map an empty block store");
            file_store_bytes_ = file_store_size_*sizeof(Point);

            std::error_code dir_error;
            std::filesystem::create_directories(blockstore_dir, dir_error);

            blockstore_filename_ = blockstore_dir+generate_random_alphanumeric_string(20);
            file_write_obj_ = std::fstream(blockstore_filename_, std::ios::out | std::ios::binary);
            if(!file_write_obj_.is_open())
                throw std::runtime_error("MmapBackend::Build: cannot open "+blockstore_filename_+
                                         " (TEMP_BLOCKSTORE_DIR missing or not writable)");

            //2. write all blocks to file write object. Also keep track of block_start and block_end locations.

            size_t running_block_end = 0;
            for(size_t block_id=0;block_id<blocks.size();block_id++){
                block_start_location_.push_back(running_block_end);
                for(const Point& pt: blocks[block_id].block_data_)
                    file_write_obj_.write(reinterpret_cast<const char*>(&pt), sizeof(Point));
                running_block_end += counts[block_id];
                block_end_location_.push_back(running_block_end);
            }

            //3. close the file pointer and mmap file into array.
            file_write_obj_.close();
            if(file_write_obj_.fail()){
                std::remove(blockstore_filename_.c_str());
                throw std::runtime_error("MmapBackend::Build: failed writing "+blockstore_filename_+
                                         " (out of disk space?)");
            }

            auto fd = open(blockstore_filename_.c_str(), O_RDONLY);
            if(fd < 0){
                std::remove(blockstore_filename_.c_str());
                throw std::runtime_error("MmapBackend::Build: cannot reopen "+blockstore_filename_+
                                         ": "+std::string(strerror(errno)));
            }

            void* mapping = mmap(nullptr, file_store_bytes_, PROT_READ, MAP_SHARED, fd, 0);
            close(fd);                                 // the mapping holds its own reference to the file
            if(mapping == MAP_FAILED){
                std::remove(blockstore_filename_.c_str());
                throw std::runtime_error("MmapBackend::Build: mmap failed for "+blockstore_filename_+
                                         ": "+std::string(strerror(errno)));
            }

            flattened_block_list_ = (Point *) mapping;
            memory_mapped_data_created_ = true;        // armed only once there is something to clean up
        }

        /**
         * Drop the mapping early, before the object is destroyed.
         *
         * Used once the mmap measurement is finished so its resident pages stop
         * counting against RSS while the buffer-pool passes run. Scanning
         * afterwards is a programming error and throws rather than dereferencing a
         * dangling pointer.
         */
        void Release(){
            if(!memory_mapped_data_created_)
                return;
            if(flattened_block_list_ && munmap(flattened_block_list_, file_store_bytes_))
                std::cerr << "munmap error " << std::string(strerror(errno));
            flattened_block_list_ = nullptr;
            std::cout<<"Deleteing blockstorefile "<<blockstore_filename_<<std::endl;
            std::remove(blockstore_filename_.c_str());
            memory_mapped_data_created_ = false;
            released_ = true;
        }

        void Scan(Query& query, const std::vector<size_t>& block_ids, std::vector<Point>& result_vec) override {
            if(released_)
                throw std::runtime_error("MmapBackend::Scan: the mapping was already released");

            for(const size_t& block_id: block_ids){
                for(size_t offset=block_start_location_[block_id];offset<block_end_location_[block_id];offset++)
                    if(query.CheckPointWithin(*(flattened_block_list_+offset)))
                        result_vec.emplace_back((flattened_block_list_+offset)->elements_[0],(flattened_block_list_+offset)->elements_[1]);
            }
        }

        const StorageStats& Stats() const override { return stats_; }
        void ResetStats() override { stats_ = StorageStats{}; }
        const char* Name() const override { return "mmap"; }

        bool IsBuilt() const { return memory_mapped_data_created_; }
        const std::string& Filename() const { return blockstore_filename_; }
        size_t PointCount() const { return file_store_size_; }
        size_t MappingBytes() const { return file_store_bytes_; }

    private:
        Point* flattened_block_list_{nullptr};         // the whole file, as one flat array of points
        std::vector<size_t> block_start_location_;
        std::vector<size_t> block_end_location_;

        std::fstream file_write_obj_;
        std::string blockstore_filename_;
        size_t file_store_size_{};                     // number of points in the mapping
        size_t file_store_bytes_{};                    // length of the mapping, in bytes

        bool memory_mapped_data_created_{};
        bool released_{false};
        StorageStats stats_;                           // stays zeroed; see the file comment
};


#endif
