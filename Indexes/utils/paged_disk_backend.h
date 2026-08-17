#ifndef PAGED_DISK_BACKEND_H
#define PAGED_DISK_BACKEND_H

/**
 * @file paged_disk_backend.h
 * @brief Disk-backed storage laid out in fixed-size pages.
 *
 * Writes a self-describing paged file (see page_layout.h) and reads blocks back
 * out of it a page at a time. Blocks map to contiguous runs of pages and never
 * share a page.
 *
 * Phase 2 scope: the writer, plus a deliberately unbuffered reader -- every
 * page access is its own pread, with no cache and no residency. That makes the
 * layout verifiable on its own, before any eviction logic exists to confuse a
 * failure. Phase 3 puts a BufferPool behind FetchPage() and starts filling in
 * StorageStats; nothing else about this class needs to move.
 */

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cerrno>
#include <cstring>
#include <cstdio>
#include <numeric>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include"storage_backend.h"
#include"page_layout.h"
#include"sort_tools.h"


/** True when the paged backend should be materialized at all. Off by default. */
inline bool PagedBackendEnabled(){
    const char* flag = std::getenv("ENABLE_PAGED_BACKEND");
    return flag != nullptr && std::string(flag) == "1";
}

/** Debug escape hatch: keep the scratch file visible on disk instead of unlinking it. */
inline bool KeepBlockstoreFiles(){
    const char* flag = std::getenv("KEEP_BLOCKSTORE_FILES");
    return flag != nullptr && std::string(flag) == "1";
}


class PagedDiskBackend: public PointStorageBackend{
    public:
        explicit PagedDiskBackend(PageGeometry geometry = PageGeometry{})
            : geometry_(geometry) {}

        // Owns a file descriptor and a scratch file; copying would double-close both.
        PagedDiskBackend(const PagedDiskBackend&) = delete;
        PagedDiskBackend& operator=(const PagedDiskBackend&) = delete;

        ~PagedDiskBackend() override {
            if(fd_ >= 0)
                close(fd_);
            // Only reachable when KEEP_BLOCKSTORE_FILES kept the name alive; the
            // normal path already unlinked the file during Build().
            if(!blockstore_filename_.empty() && !unlinked_)
                std::remove(blockstore_filename_.c_str());
        }

        /** Lay every block out into pages and write the file. */
        void Build(const std::vector<Block>& blocks, const std::vector<size_t>& counts) override {
            block_point_counts_ = counts;

            //1. Work out the page layout before touching the disk.
            total_points_ = std::accumulate(counts.begin(), counts.end(), size_t{0});
            if(total_points_ == 0)
                throw std::runtime_error("PagedDiskBackend::Build: refusing to write an empty block store");

            first_page_.resize(counts.size());
            page_count_.resize(counts.size());

            uint64_t next_page = 1;                        // page 0 is the header
            for(size_t block_id=0;block_id<counts.size();block_id++){
                const size_t pages = geometry_.PagesForBlock(counts[block_id]);
                first_page_[block_id] = next_page;
                page_count_[block_id] = uint32_t(pages);
                next_page += pages;
            }
            total_data_pages_ = next_page - 1;
            file_bytes_ = (total_data_pages_ + 1)*geometry_.page_bytes_;

            //2. Create the file.
            std::string blockstore_dir = ResolveBlockstoreDir();
            std::error_code dir_error;
            std::filesystem::create_directories(blockstore_dir, dir_error);

            blockstore_filename_ = blockstore_dir+generate_random_alphanumeric_string(20);
            fd_ = open(blockstore_filename_.c_str(), O_RDWR|O_CREAT|O_EXCL, 0600);
            if(fd_ < 0)
                throw std::runtime_error("PagedDiskBackend::Build: cannot create "+blockstore_filename_+
                                         ": "+std::string(strerror(errno)));

            std::vector<char> page(geometry_.page_bytes_);

            //3. Page 0: the header, zero-padded to a full page.
            std::memset(page.data(), 0, geometry_.page_bytes_);
            PagedStoreHeader header{};
            std::memcpy(header.magic, "LSIPAGE", 8);
            header.format_version   = kPagedFormatVersion;
            header.page_bytes       = uint32_t(geometry_.page_bytes_);
            header.record_bytes     = uint32_t(geometry_.record_bytes_);
            header.dim              = uint32_t(Constants::DIM);
            header.block_count      = counts.size();
            header.total_data_pages = total_data_pages_;
            header.total_points     = total_points_;
            std::memcpy(page.data(), &header, sizeof(header));
            WriteFully(page.data(), geometry_.page_bytes_);

            //4. Data pages, in block order. Each block starts on a fresh page.
            for(size_t block_id=0;block_id<blocks.size();block_id++){
                const std::vector<Point>& block_data = blocks[block_id].block_data_;
                size_t written = 0;
                while(written < block_data.size()){
                    const size_t here = std::min(block_data.size()-written, geometry_.records_per_page_);

                    // Zero the whole page first: the tail beyond `here` records, and any
                    // dead bytes past the last record slot, must be deterministic.
                    std::memset(page.data(), 0, geometry_.page_bytes_);
                    for(size_t r=0;r<here;r++)
                        EncodeRecord(page.data() + r*geometry_.record_bytes_,
                                     block_data[written+r], geometry_.record_bytes_);

                    WriteFully(page.data(), geometry_.page_bytes_);
                    written += here;
                }
            }

            //5. Flush, so the preads below see the data rather than racing the writeback.
            if(fsync(fd_) != 0)
                throw std::runtime_error("PagedDiskBackend::Build: fsync failed for "+blockstore_filename_+
                                         ": "+std::string(strerror(errno)));

            //6. Unlink the file immediately, while keeping fd_ open.
            //
            // unlink() removes the directory entry, not the file: POSIX keeps the
            // inode and its data blocks alive for as long as any process holds an
            // open descriptor. Reads through fd_ carry on working, but the file now
            // has no name, so the kernel reclaims its space when the process exits
            // -- normally, by exception, by OOM kill, or by the Slurm walltime
            // reaper alike. That is the point: cleanup no longer depends on a
            // destructor running, and a killed task in a 5000-job array can no
            // longer strand a scratch file. (Consequence: `ls` and `du` show
            // nothing while `df` still counts the space, which looks like a leak
            // and is not. Set KEEP_BLOCKSTORE_FILES=1 to keep the name for
            // debugging.)
            if(!KeepBlockstoreFiles()){
                if(std::remove(blockstore_filename_.c_str()) != 0)
                    throw std::runtime_error("PagedDiskBackend::Build: cannot unlink "+blockstore_filename_+
                                             ": "+std::string(strerror(errno)));
                unlinked_ = true;
            }

            built_ = true;
        }

        void Scan(Query& query, const std::vector<size_t>& block_ids, std::vector<Point>& result_vec) override {
            if(!built_)
                throw std::runtime_error("PagedDiskBackend::Scan: Build() has not run");

            if(scratch_page_.size() != geometry_.page_bytes_)
                scratch_page_.resize(geometry_.page_bytes_);

            for(const size_t& block_id: block_ids){
                size_t remaining = block_point_counts_[block_id];
                if(!remaining)
                    continue;

                uint64_t page_id = first_page_[block_id];
                while(remaining){
                    const size_t here = std::min(remaining, geometry_.records_per_page_);
                    const char* page = FetchPage(page_id);

                    for(size_t r=0;r<here;r++){
                        Point point = DecodeRecord(page + r*geometry_.record_bytes_);
                        if(query.CheckPointWithin(point))
                            result_vec.push_back(point);
                    }

                    remaining -= here;
                    page_id++;
                }
            }
        }

        const StorageStats& Stats() const override { return stats_; }
        void ResetStats() override { stats_ = StorageStats{}; }
        const char* Name() const override { return "paged"; }

        // ---- inspection surface, used by the tests and the file dumper ----

        bool IsBuilt() const { return built_; }
        const PageGeometry& Geometry() const { return geometry_; }
        const std::string& Filename() const { return blockstore_filename_; }
        uint64_t FirstPage(size_t block_id) const { return first_page_[block_id]; }
        uint32_t PageCount(size_t block_id) const { return page_count_[block_id]; }
        uint64_t TotalDataPages() const { return total_data_pages_; }
        uint64_t TotalPoints() const { return total_points_; }
        uint64_t FileBytes() const { return file_bytes_; }
        size_t BlockCount() const { return block_point_counts_.size(); }
        /** Bytes of directory metadata that must stay resident to find a page. */
        size_t DirectoryBytes() const {
            return first_page_.size()*sizeof(uint64_t) + page_count_.size()*sizeof(uint32_t);
        }

        /** Read page 0 back off the disk and return it. */
        PagedStoreHeader ReadHeader() const {
            std::vector<char> page(geometry_.page_bytes_);
            ReadPageRaw(0, page.data());
            PagedStoreHeader header{};
            std::memcpy(&header, page.data(), sizeof(header));
            return header;
        }

        /** Read one page into `dst`, which must hold at least page_bytes_ bytes. */
        void ReadPageRaw(uint64_t page_id, char* dst) const {
            const off_t offset = off_t(page_id)*off_t(geometry_.page_bytes_);
            size_t done = 0;
            while(done < geometry_.page_bytes_){
                const ssize_t n = pread(fd_, dst+done, geometry_.page_bytes_-done, offset+off_t(done));
                if(n < 0){
                    if(errno == EINTR) continue;
                    throw std::runtime_error("PagedDiskBackend: pread failed on page "+std::to_string(page_id)+
                                             ": "+std::string(strerror(errno)));
                }
                if(n == 0)
                    throw std::runtime_error("PagedDiskBackend: short read on page "+std::to_string(page_id));
                done += size_t(n);
            }
        }

    private:
        /**
         * Hand back the contents of one page.
         *
         * Phase 2 reads it straight off the disk every time -- no cache, no
         * residency, no accounting. Phase 3 replaces the body with a pool pin and
         * starts recording hits and misses in stats_; the call site in Scan() does
         * not change.
         */
        const char* FetchPage(uint64_t page_id){
            ReadPageRaw(page_id, scratch_page_.data());
            return scratch_page_.data();
        }

        void WriteFully(const char* src, size_t bytes){
            size_t done = 0;
            while(done < bytes){
                const ssize_t n = write(fd_, src+done, bytes-done);
                if(n < 0){
                    if(errno == EINTR) continue;
                    std::remove(blockstore_filename_.c_str());
                    throw std::runtime_error("PagedDiskBackend::Build: write failed on "+blockstore_filename_+
                                             ": "+std::string(strerror(errno)));
                }
                done += size_t(n);
            }
        }

        PageGeometry geometry_;

        int fd_{-1};
        std::string blockstore_filename_;
        bool built_{false};
        bool unlinked_{false};

        std::vector<size_t>   block_point_counts_;   // copied at Build; needed to size the last page
        std::vector<uint64_t> first_page_;           // block -> first page  (the block directory)
        std::vector<uint32_t> page_count_;           // block -> pages spanned

        uint64_t total_data_pages_{};
        uint64_t total_points_{};
        uint64_t file_bytes_{};

        std::vector<char> scratch_page_;
        StorageStats stats_;
};


#endif
