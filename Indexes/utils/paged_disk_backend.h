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
 * Page accesses go through a BufferPool with a fixed frame budget, so the
 * memory available to a scan is something we set rather than something the
 * kernel decides, and every access is counted as a hit or a miss.
 *
 * The pool reads through its own descriptor, separate from the one the writer
 * used. With BUFFER_POOL_DIRECT_IO=1 that descriptor is opened O_DIRECT, which
 * takes the OS page cache out of the read path: a miss becomes a real device
 * transfer rather than a memcpy out of the kernel's own cache, and this pool
 * becomes the only cache in the stack instead of the upper of two. Nothing
 * about the pool itself changes -- same frames, same LRU, same counters. Only
 * what a miss costs changes.
 *
 * `ReadPageRaw()` remains as an uncached path that bypasses the pool entirely.
 * The tests use it to read the file *without* going through the thing under
 * test, which is what makes a layout bug distinguishable from a caching bug.
 */

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <numeric>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include <algorithm>
#include <cmath>
#include <memory>

#include"storage_backend.h"
#include"page_layout.h"
#include"buffer_pool.h"
#include"sort_tools.h"


/**
 * Lower bound on the frame budget.
 *
 * `kBlock` reserves enough frames to hold the largest single block, so that a
 * block fetch is atomic. That is a modelling choice, not a correctness one --
 * Scan() pins one page at a time, so the pool is correct with a single frame.
 *
 * `kMinimal` exists because FLOOD and GRID create one block per grid cell with
 * no size cap, so on skewed data `largest_block_pages` can be a large slice of
 * the whole file. Under kBlock that would quietly turn a requested fraction of
 * 0.001 into an effective 0.4 for those indexes alone. Whichever mode is used,
 * FramesFloored() and EffectiveFraction() record what actually happened.
 */
enum class BufferPoolFloorMode { kBlock, kMinimal };


/** Frame budget and eviction settings for the paged backend's pool. */
struct BufferPoolConfig{
    double fraction{1.0};                                    // of the store's own file size
    BufferPoolFloorMode floor_mode{BufferPoolFloorMode::kBlock};
    size_t slack_frames{2};
    std::string policy{"LRU"};
};


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

/**
 * True when the pool's page reads should bypass the OS page cache.
 *
 * Off by default, so an unset environment reads exactly as it always did. When
 * on, the pool's descriptor is opened O_DIRECT and every miss is a device
 * read. See the note at the top of this file for what does and does not change.
 */
inline bool DirectIoEnabled(){
    const char* flag = std::getenv("BUFFER_POOL_DIRECT_IO");
    return flag != nullptr && std::string(flag) == "1";
}

/** Read a size_t from the environment, falling back to `fallback` when unset or unparsable. */
inline size_t EnvSizeT(const char* name, size_t fallback){
    const char* raw = std::getenv(name);
    if(raw == nullptr || std::string(raw).empty())
        return fallback;
    try { return size_t(std::stoull(raw)); }
    catch(const std::exception&){
        throw std::runtime_error(std::string(name)+": cannot parse '"+raw+"' as a number");
    }
}

/**
 * Page geometry for this run, from PAGE_BYTES / RECORD_BYTES.
 *
 * Defaults reproduce the built-in geometry, so an unset environment behaves
 * exactly as the constants do.
 */
inline PageGeometry PagedGeometryFromEnv(){
    return PageGeometry(EnvSizeT("PAGE_BYTES", kDefaultPageBytes),
                        EnvSizeT("RECORD_BYTES", kDefaultRecordBytes));
}


class PagedDiskBackend: public PointStorageBackend{
    public:
        /**
         * A pinned page, unpinned automatically when it goes out of scope.
         *
         * Pairing Pin/Unpin by hand is the kind of mistake that surfaces far from
         * its cause: a leaked pin slowly starves the pool of victims until some
         * later Pin throws "every frame is pinned". The guard makes the pairing
         * impossible to get wrong, including when an exception unwinds mid-decode.
         */
        class PinnedPage{
            public:
                PinnedPage(BufferPool* pool, uint64_t page_id, const char* data)
                    : pool_(pool), page_id_(page_id), data_(data) {}

                PinnedPage(PinnedPage&& other) noexcept
                    : pool_(other.pool_), page_id_(other.page_id_), data_(other.data_){
                    other.pool_ = nullptr;
                }

                PinnedPage(const PinnedPage&) = delete;
                PinnedPage& operator=(const PinnedPage&) = delete;

                ~PinnedPage(){ if(pool_) pool_->Unpin(page_id_); }

                const char* Data() const { return data_; }

            private:
                BufferPool* pool_;
                uint64_t page_id_;
                const char* data_;
        };

        explicit PagedDiskBackend(PageGeometry geometry = PageGeometry{},
                                  BufferPoolConfig pool_config = BufferPoolConfig{})
            : geometry_(geometry), pool_config_(pool_config) {}

        // Owns a file descriptor and a scratch file; copying would double-close both.
        PagedDiskBackend(const PagedDiskBackend&) = delete;
        PagedDiskBackend& operator=(const PagedDiskBackend&) = delete;

        ~PagedDiskBackend() override {
            if(read_fd_ >= 0)
                close(read_fd_);
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
            largest_block_pages_ = 0;
            for(size_t block_id=0;block_id<counts.size();block_id++){
                const size_t pages = geometry_.PagesForBlock(counts[block_id]);
                first_page_[block_id] = next_page;
                page_count_[block_id] = uint32_t(pages);
                next_page += pages;
                largest_block_pages_ = std::max(largest_block_pages_, pages);
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

            //5. Flush, so the reads below see the data rather than racing the writeback.
            if(fsync(fd_) != 0)
                throw std::runtime_error("PagedDiskBackend::Build: fsync failed for "+blockstore_filename_+
                                         ": "+std::string(strerror(errno)));

            //6. Open the descriptor the pool will read through.
            //
            // Separate from fd_ because the two halves want opposite things. The
            // writer above streams whole pages sequentially, once, outside any
            // measurement -- it is perfectly happy going through the page cache,
            // and its buffer is not aligned for O_DIRECT anyway. The reader wants
            // each miss to be a real device transfer. Doing this before the unlink
            // below is not optional: afterwards there is no name left to open.
            OpenReadDescriptor();

            //7. Drop the pages the writer just left behind in the OS cache.
            //
            // They are a second copy of the whole store sitting in kernel memory.
            // Harmless to correctness -- direct reads ignore them either way --
            // but they occupy RAM that the memory-budget claim says we are not
            // using. Advisory by definition, so a refusal here is not an error.
            posix_fadvise(fd_, 0, 0, POSIX_FADV_DONTNEED);

            //8. Unlink the file immediately, while keeping the descriptors open.
            //
            // unlink() removes the directory entry, not the file: POSIX keeps the
            // inode and its data blocks alive for as long as any process holds an
            // open descriptor. Reads through read_fd_ carry on working, but the file now
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

            //9. Size and open the buffer pool over the file we just wrote.
            RebuildPool(pool_config_);
        }

        /**
         * Re-open the pool with a different budget, leaving the file alone.
         *
         * Sweeping the fraction only changes how much may be resident, never the
         * bytes on disk -- so a fraction sweep costs one pool rebuild per point,
         * not a rewrite of the store.
         */
        void RebuildPool(const BufferPoolConfig& config){
            if(!built_)
                throw std::runtime_error("PagedDiskBackend::RebuildPool: Build() has not run");
            if(read_fd_ < 0)
                throw std::runtime_error("PagedDiskBackend::RebuildPool: no read descriptor");

            pool_config_ = config;
            const uint64_t total_pages = total_data_pages_ + 1;

            uint64_t requested = uint64_t(std::floor(config.fraction*double(file_bytes_)/
                                                     double(geometry_.page_bytes_)));
            const uint64_t floor_frames =
                (config.floor_mode == BufferPoolFloorMode::kBlock)
                    ? uint64_t(largest_block_pages_) + uint64_t(config.slack_frames)
                    : uint64_t(2);

            uint64_t frames = requested;
            if(frames < floor_frames) frames = floor_frames;
            if(frames > total_pages)  frames = total_pages;
            if(frames < 1)            frames = 1;

            frames_floored_ = (frames > requested);
            effective_fraction_ = double(frames)*double(geometry_.page_bytes_)/double(file_bytes_);

            pool_ = std::unique_ptr<BufferPool>(
                new BufferPool(read_fd_, geometry_.page_bytes_, size_t(frames),
                               MakeReplacementPolicy(config.policy)));
            stats_ = StorageStats{};
        }

        /** Evict everything, keeping the frames. Use before a cold-pass measurement. */
        void ClearCache(){
            if(pool_) pool_->Clear();
        }

        void Scan(Query& query, const std::vector<size_t>& block_ids, std::vector<Point>& result_vec) override {
            if(!built_ || !pool_)
                throw std::runtime_error("PagedDiskBackend::Scan: Build() has not run");

            for(const size_t& block_id: block_ids){
                size_t remaining = block_point_counts_[block_id];
                if(!remaining)
                    continue;
                stats_.blocks_scanned++;

                uint64_t page_id = first_page_[block_id];
                while(remaining){
                    const size_t here = std::min(remaining, geometry_.records_per_page_);
                    PinnedPage page = FetchPage(page_id);   // unpinned when it leaves scope

                    for(size_t r=0;r<here;r++){
                        Point point = DecodeRecord(page.Data() + r*geometry_.record_bytes_);
                        if(query.CheckPointWithin(point))
                            result_vec.push_back(point);
                    }

                    stats_.points_decoded += here;
                    remaining -= here;
                    page_id++;
                }
            }
        }

        /** Backend counters, with the pool's page accounting merged in. */
        const StorageStats& Stats() const override {
            if(pool_){
                const BufferPoolStats& pool_stats = pool_->Stats();
                stats_.pages_requested = pool_stats.pages_requested;
                stats_.page_hits       = pool_stats.page_hits;
                stats_.page_misses     = pool_stats.page_misses;
                stats_.bytes_read      = pool_stats.bytes_read;
                stats_.evictions       = pool_stats.evictions;
            }
            return stats_;
        }

        void ResetStats() override {
            stats_ = StorageStats{};
            if(pool_) pool_->ResetStats();
        }

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
        size_t LargestBlockPages() const { return largest_block_pages_; }
        /** True when the pool's misses bypass the OS page cache. */
        bool DirectIo() const { return direct_io_; }

        // ---- buffer-pool budget, all of it worth logging alongside the counters ----

        size_t PoolFrames() const { return pool_ ? pool_->FrameCount() : 0; }
        size_t PoolBytes() const { return PoolFrames()*geometry_.page_bytes_; }
        double RequestedFraction() const { return pool_config_.fraction; }
        /** What the pool actually got, after the floor and ceiling were applied. */
        double EffectiveFraction() const { return effective_fraction_; }
        /** True when the floor overrode the requested fraction -- flag these points in plots. */
        bool FramesFloored() const { return frames_floored_; }
        const char* PolicyName() const { return pool_ ? pool_->PolicyName() : "none"; }
        const BufferPool* Pool() const { return pool_.get(); }
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
         * Open the descriptor the buffer pool reads through.
         *
         * O_DIRECT carries a three-part alignment contract -- memory buffer, file
         * offset and transfer length -- and the required alignment belongs to the
         * filesystem, not to the device. Lustre insists on 4096 and returns EINVAL
         * below it; local XFS over a 512e NVMe accepts 512. Rather than guess, we
         * require a page size that is a multiple of 4096 and then verify with one
         * real read.
         *
         * The 4096 requirement also keeps the pool's frames aligned: its slab is
         * allocated with posix_memalign(4096) but frame i sits at i*page_bytes, so
         * only a 4096-multiple page size makes every frame a legal target.
         *
         * Probing here rather than on first use means a filesystem that refuses
         * direct I/O fails during construction, with a message naming the
         * directory, instead of somewhere in the middle of a 5000-task sweep.
         */
        void OpenReadDescriptor(){
            direct_io_ = DirectIoEnabled();

            if(direct_io_ && geometry_.page_bytes_ % 4096 != 0)
                throw std::runtime_error(
                    "PagedDiskBackend: BUFFER_POOL_DIRECT_IO=1 requires PAGE_BYTES to be a multiple "
                    "of 4096, got "+std::to_string(geometry_.page_bytes_));

            int flags = O_RDONLY;
            if(direct_io_){
#ifdef O_DIRECT
                flags |= O_DIRECT;
#else
                throw std::runtime_error("PagedDiskBackend: O_DIRECT is not available on this platform");
#endif
            }

            read_fd_ = open(blockstore_filename_.c_str(), flags);
            if(read_fd_ < 0)
                throw std::runtime_error("PagedDiskBackend::Build: cannot reopen "+blockstore_filename_+
                                         " for reading: "+std::string(strerror(errno)));

            if(direct_io_)
                ProbeDirectRead();
        }

        /** One aligned read, so an unsupported filesystem fails here and says why. */
        void ProbeDirectRead(){
            void* buffer = nullptr;
            if(posix_memalign(&buffer, 4096, geometry_.page_bytes_) != 0 || buffer == nullptr)
                throw std::runtime_error("PagedDiskBackend: cannot allocate an aligned probe buffer");

            const ssize_t n = pread(read_fd_, buffer, geometry_.page_bytes_, 0);
            const int probe_errno = errno;
            std::free(buffer);

            if(n != ssize_t(geometry_.page_bytes_))
                throw std::runtime_error(
                    "PagedDiskBackend: an O_DIRECT read of "+std::to_string(geometry_.page_bytes_)+
                    " bytes failed under "+ResolveBlockstoreDir()+" ("+std::string(strerror(probe_errno))+
                    "). That filesystem will not serve direct I/O at this page size -- Lustre and NFS "
                    "commonly refuse anything below 4096. Point TEMP_BLOCKSTORE_DIR at node-local "
                    "storage, or clear BUFFER_POOL_DIRECT_IO to read through the page cache.");
        }

        /** Make one page resident and hand it back, pinned for the guard's lifetime. */
        PinnedPage FetchPage(uint64_t page_id){
            return PinnedPage(pool_.get(), page_id, pool_->Pin(page_id));
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

        int fd_{-1};                     // the writer's descriptor; buffered, sequential
        int read_fd_{-1};                // the pool's descriptor; O_DIRECT when enabled
        bool direct_io_{false};
        std::string blockstore_filename_;
        bool built_{false};
        bool unlinked_{false};

        std::vector<size_t>   block_point_counts_;   // copied at Build; needed to size the last page
        std::vector<uint64_t> first_page_;           // block -> first page  (the block directory)
        std::vector<uint32_t> page_count_;           // block -> pages spanned

        uint64_t total_data_pages_{};
        uint64_t total_points_{};
        uint64_t file_bytes_{};
        size_t largest_block_pages_{};

        BufferPoolConfig pool_config_;
        std::unique_ptr<BufferPool> pool_;
        bool frames_floored_{false};
        double effective_fraction_{0.0};

        mutable StorageStats stats_;   // mutable: Stats() folds in the pool's counters
};


#endif
