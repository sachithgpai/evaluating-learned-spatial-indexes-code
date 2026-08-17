#ifndef BUFFER_POOL_H
#define BUFFER_POOL_H

/**
 * @file buffer_pool.h
 * @brief A manually managed page cache over a paged store file.
 *
 * The point of doing this by hand rather than leaning on the OS page cache is
 * that we get to set the memory budget and, more importantly, to *count* what
 * happens: every page access is either a hit or a miss, and the miss count is
 * deterministic and machine-independent in a way a wall-clock number never is.
 *
 * IMPORTANT -- what is deliberately absent. The paged store is read-only after
 * it is built, so this pool has no dirty bit, no write-back on eviction, no
 * flush, no checkpoint and no page latches. Evicting a page is an erase from
 * the page table and nothing else. If you came here looking for the other half
 * of a textbook buffer pool, it genuinely does not exist rather than having
 * been forgotten.
 *
 * Equally important -- what a "miss" means. The file was written moments ago
 * and is usually still in the OS page cache, so a miss here is normally an
 * in-RAM copy rather than a device read. These are LOGICAL misses. That is the
 * intent: logical misses reproduce exactly across machines, physical ones do
 * not. Do not read `page_misses` as device I/O.
 */

#include <unistd.h>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>


struct BufferPoolStats{
    uint64_t pages_requested = 0;   // Pin() calls == page_hits + page_misses
    uint64_t page_hits       = 0;
    uint64_t page_misses     = 0;   // identically the number of pread() calls issued
    uint64_t bytes_read      = 0;
    uint64_t evictions       = 0;

    double HitRate() const { return pages_requested ? double(page_hits)/double(pages_requested) : 0.0; }
};


/**
 * Chooses which frame to reuse when the pool is full.
 *
 * The policy tracks exactly the *unpinned, occupied* frames -- the eviction
 * candidates. Frames that were never used are the pool's free list, and pinned
 * frames are nobody's candidate.
 *
 * Note there is no OnEvict hook: Victim() unlinks the frame before returning
 * it, so there is no second step to forget or to perform twice.
 */
class ReplacementPolicy{
    public:
        virtual ~ReplacementPolicy() = default;

        /** Forget everything and size the internal tables for `frame_count` frames. */
        virtual void Reset(size_t frame_count) = 0;

        /** `frame` became pinned: it leaves the candidate set. */
        virtual void OnPin(size_t frame) = 0;

        /** `frame`'s pin count reached zero: it rejoins the candidate set as most-recent. */
        virtual void OnUnpin(size_t frame) = 0;

        /** Take the next frame to reuse. False when every frame is pinned. */
        virtual bool Victim(size_t& frame) = 0;

        virtual const char* Name() const = 0;
};


/**
 * Least-recently-used, as an intrusive doubly-linked list over frame indices.
 *
 * Two index arrays plus a membership flag; head is the LRU end (the next
 * victim), tail is the MRU end. Every operation is O(1) and allocates nothing
 * after Reset() -- which matters, because this bookkeeping runs on every page
 * access and any allocation here would land inside the measurement we are
 * trying to take.
 */
class LruPolicy: public ReplacementPolicy{
    public:
        void Reset(size_t frame_count) override {
            prev_.assign(frame_count, kNil);
            next_.assign(frame_count, kNil);
            in_list_.assign(frame_count, false);
            head_ = kNil;
            tail_ = kNil;
        }

        void OnPin(size_t frame) override {
            if(in_list_[frame])
                Unlink(frame);
        }

        void OnUnpin(size_t frame) override {
            if(!in_list_[frame])
                PushMostRecent(frame);
        }

        bool Victim(size_t& frame) override {
            if(head_ == kNil)
                return false;
            frame = head_;
            Unlink(frame);
            return true;
        }

        const char* Name() const override { return "LRU"; }

    private:
        static constexpr uint32_t kNil = UINT32_MAX;

        void Unlink(size_t frame){
            const uint32_t f = uint32_t(frame);
            if(prev_[f] != kNil) next_[prev_[f]] = next_[f]; else head_ = next_[f];
            if(next_[f] != kNil) prev_[next_[f]] = prev_[f]; else tail_ = prev_[f];
            prev_[f] = kNil;
            next_[f] = kNil;
            in_list_[f] = false;
        }

        void PushMostRecent(size_t frame){
            const uint32_t f = uint32_t(frame);
            prev_[f] = tail_;
            next_[f] = kNil;
            if(tail_ != kNil) next_[tail_] = f; else head_ = f;
            tail_ = f;
            in_list_[f] = true;
        }

        std::vector<uint32_t> prev_, next_;
        std::vector<bool> in_list_;
        uint32_t head_{kNil}, tail_{kNil};
};


/** Construct a policy by name. Unknown names are an error, not a silent fallback. */
inline std::unique_ptr<ReplacementPolicy> MakeReplacementPolicy(const std::string& name){
    if(name == "LRU" || name == "lru")
        return std::unique_ptr<ReplacementPolicy>(new LruPolicy());
    throw std::runtime_error("MakeReplacementPolicy: unknown policy '"+name+"'");
}


class BufferPool{
    public:
        /**
         * @param fd          descriptor of the paged file; the pool does not own or close it
         * @param page_bytes  size of one frame, matching the file's page size
         * @param frame_count how many pages may be resident at once -- the budget
         */
        BufferPool(int fd, size_t page_bytes, size_t frame_count,
                   std::unique_ptr<ReplacementPolicy> policy)
            : fd_(fd), page_bytes_(page_bytes), frame_count_(frame_count), policy_(std::move(policy)){

            if(frame_count_ == 0)
                throw std::runtime_error("BufferPool: needs at least one frame");

            // Align the slab to a page boundary so switching to O_DIRECT later is a
            // one-flag change rather than a rewrite of the allocation path.
            void* slab = nullptr;
            if(posix_memalign(&slab, 4096, frame_count_*page_bytes_) != 0 || slab == nullptr)
                throw std::runtime_error("BufferPool: cannot allocate "+
                                         std::to_string(frame_count_*page_bytes_)+" bytes of frames");
            frames_ = static_cast<char*>(slab);

            frame_page_.assign(frame_count_, kInvalidPage);
            pin_count_.assign(frame_count_, 0);
            page_table_.reserve(frame_count_*2);

            free_frames_.reserve(frame_count_);
            for(size_t f=frame_count_;f-- > 0;)
                free_frames_.push_back(f);

            policy_->Reset(frame_count_);
        }

        // Owns a raw slab; copying would double-free it.
        BufferPool(const BufferPool&) = delete;
        BufferPool& operator=(const BufferPool&) = delete;

        ~BufferPool(){ std::free(frames_); }

        /**
         * Make `page_id` resident and return a pointer to its frame.
         *
         * The frame stays valid until the matching Unpin(). Callers should go
         * through PagedDiskBackend's RAII guard rather than pairing these by hand.
         */
        const char* Pin(uint64_t page_id){
            stats_.pages_requested++;

            auto it = page_table_.find(page_id);
            if(it != page_table_.end()){
                const size_t frame = it->second;
                if(pin_count_[frame]++ == 0)
                    policy_->OnPin(frame);
                stats_.page_hits++;
                return frames_ + frame*page_bytes_;
            }

            const size_t frame = ClaimFrame();
            ReadPage(page_id, frames_ + frame*page_bytes_);

            page_table_[page_id] = frame;
            frame_page_[frame] = page_id;
            pin_count_[frame] = 1;
            policy_->OnPin(frame);

            stats_.page_misses++;
            stats_.bytes_read += page_bytes_;
            return frames_ + frame*page_bytes_;
        }

        void Unpin(uint64_t page_id){
            auto it = page_table_.find(page_id);
            if(it == page_table_.end())
                throw std::runtime_error("BufferPool::Unpin: page "+std::to_string(page_id)+" is not resident");

            const size_t frame = it->second;
            if(pin_count_[frame] == 0)
                throw std::runtime_error("BufferPool::Unpin: page "+std::to_string(page_id)+" is not pinned");

            if(--pin_count_[frame] == 0)
                policy_->OnUnpin(frame);
        }

        /**
         * Evict everything, keeping the frames allocated.
         *
         * This is the cold-pass reset: residency goes back to empty without the
         * cost or the timing perturbation of reallocating the slab.
         */
        void Clear(){
            for(size_t f=0;f<frame_count_;f++){
                if(pin_count_[f] != 0)
                    throw std::runtime_error("BufferPool::Clear: frame "+std::to_string(f)+" is still pinned");
                frame_page_[f] = kInvalidPage;
            }
            page_table_.clear();
            free_frames_.clear();
            for(size_t f=frame_count_;f-- > 0;)
                free_frames_.push_back(f);
            policy_->Reset(frame_count_);
        }

        void ResetStats(){ stats_ = BufferPoolStats{}; }

        const BufferPoolStats& Stats() const { return stats_; }
        size_t FrameCount() const { return frame_count_; }
        size_t PageBytes() const { return page_bytes_; }
        size_t ResidentPages() const { return page_table_.size(); }
        const char* PolicyName() const { return policy_->Name(); }
        bool IsResident(uint64_t page_id) const { return page_table_.count(page_id) != 0; }

    private:
        static constexpr uint64_t kInvalidPage = UINT64_MAX;

        /** A frame to load into: a never-used one if possible, otherwise a victim. */
        size_t ClaimFrame(){
            if(!free_frames_.empty()){
                const size_t frame = free_frames_.back();
                free_frames_.pop_back();
                return frame;
            }

            size_t frame = 0;
            if(!policy_->Victim(frame))
                throw std::runtime_error("BufferPool: every frame is pinned, cannot make room "
                                         "(frames="+std::to_string(frame_count_)+")");

            page_table_.erase(frame_page_[frame]);
            frame_page_[frame] = kInvalidPage;
            stats_.evictions++;
            return frame;
        }

        void ReadPage(uint64_t page_id, char* dst){
            const off_t offset = off_t(page_id)*off_t(page_bytes_);
            size_t done = 0;
            while(done < page_bytes_){
                const ssize_t n = pread(fd_, dst+done, page_bytes_-done, offset+off_t(done));
                if(n < 0){
                    if(errno == EINTR) continue;
                    throw std::runtime_error("BufferPool: pread failed on page "+std::to_string(page_id)+
                                             ": "+std::string(strerror(errno)));
                }
                if(n == 0)
                    throw std::runtime_error("BufferPool: short read on page "+std::to_string(page_id));
                done += size_t(n);
            }
        }

        int fd_;
        size_t page_bytes_;
        size_t frame_count_;
        std::unique_ptr<ReplacementPolicy> policy_;

        char* frames_{nullptr};
        std::vector<uint64_t> frame_page_;
        std::vector<uint32_t> pin_count_;
        std::unordered_map<uint64_t,size_t> page_table_;
        std::vector<size_t> free_frames_;

        BufferPoolStats stats_;
};


#endif
