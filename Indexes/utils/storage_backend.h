#ifndef STORAGE_BACKEND_H
#define STORAGE_BACKEND_H

/**
 * @file storage_backend.h
 * @brief Storage seam for the block store.
 *
 * `BlockStore` keeps per-block metadata (MBRs, point counts) in memory always,
 * but the point data itself can be served from several places: the in-memory
 * block vector, a memory-mapped file, or (later) a manually managed buffer
 * pool. `PointStorageBackend` is the one interface every index reaches storage
 * through, so a new strategy is a new subclass rather than another arm of a
 * runtime branch.
 */

#include <vector>
#include <cstdint>

#include"point.h"
#include"query.h"


/**
 * In-memory representation of one block/page of points.
 */
class Block{
    public:
        std::vector<Point> block_data_;


        /** Materialize a block from an existing point vector. */
        Block(std::vector<Point> & data):block_data_(data){}

        /** Materialize a block from an iterator range. */
        Block(const std::vector<Point>::iterator & it_data_begin, const std::vector<Point>::iterator & it_data_end){
            // block_data_.reserve(BLOCK_SIZE+2);
            // assert(std::distance(it_data_begin,it_data_end)<=BLOCK_SIZE);
            block_data_.insert(block_data_.begin(),it_data_begin,it_data_end);
        }

        Block(){}

        /** Replace the block contents with a new iterator range. */
        void AssignPoints(std::vector<Point>::iterator it_data_begin,std::vector<Point>::iterator it_data_end){
            block_data_.assign(it_data_begin,it_data_end);
        }

        /** Append points that satisfy `query` into `result_vec`. */
        void FilterPointsForQuery(Query &query, std::vector<Point>& result_vec) const {
            size_t block_size_here = block_data_.size();
            for(auto& pnt: block_data_)
                if(query.CheckPointWithin(pnt))
                    result_vec.push_back(pnt);
        }

        /** Copy the full block contents into the caller-owned result buffer. */
        void CopyAllPointsIntoResult(std::vector<Point>& result_vec){
            result_vec.insert(result_vec.end(),block_data_.begin(),block_data_.end());
        }

        /** Append a single point to the block. */
        void InsertPoint(Point& pnt){
            block_data_.push_back(pnt);
        }

        /** Return the number of points currently stored in the block. */
        size_t BlockDataSize(){
            return block_data_.size();
        }
};


/** Which representation a `BlockStore` currently serves point data from. */
enum class StorageMode { kInMemory, kMmap, kBufferPool };


/**
 * Per-backend I/O accounting.
 *
 * The in-memory and mmap backends leave these at zero — neither can observe
 * its own I/O, which is precisely the gap the buffer-pool backend closes.
 */
struct StorageStats{
    uint64_t blocks_scanned  = 0;   // block-fetch requests, as issued by the index
    uint64_t points_decoded  = 0;
    uint64_t pages_requested = 0;   // logical pin calls == page_hits + page_misses
    uint64_t page_hits       = 0;
    uint64_t page_misses     = 0;   // == number of pread() calls actually issued
    uint64_t bytes_read      = 0;
    uint64_t evictions       = 0;

    double HitRate() const { return pages_requested ? double(page_hits)/double(pages_requested) : 0.0; }
};


/**
 * One way of storing and scanning the point data behind a `BlockStore`.
 *
 * `Build` is called once, from `BlockStore::FinishedConstruction()`, while the
 * in-memory blocks are still resident. After that the store is read-only, so
 * `Scan` is the whole read surface.
 */
class PointStorageBackend{
    public:
        virtual ~PointStorageBackend() = default;

        /** Materialize this backend's representation from the in-memory blocks. */
        virtual void Build(const std::vector<Block>& blocks, const std::vector<size_t>& counts) = 0;

        /** Append points of `block_ids` that satisfy `query` into `result_vec`. */
        virtual void Scan(Query& query, const std::vector<size_t>& block_ids, std::vector<Point>& result_vec) = 0;

        virtual const StorageStats& Stats() const = 0;
        virtual void ResetStats() = 0;
        virtual const char* Name() const = 0;
};


/**
 * Serves points straight out of the `BlockStore`'s own block vector.
 *
 * Holds back-pointers rather than copies. That is safe because `BlockStore` is
 * neither copyable nor movable, so the vector objects it owns never relocate.
 */
class InMemoryBackend: public PointStorageBackend{
    public:
        InMemoryBackend(const std::vector<Block>& blocks, const std::vector<size_t>& counts)
            : blocks_(&blocks), counts_(&counts) {}

        /** Nothing to materialize — the blocks are already the representation. */
        void Build(const std::vector<Block>&, const std::vector<size_t>&) override {}

        void Scan(Query& query, const std::vector<size_t>& block_ids, std::vector<Point>& result_vec) override {
            for(const size_t& block_id: block_ids){
                if((*counts_)[block_id])
                    (*blocks_)[block_id].FilterPointsForQuery(query,result_vec);
            }
        }

        const StorageStats& Stats() const override { return stats_; }
        void ResetStats() override { stats_ = StorageStats{}; }
        const char* Name() const override { return "in-memory"; }

    private:
        const std::vector<Block>*  blocks_;
        const std::vector<size_t>* counts_;
        StorageStats stats_;
};


#endif
