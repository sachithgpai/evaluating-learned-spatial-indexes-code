#ifndef LOCALMODEL_H
#define LOCALMODEL_H


#include <algorithm>
#include <numeric>
#include <cassert>
#include <memory>

#include"point.h"
#include"query.h"
#include"sort_tools.h"
#include"storage_backend.h"
#include"mmap_backend.h"
#include"paged_disk_backend.h"


/**
 * Shared block store used by multiple index implementations.
 *
 * It tracks per-block MBR metadata in memory always, and delegates the point
 * data itself to a `PointStorageBackend` (see storage_backend.h). Reads go
 * through `FilterPointsFromBlocksForQuery`; everything else here is build-time
 * or metadata.
 */
class BlockStore{
    public:
        std::vector<Block> block_list_;                                 // List of block objects
        std::vector<BoundingRectangle> block_mbrs_;                     // MBRs for each block. Metadata that is usually stored in memory.
        std::vector<size_t> block_point_counts_;                        // Block sizes.

        BlockStore(const std::string &out_filename="")
            : mem_backend_(new InMemoryBackend(block_list_, block_point_counts_)){
            active_ = mem_backend_.get();
        }

        // Backends own a mapping and a temp file, and InMemoryBackend points back at
        // our own vectors — so a BlockStore must never be copied or relocated.
        BlockStore(const BlockStore&) = delete;
        BlockStore& operator=(const BlockStore&) = delete;


        /** Create an empty block and return its local identifier. */
        size_t CreateEmptyBlock(){
            block_list_.emplace_back();
            block_mbrs_.emplace_back();
            block_point_counts_.push_back(0);
            return block_list_.size()-1;
        }

        /** Insert a point into an existing block and return the new block size. */
        size_t InsertNewPointInBlock(Point& pnt, size_t local_block_id){

            block_mbrs_[local_block_id].UpdateBoundingBoxWithPoint(pnt);
            block_point_counts_[local_block_id]++;
            // std::cout<<"InsertNewPointInBlock "<<local_block_id<<" "<<block_mbrs_.size()<<" "<<block_point_counts_.size()<<" "<<block_list_.size()<<"\n";

            block_list_[local_block_id].InsertPoint(pnt);
            // std::cout<<"END OF InsertNewPointInBlock"<<"\n";

            return block_point_counts_[local_block_id];
        }

        /** Create a new block from an iterator range and return its local ID. */
        size_t InsertNewBlock(std::vector<Point>::iterator it_data_begin,std::vector<Point>::iterator it_data_end){
            
            size_t curr_block_id = block_mbrs_.size();
            block_mbrs_.emplace_back();
            // assert(std::distance(it_data_begin,it_data_end)<=BLOCK_SIZE);

            for(std::vector<Point>::iterator itr=it_data_begin;itr!=it_data_end;itr++)
                block_mbrs_[curr_block_id].UpdateBoundingBoxWithPoint(*itr);
            

            block_list_.emplace_back(it_data_begin,it_data_end);


            block_point_counts_.push_back(block_list_[curr_block_id].BlockDataSize());
            
            return curr_block_id;
        }

        /** Replace the contents of an existing block and recompute its MBR. */
        BoundingRectangle ReassignPointsInBlock(size_t local_block_id,std::vector<Point>::iterator it_data_begin,std::vector<Point>::iterator it_data_end){
            block_mbrs_[local_block_id].SetToDefault();
            for(std::vector<Point>::iterator itr=it_data_begin;itr!=it_data_end;itr++)
                block_mbrs_[local_block_id].UpdateBoundingBoxWithPoint(*itr);
            
            block_list_[local_block_id].AssignPoints(it_data_begin,it_data_end);
            block_point_counts_[local_block_id]=block_list_[local_block_id].BlockDataSize();

            return block_mbrs_[local_block_id];
        }

        /** Return a materialized copy of the points stored in one block. */
        std::vector<Point> FetchPointsInBlock(size_t local_block_id){
            std::vector<Point> result;
            block_list_[local_block_id].CopyAllPointsIntoResult(result);
            return result;
        }
        /** Scan the supplied blocks and append matching points into `result_vec`. */
        void FilterPointsFromBlocksForQuery(Query &query,std::vector<size_t>& refined_blocks, std::vector<Point>& result_vec){
            active_->Scan(query,refined_blocks,result_vec);
        }

        /** Select which representation subsequent scans read from. */
        void SetStorageMode(StorageMode requested){
            if(requested == mode_)
                return;

            switch(requested){
                case StorageMode::kInMemory:
                    active_ = mem_backend_.get();
                    break;
                case StorageMode::kMmap:
                    assert(mmap_backend_ && "FinishedConstruction() must run before switching to the mmap backend");
                    active_ = mmap_backend_.get();
                    break;
                case StorageMode::kBufferPool:
                    assert(paged_backend_ && "ENABLE_PAGED_BACKEND=1 must be set before FinishedConstruction()");
                    active_ = paged_backend_.get();
                    break;
            }
            mode_ = requested;
        }

        StorageMode CurrentStorageMode() const { return mode_; }

        /** I/O accounting for the active backend. */
        const StorageStats& StorageStatsRef() const { return active_->Stats(); }
        void ResetStorageStats(){ active_->ResetStats(); }

        /** Retain only those local blocks whose MBR overlaps the query. */
        void RefinedBlocksForQueryFromLocalBlocks(Query &query, std::vector<size_t>& local_blocks,std::vector<size_t>& refined_blocks){
            for(size_t& blk_id: local_blocks)
                if(query.IsThereOverlap(block_mbrs_[blk_id]))
                    refined_blocks.push_back(blk_id);
        }
        
        /** Return the cached MBR for one block. */
        BoundingRectangle FetchBoundingBoxForBlock(size_t local_block_id){
            return block_mbrs_[local_block_id];
        }


        /** Sum the point counts of a list of blocks. */
        size_t NumOfPointsInBlocks(std::vector<size_t>& refined_blocks){
            size_t sum_points=0;
            for(auto &block_id: refined_blocks)
                sum_points+=block_point_counts_[block_id];
            return sum_points;
        }

        /** Sum the counts of blocks fully covered by the query rectangle. */
        size_t NumOfPointsCopiedDirectlyForQuery(Query& query){
            size_t points_in_exact_ranges=0;
            for(size_t block_id=0;block_id<block_list_.size();block_id++)
                if(query.IsCompletelyCovering(block_mbrs_[block_id]))
                    points_in_exact_ranges+=block_point_counts_[block_id];
            return points_in_exact_ranges;
        }

        /**
         * Return the number of blocks currently stored.
         *
         * Counted from the metadata rather than from `block_list_`, so the answer
         * survives ReleaseInMemoryBlocks().
         */
        size_t NumOfBlocks() const {
            return block_point_counts_.size();
        }

        /**
         * Free the in-memory point data, keeping only the metadata.
         *
         * Until this runs, a "disk-backed" measurement is taken with the whole
         * dataset also resident in RAM, which makes any memory-budget claim
         * fictional. Call it once the disk backends are built and the in-memory
         * pass is finished.
         *
         * `block_mbrs_` and `block_point_counts_` deliberately stay: those are the
         * index metadata a real system also keeps resident, and the paged backend
         * needs the counts to size each block's final page.
         */
        void ReleaseInMemoryBlocks(){
            if(mode_ == StorageMode::kInMemory)
                return;                                  // would pull the ground out from under the active backend
            if(!mmap_backend_ && !paged_backend_)
                return;                                  // no disk copy exists to fall back on

            // swap-with-empty, not clear(): clear() destroys the elements but keeps the
            // outer vector's capacity allocated.
            std::vector<Block>().swap(block_list_);

            // The mapping's resident pages count against RSS too, so leaving it in
            // place would keep the memory claim false in a way nothing inside the
            // process can see.
            if(mmap_backend_)
                mmap_backend_->Release();

            blocks_released_ = true;
        }

        bool BlocksReleased() const { return blocks_released_; }

        /** Bytes of per-block metadata that stay resident regardless of backend. */
        size_t MetadataBytes() const {
            return block_mbrs_.size()*sizeof(BoundingRectangle) +
                   block_point_counts_.size()*sizeof(size_t);
        }


        /** Print all block sizes for debugging. */
        void PrintBlockSizes(){
            for(auto& blk:block_point_counts_)
                std::cout<<blk<<" ";
            std::cout<<"\n";
        }

        /** Return coarse block-size quantiles plus the arithmetic mean. */
        std::vector<size_t> QuantilesOfBlockSizes(){
            std::vector<size_t> temp_block_point_counts(block_point_counts_);
            std::sort(temp_block_point_counts.begin(),temp_block_point_counts.end());


            size_t block_count = temp_block_point_counts.size();
            std::vector<size_t> quantiles;

            if(block_count == 0)
                return std::vector<size_t>(6, 0);   // keep the array length stable for downstream plotting

            quantiles.push_back(temp_block_point_counts[size_t(block_count*0.1)]);
            quantiles.push_back(temp_block_point_counts[size_t(block_count*0.25)]);
            quantiles.push_back(temp_block_point_counts[size_t(block_count*0.5)]);
            quantiles.push_back(temp_block_point_counts[size_t(block_count*0.75)]);
            quantiles.push_back(temp_block_point_counts[size_t(block_count*0.9)]);

            quantiles.push_back(std::accumulate(temp_block_point_counts.begin(), temp_block_point_counts.end(), size_t{0})/block_count);
            
            return quantiles;
            
        }

        /**
         * Finalize the block store: materialize the disk-backed representations.
         *
         * Called once, as the last step of each index's construction, while the
         * in-memory blocks are still resident. The store is read-only afterwards.
         */
        void FinishedConstruction(){
            mmap_backend_ = std::make_unique<MmapBackend>();
            mmap_backend_->Build(block_list_, block_point_counts_);

            // Off unless ENABLE_PAGED_BACKEND=1, so a default run writes exactly the
            // files it always did.
            if(PagedBackendEnabled()){
                paged_backend_ = std::make_unique<PagedDiskBackend>(PagedGeometryFromEnv());
                paged_backend_->Build(block_list_, block_point_counts_);
            }
        }

        /** The mmap backend, or nullptr before FinishedConstruction() has run. */
        const MmapBackend* MmapBackendPtr() const { return mmap_backend_.get(); }

        /** The paged backend, or nullptr when it was not enabled. */
        PagedDiskBackend* PagedBackendPtr() const { return paged_backend_.get(); }


        // **************************************

    private:
        StorageMode mode_{StorageMode::kInMemory};
        bool blocks_released_{false};
        std::unique_ptr<InMemoryBackend> mem_backend_;
        std::unique_ptr<MmapBackend> mmap_backend_;
        std::unique_ptr<PagedDiskBackend> paged_backend_;
        PointStorageBackend* active_{nullptr};
};


#endif



