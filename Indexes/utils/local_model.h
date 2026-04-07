#ifndef LOCALMODEL_H
#define LOCALMODEL_H


#include <algorithm>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <numeric>
#include <cassert>
#include <cstring>
#include <cstdio>

#include"point.h"
#include"query.h"
#include"sort_tools.h"


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
        void FilterPointsForQuery(Query &query, std::vector<Point>& result_vec){
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


/**
 * Shared block store used by multiple index implementations.
 *
 * It tracks per-block MBR metadata in memory and can optionally materialize a
 * disk-backed, memory-mapped representation for scan experiments.
 */
class BlockStore{
    public:
        std::vector<Block> block_list_;                                 // List of block objects
        std::vector<BoundingRectangle> block_mbrs_;                     // MBRs for each block. Metadata that is usually stored in memory.
        std::vector<size_t> block_point_counts_;                        // Block sizes.
        
        //** Storage structures for a disk backed blockstore. **
        PaddedPoint* flattened_block_list_;                           // when storing within a memory mapped structure use a flattened single list
        std::vector<size_t> block_start_location_;
        std::vector<size_t> block_end_location_;

        std::fstream file_write_obj_;
        std::string blockstore_filename_;   
        size_t file_store_size_{}; 

        bool memory_mapped_data_created{};
        bool use_memory_mapped_data{};
        // ***************************************************

        BlockStore(const std::string &out_filename=""){};

        ~BlockStore() { 
            if(memory_mapped_data_created){
                if (flattened_block_list_ && munmap(flattened_block_list_, file_store_size_))
                    std::cerr << "munmap error " << std::string(strerror(errno));
                std::cout<<"Deleteing blockstorefile "<<blockstore_filename_<<std::endl;
                std::remove(blockstore_filename_.c_str());
            }
        }


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
        /**
         * Scan the supplied blocks and append matching points into `result_vec`.
         *
         * When `use_memory_mapped_data` is enabled this reads from the mmap'd
         * flattened representation; otherwise it scans the in-memory blocks.
         */
        void FilterPointsFromBlocksForQuery(Query &query,std::vector<size_t>& refined_blocks, std::vector<Point>& result_vec){

            if(use_memory_mapped_data){
                for(size_t& block_id: refined_blocks){
                    for(size_t offset=block_start_location_[block_id];offset<block_end_location_[block_id];offset++)
                        if(query.CheckPointWithin(*(flattened_block_list_+offset)))
                            result_vec.emplace_back((flattened_block_list_+offset)->elements_[0],(flattened_block_list_+offset)->elements_[1]);
                }
            }
            else 
                for(size_t& block_id: refined_blocks){
                    if(block_point_counts_[block_id])
                        block_list_[block_id].FilterPointsForQuery(query,result_vec);
                }
        }

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

        /** Return the number of blocks currently stored. */
        size_t NumOfBlocks(){
            return block_list_.size();
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

            quantiles.push_back(temp_block_point_counts[size_t(block_count*0.1)]);
            quantiles.push_back(temp_block_point_counts[size_t(block_count*0.25)]);
            quantiles.push_back(temp_block_point_counts[size_t(block_count*0.5)]);
            quantiles.push_back(temp_block_point_counts[size_t(block_count*0.75)]);
            quantiles.push_back(temp_block_point_counts[size_t(block_count*0.9)]);

            quantiles.push_back(size_t(std::accumulate(temp_block_point_counts.begin(), temp_block_point_counts.end(), 0)/block_count));
            
            return quantiles;
            
        }

        /** Finalize the block store into a memory-mapped, disk-backed layout. */
        void FinishedConstruction(){
            memory_mapped_data_created = true;
            //1. open file write object.
            blockstore_filename_ = "/scratch/project_2005865/sachithp/experiments-md-index/temp_blockstore/"+generate_random_alphanumeric_string(20);
            file_write_obj_ = std::fstream(blockstore_filename_, std::ios::out | std::ios::binary);

            //2. write all blocks to file write object. Also keep track of block_start and block_end locations.

            size_t running_block_end = 0;
            for(size_t block_id=0;block_id<block_list_.size();block_id++){
                block_start_location_.push_back(running_block_end);
                for(auto pt: block_list_[block_id].block_data_){
                    PaddedPoint pd_pt(pt);
                    file_write_obj_.write(reinterpret_cast<const char*>(&pd_pt), sizeof(PaddedPoint));
                }
                running_block_end += block_point_counts_[block_id];
                block_end_location_.push_back(running_block_end);
            }

            //3. close the file pointer and mmap file into array.
            file_write_obj_.close();
            file_store_size_ = std::accumulate(block_point_counts_.begin(), block_point_counts_.end(), 0);

            auto fd = open(blockstore_filename_.c_str(), O_RDONLY);
            flattened_block_list_ = (PaddedPoint *) mmap(nullptr, file_store_size_*sizeof(PaddedPoint), PROT_READ, MAP_SHARED, fd, 0);
        }




        // **************************************

};




// #ifdef DISC_BACKED_BLOCKS
// // ! Can just copy the normal index construction and then make the disk backedness in in the BlockStore::FinishedConstruction function instead of this complex mechanism.
//     /* A class which stores all the blocks.*/
//     class BlockStore{
//         private:
//             Point* flattened_block_list_;                           // when storing within a memory mapped structure use a flattened single list
//             std::vector<size_t> block_point_counts_;                     // in a memory mapped file describes the byte offset for each block
//             std::vector<BoundingRectangle> block_mbrs_;
//             std::vector<size_t> block_start_location_;
//             std::vector<size_t> block_end_location_;

//             std::fstream file_write_obj_;
//             std::string blockstore_filename_;   
//             size_t file_store_size_{};    

//         public:

//             BlockStore(){
//                 filename_ = "blockstore";
//                 file_write_obj_ = std::fstream(filename_, std::ios::out | std::ios::binary);
//             }

//             /*
//             Constructor to create a file and mmap data
//             */
//             BlockStore(const std::string &out_filename):filename_(out_filename){
//                 file_write_obj_ = std::fstream(out_filename, std::ios::out | std::ios::binary);
//             }


//             ~BlockStore() { 
//                 if (flattened_block_list_ && munmap(flattened_block_list_, file_store_size_))
//                     std::cerr << "munmap error " << std::string(strerror(errno));
//             }



//             /* Create and insert a new block from the vector iterators given.
//                 Returns the id of inserted block.
//             */
//             size_t InsertNewBlock(std::vector<Point>::iterator it_data_begin,std::vector<Point>::iterator it_data_end){
                
//                 size_t curr_block_id = block_mbrs_.size(),curr_block_size=0;
//                 // assert(std::distance(it_data_begin,it_data_end)<=BLOCK_SIZE);
//                 block_mbrs_.emplace_back();
//                 for(std::vector<Point>::iterator itr=it_data_begin;itr!=it_data_end;itr++,curr_block_size++)
//                     block_mbrs_[curr_block_id].UpdateBoundingBoxWithPoint(*itr);


//                 block_point_counts_.push_back(curr_block_size);


//                 for (auto it = it_data_begin; it != it_data_end; ++it)
//                     file_write_obj_.write(reinterpret_cast<const char*>(&(*it)), sizeof(Point));
                
//                 return curr_block_id;
//             }

//             /* closes the write pointer creates an mmap of the file to be accesed*/
//             void FinishedConstruction(){
//                 file_write_obj_.close();
//                 file_store_size_ = std::accumulate(block_point_counts_.begin(), block_point_counts_.end(), 0);

//                 auto fd = open(filename_.c_str(), O_RDONLY);
//                 flattened_block_list_ = (Point *) mmap(nullptr, file_store_size_*sizeof(Point), PROT_READ, MAP_SHARED, fd, 0);

//                 size_t cum_block_sizes=0;
//                 for(auto& block_sizes: block_point_counts_){
//                     block_start_location_.push_back(cum_block_sizes);
//                     cum_block_sizes += block_sizes;
//                     block_end_location_.push_back(cum_block_sizes);
//                 }

//             }

//             /*From a list of blocks filter answers for a query.*/
//             void FilterPointsFromBlocksForQuery(Query &query,std::vector<size_t>& refined_blocks, std::vector<Point>& result_vec){
//                 for(size_t& block_id: refined_blocks){
//                     if(query.IsCompletelyCovering(block_mbrs_[block_id]))    // Skipping tests if it can be directlly appended.
//                         result_vec.insert(result_vec.end(),flattened_block_list_+block_start_location_[block_id],flattened_block_list_+block_end_location_[block_id]);
//                     else{
//                         for(size_t offset=block_start_location_[block_id];offset<block_end_location_[block_id];offset++)
//                             if(query.CheckPointWithin(*(flattened_block_list_+offset)))
//                                 result_vec.push_back(*(flattened_block_list_+offset));
//                     }
                        
//                 }
//             }


//             /* Test for block overlap with query*/
//             void RefinedBlocksForQueryFromLocalBlocks(Query &query,BlockStore &block_obj, std::vector<size_t>& local_blocks,std::vector<size_t>& refined_blocks){
//                 for(size_t& block_id: local_blocks)
//                     if(query.IsThereOverlap(block_mbrs_[block_id]))
//                         refined_blocks.push_back(block_id);
//             }
            
//             /*Fetch bounding box*/
//             BoundingRectangle FetchBoundingBoxForBlock(size_t block_id){
//                 return block_mbrs_[block_id];
//             }


//             size_t SumOfPointsInBlocks(std::vector<size_t>& refined_blocks){
//                 size_t sum_points=0;
//                 for(auto &block_id: refined_blocks)
//                     sum_points+=block_point_counts_[block_id];
//                 return sum_points;
//             }

//             size_t NumOfBlocks(){
//                 return block_point_counts_.size();
//             }
            
//     };

    

// #endif




// /* A super class for indexes not splitting down to specific block size. (FLOOD,UniformGridIndex,RSMI)*/
// class LocalModel{
//     public:
//         std:: vector<size_t> local_block_ids_;  // ids of blocks within this local model within the BlockStore
//         size_t block_count_{};
//         size_t point_count_{};
//         BoundingRectangle local_model_mbr_;

        

//         /* Constructor. Splits the already sorted data into blocks.*/
//         LocalModel(std::vector<Point>::iterator data_begin,std::vector<Point>::iterator data_end, BlockStore &block_obj){
//             size_t data_size = std::distance(data_begin,data_end);
//             for (size_t i = 0; i < data_size; i += BLOCK_SIZE,block_count_++) {
//                 size_t begin = i;
//                 size_t end = min(data_size, i + BLOCK_SIZE);
//                 local_block_ids_.push_back(block_obj.InsertNewBlock(data_begin + begin, data_begin + end));
//             }

//             local_model_mbr_.SetToDefault();
//             for(auto id: local_block_ids_)
//                 local_model_mbr_.UpdateBoundingBoxWithBoundingBox(block_obj.FetchBoundingBoxForBlock(id));

//             point_count_+=data_size;
//         }


//         /* Constructor. Splits the already sorted data into blocks.*/
//         LocalModel(std::vector<Point>::iterator data_begin,std::vector<Point>::iterator data_end, BlockStore &block_obj,BoundingRectangle mbr){
//             size_t data_size = std::distance(data_begin,data_end);
//             for (size_t i = 0; i < data_size; i += BLOCK_SIZE,block_count_++) {
//                 size_t begin = i;
//                 size_t end = min(data_size, i + BLOCK_SIZE);
//                 local_block_ids_.push_back(block_obj.InsertNewBlock(data_begin + begin, data_begin + end));

//             }
//             local_model_mbr_=mbr;

//             point_count_+=data_size;

//         }


//         /*Function refines search within LocalModels and returns the exact pages to fetch*/
//         void RefinedBlocksForQuery(Query &query,BlockStore &block_obj, std::vector<size_t>& refined_blocks){
//            block_obj.RefinedBlocksForQueryFromLocalBlocks(query,local_block_ids_,refined_blocks);
//         }
// };


#endif




