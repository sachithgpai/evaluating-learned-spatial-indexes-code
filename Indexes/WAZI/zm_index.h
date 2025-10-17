/**
 * @file Ztree.h
 * @author Sachith Pai (sachith.pai@helsinki.fi)
 * @brief 
 * @version 0.1
 * @date 2022-01-02
 * 
 * @copyright Copyright (c) 2022
 * 
 */


#ifndef ZM_INDEX_H
#define ZM_INDEX_H

#include<vector>
#include<cmath>
#include<iostream>
#include<algorithm>
#include<fstream>
#include<string>
#include<map>
#include<list>
#include<tuple>
#include<chrono>
#include"../utils/sort_tools.h"
#include"../utils/query.h"
#include"../utils/pgm/pgm_index.hpp"

#include <cassert>
 
// Use (void) to silence unused warnings.
#define assertm(exp, msg) std::assert(((void)msg, exp))


class ZMIndex{
    public:
        pgm::PGMIndex<uint64_t, 64>* pgm_index_;
        BlockStore block_store_;
        RankSpaceMapper* rank_space_map_;

        std::vector<uint64_t> block_lower_upper_bounds;
        std::vector<size_t> block_ids;

        ZMIndex(std::vector<Point> data){
            rank_space_map_ = new RankSpaceMapper(data);
            std::vector<WrappedPoint> wrapped_data;
            for(auto& pt:data) wrapped_data.emplace_back(pt);
            
            rank_space_map_->Reintegerize(wrapped_data.begin(),wrapped_data.end());

            for(auto& wrap_pt: wrapped_data) wrap_pt.curve_value_ = compute_Z_value(wrap_pt);

            // std::cout<<"ZMIndex finished wrapping points"<<"\n";

            std::sort(wrapped_data.begin(), wrapped_data.end(), WrappedPointCurveValueOrder());


            std::vector<Point> curve_ordered_data;
            for(auto& wrap_pt: wrapped_data) curve_ordered_data.push_back(wrap_pt.ExtractPoint());

            /* Split the sorted data into pages; insert pages into block_store; keep a list of page lower bounds and upper bounds */
            for(int start_it=0,end_it=BLOCK_SIZE;start_it<data.size(); start_it+=BLOCK_SIZE,end_it+=BLOCK_SIZE){
                if(end_it>=data.size())
                    end_it=data.size()-1;

                size_t page_id = block_store_.InsertNewBlock(curve_ordered_data.begin()+start_it,curve_ordered_data.begin()+end_it);
                block_ids.push_back(page_id);
                block_ids.push_back(page_id);

                block_lower_upper_bounds.push_back(wrapped_data[start_it].curve_value_);
                block_lower_upper_bounds.push_back(wrapped_data[end_it-1].curve_value_);
                
            }

            pgm_index_ = new pgm::PGMIndex<uint64_t, 64>(block_lower_upper_bounds);


            block_store_.FinishedConstruction();

        }



        /**
        * @brief A function to perform a range query and return the query results along 
        *        with stats of range query processing. */
        std::vector<Point> RangeQuery(Query& query,WrappedPoint& query_low_rankspace,WrappedPoint& query_high_rankspace){

            std::vector<size_t> projected_blocks;
            Projection(projected_blocks,query_low_rankspace,query_high_rankspace);

            std::vector<size_t> refined_blocks;
            Refinement(refined_blocks,query,projected_blocks);

            std::vector<Point> result_vec;
            Scan(result_vec,query,refined_blocks);

            return result_vec;
        }

        void Projection(std::vector<size_t> &projected_blocks, WrappedPoint& query_low_rankspace,WrappedPoint& query_high_rankspace){

            auto approx_lower_range = pgm_index_->search(query_low_rankspace.curve_value_);
            size_t exact_lower = BinarySearch<uint64_t>(block_lower_upper_bounds, query_low_rankspace.curve_value_, approx_lower_range.lo, approx_lower_range.hi-1);
            if(exact_lower&1) exact_lower--; // exact lower should start on even.



            auto approx_upper_range = pgm_index_->search(query_high_rankspace.curve_value_);
            size_t exact_upper = BinarySearch<uint64_t>(block_lower_upper_bounds, query_high_rankspace.curve_value_, approx_upper_range.lo, approx_upper_range.hi-1);
            if(exact_upper&1 == 0) exact_upper++; // exact upper should end on odd.

            for(size_t it=exact_lower;it<=exact_upper;it+=2)
                projected_blocks.push_back(block_ids[it]);
        }

        void Refinement(std::vector<size_t> &refined_blocks, Query &query, std::vector<size_t> &projected_blocks){
            block_store_.RefinedBlocksForQueryFromLocalBlocks(query,projected_blocks,refined_blocks);
        } 


        void Scan(std::vector<Point> &result_vec, Query &query, std::vector<size_t> &refined_blocks){
            block_store_.FilterPointsFromBlocksForQuery(query,refined_blocks,result_vec);
        }




       
};
#endif