/**
 * @file query_dens_est.h
 * Exact 4D range-count structure for training query rectangles.
 */

#ifndef QUERY_DENS_EST_TREE_H
#define QUERY_DENS_EST_TREE_H

#include<algorithm>
#include<cstddef>
#include<iterator>
#include<limits>
#include<vector>

#include"../query.h"


/**
 * A 4D box over query endpoints:
 * (query.low.x, query.low.y, query.high.x, query.high.y).
 */
class QueryEndpointBox{
    public:
        double_t low_[4];
        double_t high_[4];

        QueryEndpointBox(){
            std::fill_n(low_, 4, std::numeric_limits<double_t>::max());
            std::fill_n(high_, 4, std::numeric_limits<double_t>::lowest());
        }

        QueryEndpointBox(const BoundingRectangle& endpoint_mbr){
            low_[0] = endpoint_mbr.low_.elements_[0];
            low_[1] = endpoint_mbr.low_.elements_[1];
            low_[2] = endpoint_mbr.low_.elements_[0];
            low_[3] = endpoint_mbr.low_.elements_[1];

            high_[0] = endpoint_mbr.high_.elements_[0];
            high_[1] = endpoint_mbr.high_.elements_[1];
            high_[2] = endpoint_mbr.high_.elements_[0];
            high_[3] = endpoint_mbr.high_.elements_[1];
        }

        bool IsThereOverlap(const QueryEndpointBox& other) const{
            bool result = true;
            for(size_t i=0;i<4;i++)
                result &= (std::max(low_[i], other.low_[i]) < std::min(high_[i], other.high_[i]));
            return result;
        }
};


class QueryEndpointSortOrderer{
    size_t dim_;

    public:
        QueryEndpointSortOrderer(size_t dim):dim_(dim){}

        bool operator()(const Query& lhs, const Query& rhs) const{
            if(dim_<2)
                return lhs.low_.elements_[dim_] < rhs.low_.elements_[dim_];
            return lhs.high_.elements_[dim_-2] < rhs.high_.elements_[dim_-2];
        }
};


/**
 * Node in the exact query endpoint range-count tree.
 */
class QueryDensEstNode{
    public:
        QueryEndpointBox endpoint_box_;
        std::vector<Query> leaf_queries_;
        QueryDensEstNode* children_[2];
        size_t count_;
        bool is_leaf_;

        QueryDensEstNode(){
            children_[0] = NULL;
            children_[1] = NULL;
            count_ = 0;
            is_leaf_ = true;
        }
};


/**
 * Exact count index for training query rectangles.
 *
 * Each query is indexed as the 4D endpoint point
 * (low_x, low_y, high_x, high_y). A rectangle query overlaps an R-tree MBR iff:
 * low_x < mbr.high_x, low_y < mbr.high_y,
 * high_x > mbr.low_x, and high_y > mbr.low_y.
 */
class QueryDensEstTree{
        QueryDensEstNode* root_;
        size_t granularity_;

    public:

        QueryDensEstTree(std::vector<Query> queries, size_t granularity){
            granularity_ = std::max<size_t>(granularity, 1);
            root_ = new QueryDensEstNode();
            QueryEndpointBox endpoint_box;
            if(!queries.empty())
                endpoint_box = QueryEndpointBox(EndpointBounds(queries));
            BuildTree(root_, queries.begin(), queries.end(), endpoint_box);
        }

        QueryDensEstTree(const QueryDensEstTree&) = delete;
        QueryDensEstTree& operator=(const QueryDensEstTree&) = delete;

        ~QueryDensEstTree(){
            DeleteTree(root_);
        }

        double_t EstimateOverlapCount(const BoundingRectangle& mbr) const{
            if(root_->count_==0)
                return 0.0;

            QueryEndpointBox overlap_range;
            overlap_range.low_[0] = std::numeric_limits<double_t>::lowest();
            overlap_range.low_[1] = std::numeric_limits<double_t>::lowest();
            overlap_range.low_[2] = mbr.low_.elements_[0];
            overlap_range.low_[3] = mbr.low_.elements_[1];

            overlap_range.high_[0] = mbr.high_.elements_[0];
            overlap_range.high_[1] = mbr.high_.elements_[1];
            overlap_range.high_[2] = std::numeric_limits<double_t>::max();
            overlap_range.high_[3] = std::numeric_limits<double_t>::max();

            return EstimateOverlapCount(root_, mbr, overlap_range);
        }

    private:
        void DeleteTree(QueryDensEstNode* node){
            if(node==NULL)
                return;
            DeleteTree(node->children_[0]);
            DeleteTree(node->children_[1]);
            delete node;
        }

        void StoreLeafQueries(QueryDensEstNode* node, std::vector<Query>::iterator queries_begin, std::vector<Query>::iterator queries_end){
            node->is_leaf_ = true;
            node->leaf_queries_.assign(queries_begin, queries_end);
        }

        BoundingRectangle EndpointBounds(const std::vector<Query>& queries) const{
            BoundingRectangle endpoint_mbr(queries[0].low_, queries[0].low_);
            for(auto& query: queries){
                endpoint_mbr.UpdateBoundingBoxWithPoint(query.low_);
                endpoint_mbr.UpdateBoundingBoxWithPoint(query.high_);
            }

            for(size_t dim=0;dim<Constants::DIM;dim++){
                double_t span = endpoint_mbr.high_.elements_[dim] - endpoint_mbr.low_.elements_[dim];
                double_t padding = std::max(Constants::EPSILON_ERR, span*Constants::EPSILON_ERR);
                endpoint_mbr.low_.elements_[dim] -= padding;
                endpoint_mbr.high_.elements_[dim] += padding;
            }

            return endpoint_mbr;
        }

        double_t EndpointValue(const Query& query, size_t dim) const{
            if(dim<2)
                return query.low_.elements_[dim];
            return query.high_.elements_[dim-2];
        }

        void BuildTree(QueryDensEstNode* node, std::vector<Query>::iterator queries_begin, std::vector<Query>::iterator queries_end, const QueryEndpointBox& endpoint_box){
            const auto query_count = std::distance(queries_begin, queries_end);
            node->count_ = query_count;
            node->endpoint_box_ = endpoint_box;

            if(query_count<=static_cast<decltype(query_count)>(granularity_) || query_count<2){
                StoreLeafQueries(node, queries_begin, queries_end);
                return;
            }

            auto split_iter = queries_begin;
            bool found_split = false;
            size_t split_dim_for_children = 0;
            double_t split_location = 0.0;
            for(size_t split_dim=0;split_dim<4;split_dim++){
                std::sort(queries_begin, queries_end, QueryEndpointSortOrderer(split_dim));

                auto mid_iter = queries_begin + (query_count/2);
                double_t lower_mid_value = EndpointValue(*(mid_iter-1), split_dim);
                double_t upper_mid_value = EndpointValue(*mid_iter, split_dim);
                if(upper_mid_value<=lower_mid_value)
                    continue;

                split_location = (lower_mid_value+upper_mid_value)/2.0;
                auto bs_comp = [this, split_dim](const Query& query, double_t split_location) {
                    return EndpointValue(query, split_dim) < split_location;
                };

                split_iter = std::lower_bound(queries_begin, queries_end, split_location, bs_comp);
                if(split_iter==queries_begin || split_iter==queries_end)
                    continue;

                split_dim_for_children = split_dim;
                found_split = true;
                break;
            }

            if(!found_split){
                StoreLeafQueries(node, queries_begin, queries_end);
                return;
            }

            node->is_leaf_ = false;
            QueryEndpointBox child0_box(node->endpoint_box_);
            QueryEndpointBox child1_box(node->endpoint_box_);

            if(split_dim_for_children<2){
                child0_box.high_[split_dim_for_children] = split_location;

                child1_box.low_[split_dim_for_children] = split_location;
                child1_box.low_[split_dim_for_children+2] = std::max(child1_box.low_[split_dim_for_children+2], split_location);
            }
            else{
                child0_box.high_[split_dim_for_children] = split_location;
                child0_box.high_[split_dim_for_children-2] = std::min(child0_box.high_[split_dim_for_children-2], split_location);

                child1_box.low_[split_dim_for_children] = split_location;
            }

            node->children_[0] = new QueryDensEstNode();
            BuildTree(node->children_[0], queries_begin, split_iter, child0_box);

            node->children_[1] = new QueryDensEstNode();
            BuildTree(node->children_[1], split_iter, queries_end, child1_box);
        }

        bool IsCompletelyCoveredByOverlapRange(const QueryEndpointBox& node_box, const QueryEndpointBox& overlap_range) const{
            bool result = true;

            result &= overlap_range.low_[0] <= node_box.low_[0];
            result &= overlap_range.low_[1] <= node_box.low_[1];
            result &= overlap_range.low_[2] < node_box.low_[2];
            result &= overlap_range.low_[3] < node_box.low_[3];

            for(size_t i=0;i<4;i++)
                result &= overlap_range.high_[i] >= node_box.high_[i];

            return result;
        }

        double_t ExactLeafOverlapCount(QueryDensEstNode* node, const BoundingRectangle& mbr) const{
            double_t result = 0.0;
            for(auto& query: node->leaf_queries_)
                if(query.IsThereOverlap(mbr))
                    result += 1.0;
            return result;
        }

        double_t EstimateOverlapCount(QueryDensEstNode* node, const BoundingRectangle& mbr, const QueryEndpointBox& overlap_range) const{
            if(IsCompletelyCoveredByOverlapRange(node->endpoint_box_, overlap_range))
                return node->count_;

            if(!(node->endpoint_box_.IsThereOverlap(overlap_range)))
                return 0.0;

            if(node->is_leaf_)
                return ExactLeafOverlapCount(node, mbr);

            return EstimateOverlapCount(node->children_[0], mbr, overlap_range) +
                   EstimateOverlapCount(node->children_[1], mbr, overlap_range);
        }
};

#endif
