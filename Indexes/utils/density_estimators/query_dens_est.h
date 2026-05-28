/**
 * @file dens_est.h
 * @author Sachith (sachith.pai@helsinki.fi)
 * @brief A density estimation model based on K-D trees.
 * @version 0.1
 * @date 2022-08-17
 * 
 * @copyright Copyright (c) 2022
 * 
 */


#ifndef QUERY_DENS_EST_TREE_H
#define QUERY_DENS_EST_TREE_H


#include<vector>
#include<algorithm>
#include<functional>
#include<iostream>
#include<cstdlib>
#include<string>
#include"../bounding_rectangle.h"
#include"../sort_tools.h"
#include"../constants.h"
#include"../query.h"


#define QUERY_NUM_TREES_IN_FOREST 1


class HyperRectangle{
    public:
    double_t low_hyperpoint_[4];
    double_t high_hyperpoint_[4];

    HyperRectangle(){ 
        std::fill_n(low_hyperpoint_, 4, std::numeric_limits<double_t>::max());
        std::fill_n(high_hyperpoint_, 4, std::numeric_limits<double_t>::min());
    }

    HyperRectangle(const BoundingRectangle& mbr){ 
        low_hyperpoint_[0] = mbr.low_.elements_[0];  
        low_hyperpoint_[2] = mbr.low_.elements_[0];
        low_hyperpoint_[1] = mbr.low_.elements_[1]; 
        low_hyperpoint_[3] = mbr.low_.elements_[1];


        high_hyperpoint_[0] = mbr.high_.elements_[0]; 
        high_hyperpoint_[2] = mbr.high_.elements_[0];
        high_hyperpoint_[1] = mbr.high_.elements_[1]; 
        high_hyperpoint_[3] = mbr.high_.elements_[1];
    }



    HyperRectangle(const HyperRectangle& other_hymbr){ 
        std::copy(other_hymbr.low_hyperpoint_, other_hymbr.low_hyperpoint_+4, low_hyperpoint_);
        std::copy(other_hymbr.high_hyperpoint_, other_hymbr.high_hyperpoint_+4, high_hyperpoint_);
    }
    

    /* Returns true if there is overlap.*/
    bool IsThereOverlap(const HyperRectangle& other_box){
        bool result = true;
        for(size_t i =0;i<4;i++)
            result &= (std::max(low_hyperpoint_[i],other_box.low_hyperpoint_[i])<std::min(high_hyperpoint_[i],other_box.high_hyperpoint_[i]));
        return result;
    }


    /* Returns true if the passed box is completely within.*/
    bool IsCompletelyCovering(const HyperRectangle& other_box){
        bool result = true;
        for(size_t i =0;i<4;i++)
            result &= (low_hyperpoint_[i]<=other_box.low_hyperpoint_[i]) && (high_hyperpoint_[i]>other_box.high_hyperpoint_[i]);
        return result;
    }    

    bool CheckBoundingRectangleWithin(const BoundingRectangle& mbr){

        bool result = true;
        result &= (mbr.low_.elements_[0] >= low_hyperpoint_[0] && mbr.low_.elements_[0]<low_hyperpoint_[1]);
        result &= (mbr.low_.elements_[1] >= high_hyperpoint_[0] && mbr.low_.elements_[1]<high_hyperpoint_[1]);

        result &= (mbr.high_.elements_[0] >= low_hyperpoint_[2] && mbr.high_.elements_[0]<low_hyperpoint_[3]);
        result &= (mbr.high_.elements_[1] >= high_hyperpoint_[2] && mbr.high_.elements_[1]<high_hyperpoint_[3]);

        return result;
    }

    double_t Area(){
        double_t result = 1;
        for(size_t i =0;i<4;i++){
            double_t width = high_hyperpoint_[i]-low_hyperpoint_[i];
            if(width<=0.0)
                return 0.0;
            result *= width;
        }
        return result;
    }

    /* Calculates the ration of overlap between two mbrs*/
    double_t RatioOfOverlap(const HyperRectangle& other_hymbr){
        double_t area_of_node = Area();
        if(area_of_node<=0.0)
            return 0.0;
        double_t area_of_overlap = 1.0;
        for(int i=0;i<4;i++){
            double_t overlap = std::min(high_hyperpoint_[i],other_hymbr.high_hyperpoint_[i]) - std::max(low_hyperpoint_[i],other_hymbr.low_hyperpoint_[i]);
            if(overlap<=0.0)
                return 0.0;
            area_of_overlap *= overlap;
        }
        return area_of_overlap/area_of_node;
    }

    void Print(){
        std::cout<<" (";
        for(int i=0;i<4;i++) cout<<low_hyperpoint_[i]<<", ";
        cout<<") -> (";
        for(int i=0;i<4;i++) cout<<high_hyperpoint_[i]<<", ";
        cout<<")"<<"\n";
    }
};


class BoundingRectSortOrderer {
  size_t i;
public:
  BoundingRectSortOrderer(size_t i) : i{i}{}
    constexpr bool operator()(const BoundingRectangle& a, const BoundingRectangle& b) const  {
    if(i<2)
        return a.low_.elements_[i] < b.low_.elements_[i];
    else    
        return a.high_.elements_[i-2] < b.high_.elements_[i-2];
  }
};



/**
 * @brief Class to hold all the node information.*/
class QueryDensEstNode{
    public:
    HyperRectangle hy_mbr_;  // it is a hyper mbr because we need 8 dim. [0] is for low_, [1] is for high_
    double_t counts_;
    size_t split_dim_;
    double_t split_location_;
    QueryDensEstNode* children_[2];
    std::vector<Query> leaf_queries_;
    bool is_leaf_;

    QueryDensEstNode(){
        counts_ = 0;
        children_[0]=NULL;
        children_[1]=NULL;
        is_leaf_ = true;
    }
    
};



/**
 * @brief Root class of density estimation.
 * 
 */
class QueryDensEstTree{
    public:
    QueryDensEstNode* tree_list_[QUERY_NUM_TREES_IN_FOREST];
    size_t granularity_;
    BoundingRectangle global_mbr_;
    
    /* Given the query and the amount of granularity, partition the tree until you reach that granularity. */
    QueryDensEstTree(std::vector<Query> queries, size_t granularity, const BoundingRectangle& global_mbr){

        granularity_ = granularity;
        global_mbr_ = global_mbr;

        HyperRectangle hymbr(global_mbr);

        for(int i=0;i<QUERY_NUM_TREES_IN_FOREST;i++)
            tree_list_[i] = NULL;

        for(int i=0;i<QUERY_NUM_TREES_IN_FOREST;i++)
        {   
            tree_list_[i] = new QueryDensEstNode();
            BuildTree(tree_list_[i],queries.begin(),queries.end(), hymbr);
        }
    }

    QueryDensEstTree(const QueryDensEstTree&) = delete;
    QueryDensEstTree& operator=(const QueryDensEstTree&) = delete;

    ~QueryDensEstTree(){
        for(int i=0;i<QUERY_NUM_TREES_IN_FOREST;i++)
            DeleteTree(tree_list_[i]);
    }

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


    /**
    * @brief A helper function to build the tree.
    * 
    * @param node 
    * @param data 
    * @param granularity 
    * @param order_x `true` = split along x
    */
    void BuildTree(QueryDensEstNode* node, std::vector<Query>::iterator queries_begin, std::vector<Query>::iterator queries_end, HyperRectangle& hymbr){


        const auto query_count = std::distance(queries_begin,queries_end);
        node->counts_ = query_count;
        node->hy_mbr_ = hymbr;

        // std::cout<<deb<<" Count:"<<node->counts_<<"  Split Dim:"<<node->split_dim_<<"  "; 
        // hymbr.Print();
        // std::cin.get();

        if(query_count<=static_cast<decltype(query_count)>(granularity_) || query_count<2){
            StoreLeafQueries(node, queries_begin, queries_end);
            return;
        }

        auto split_iter = queries_begin;
        bool found_split = false;
        size_t first_split_dim = 0;
        for(size_t split_dim_offset=0;split_dim_offset<4;split_dim_offset++){
            node->split_dim_ = (first_split_dim+split_dim_offset)%4;
            std::sort(queries_begin,queries_end,BoundingRectSortOrderer(node->split_dim_));

            auto endpoint_value = [node](const Query& query) {
                return (node->split_dim_<2) ?
                    query.low_.elements_[node->split_dim_] :
                    query.high_.elements_[node->split_dim_-2];
            };

            auto mid_iter = queries_begin + (query_count/2);
            double_t lower_mid_value = endpoint_value(*(mid_iter-1));
            double_t upper_mid_value = endpoint_value(*mid_iter);
            if(upper_mid_value<=lower_mid_value)
                continue;

            node->split_location_ = (lower_mid_value+upper_mid_value)/2.0;

            /* Custom comparator function to find the first position in the vector<Query> with split_dim element greater than split_location */
            auto bs_comp = [node](BoundingRectangle& pt1_iter,double_t split_location) {
                return  (node->split_dim_<2) ? (pt1_iter.low_.elements_[node->split_dim_]< split_location) : (pt1_iter.high_.elements_[node->split_dim_-2] < split_location); };

            split_iter = std::lower_bound (queries_begin, queries_end, node->split_location_,bs_comp);
            if(split_iter==queries_begin || split_iter==queries_end)
                continue;

            found_split = true;
            break;
        }

        if(!found_split){
            StoreLeafQueries(node, queries_begin, queries_end);
            return;
        }

        node->is_leaf_ = false;

        HyperRectangle child0_hymbr(node->hy_mbr_);
        HyperRectangle child1_hymbr(node->hy_mbr_);

        if(node->split_dim_<2){ // split dim 0 or 1
            child0_hymbr.high_hyperpoint_[node->split_dim_] = node->split_location_;

            child1_hymbr.low_hyperpoint_[node->split_dim_] = node->split_location_;
            child1_hymbr.low_hyperpoint_[node->split_dim_+2] = std::max(child1_hymbr.low_hyperpoint_[node->split_dim_+2],node->split_location_);

        }
        else{
            child0_hymbr.high_hyperpoint_[node->split_dim_] = node->split_location_;
            child0_hymbr.high_hyperpoint_[node->split_dim_-2] =  std::min(child0_hymbr.high_hyperpoint_[node->split_dim_-2],node->split_location_);

            child1_hymbr.low_hyperpoint_[node->split_dim_] = node->split_location_;
            // child1_hymbr.low[node->split_dim_-2] = std::min(child1_hymbr.low[node->split_dim_-2],node->split_location_);
        }

        node->children_[0] = new QueryDensEstNode();
        BuildTree(node->children_[0],queries_begin,split_iter,child0_hymbr);

        node->children_[1] = new QueryDensEstNode();
        BuildTree(node->children_[1],split_iter,queries_end, child1_hymbr);

        return;
        
    }


    /**
    * @brief The actual exposed function that estimates the counts of data in a given box.
    *        - It estimates the number of points returned by both the trees.
    *        - Averages it to return the values.
    */
    std::vector<double_t> EstimateAllQueryCounts(BoundingRectangle& local_mbr,Point& split){
        
        std::vector<double_t> result(9,0.0);
        for(int i=0;i<QUERY_NUM_TREES_IN_FOREST;i++){
            result[0] += EstimateCountQueryAA(tree_list_[i],local_mbr,split);
            result[1] += EstimateCountQueryBB(tree_list_[i],local_mbr,split);
            result[2] += EstimateCountQueryCC(tree_list_[i],local_mbr,split);
            result[3] += EstimateCountQueryDD(tree_list_[i],local_mbr,split);
            result[4] += EstimateCountQueryAB(tree_list_[i],local_mbr,split);
            result[5] += EstimateCountQueryAC(tree_list_[i],local_mbr,split);
            result[6] += EstimateCountQueryBD(tree_list_[i],local_mbr,split);
            result[7] += EstimateCountQueryCD(tree_list_[i],local_mbr,split);
            result[8] += EstimateCountQueryAD(tree_list_[i],local_mbr,split);
            
        }

        for(int i=0;i<9;i++) result[i] /= QUERY_NUM_TREES_IN_FOREST;
        return result;
    }

    double_t EstimateCountHelper(HyperRectangle& query_hymbr){
        double_t result=0.0;
        for(int i=0;i<QUERY_NUM_TREES_IN_FOREST;i++)
            result+= EstimateCountHelper(tree_list_[i],query_hymbr);
        
        return result/QUERY_NUM_TREES_IN_FOREST;
    }

    bool CheckQueryEndpointWithinHyperRectangle(const Query& query, const HyperRectangle& query_hymbr){
        bool result = true;
        result &= (query.low_.elements_[0] >= query_hymbr.low_hyperpoint_[0] && query.low_.elements_[0] < query_hymbr.high_hyperpoint_[0]);
        result &= (query.low_.elements_[1] >= query_hymbr.low_hyperpoint_[1] && query.low_.elements_[1] < query_hymbr.high_hyperpoint_[1]);
        result &= (query.high_.elements_[0] >= query_hymbr.low_hyperpoint_[2] && query.high_.elements_[0] < query_hymbr.high_hyperpoint_[2]);
        result &= (query.high_.elements_[1] >= query_hymbr.low_hyperpoint_[3] && query.high_.elements_[1] < query_hymbr.high_hyperpoint_[3]);
        return result;
    }

    double_t ExactLeafCount(QueryDensEstNode* node, HyperRectangle& query_hymbr){
        double_t result = 0.0;
        for(auto& query: node->leaf_queries_)
            if(CheckQueryEndpointWithinHyperRectangle(query, query_hymbr))
                result += 1.0;
        return result;
    }

    double_t ExactLeafOverlapCount(QueryDensEstNode* node, const BoundingRectangle& local_mbr){
        double_t result = 0.0;
        for(auto& query: node->leaf_queries_)
            if(query.IsThereOverlap(local_mbr))
                result += 1.0;
        return result;
    }

    bool IsCompletelyCoveredByOverlapRange(const HyperRectangle& node_hymbr, const HyperRectangle& query_hymbr){
        bool result = true;
        for(size_t i=0;i<4;i++)
            result &= (query_hymbr.low_hyperpoint_[i]<=node_hymbr.low_hyperpoint_[i]) &&
                      (query_hymbr.high_hyperpoint_[i]>node_hymbr.high_hyperpoint_[i]);

        result &= query_hymbr.low_hyperpoint_[2] < node_hymbr.low_hyperpoint_[2];
        result &= query_hymbr.low_hyperpoint_[3] < node_hymbr.low_hyperpoint_[3];
        return result;
    }




    /**
    * @brief The recursive function that goes through the density estimation tree and calculates
    *        counts within a range query.
    */
    double_t EstimateCountHelper(QueryDensEstNode* node, HyperRectangle& query_hymbr){

        /* IF the node region is completely inside the query range return the whole count.*/
        // node->hy_mbr_.Print();
        // query_hymbr.Print();
        // std::cin.get();
        if(query_hymbr.IsCompletelyCovering(node->hy_mbr_))
            return node->counts_;

        /* Returning if the current tree node has no overlap with query region.*/
        if(!(node->hy_mbr_.IsThereOverlap(query_hymbr)))
            return 0L;


        if(node->is_leaf_)
            return  ExactLeafCount(node, query_hymbr);
        

        /* recurse for lower level nodes if non of the above criteria are met*/
        return EstimateCountHelper(node->children_[0],query_hymbr) + EstimateCountHelper(node->children_[1],query_hymbr);
        
    }




    double_t EstimateCountQuery(BoundingRectangle& local_mbr){
        HyperRectangle query_hymbr;
        query_hymbr.low_hyperpoint_[0]=local_mbr.low_.elements_[0];
        query_hymbr.low_hyperpoint_[1]=local_mbr.low_.elements_[1];
        query_hymbr.low_hyperpoint_[2]=local_mbr.low_.elements_[0];
        query_hymbr.low_hyperpoint_[3]=local_mbr.low_.elements_[1];
        

        query_hymbr.high_hyperpoint_[0]=local_mbr.high_.elements_[0];
        query_hymbr.high_hyperpoint_[1]=local_mbr.high_.elements_[1];
        query_hymbr.high_hyperpoint_[2]=local_mbr.high_.elements_[0];
        query_hymbr.high_hyperpoint_[3]=local_mbr.high_.elements_[1];

        double_t result =0;
        for(int i=0;i<QUERY_NUM_TREES_IN_FOREST;i++)
            result+=EstimateCountHelper(tree_list_[i],query_hymbr);
        
        return result/QUERY_NUM_TREES_IN_FOREST;
    }

    /**
    * @brief Estimate how many training query rectangles overlap `local_mbr`.
    *
    * Each query rectangle is represented as a 4D point:
    * (low_x, low_y, high_x, high_y). A query overlaps the local MBR iff
    * low_x < mbr.high_x, low_y < mbr.high_y,
    * high_x > mbr.low_x, and high_y > mbr.low_y.
    */
    double_t EstimateOverlapCount(const BoundingRectangle& local_mbr){
        HyperRectangle query_hymbr;
        query_hymbr.low_hyperpoint_[0]=std::numeric_limits<double_t>::lowest();
        query_hymbr.low_hyperpoint_[1]=std::numeric_limits<double_t>::lowest();
        query_hymbr.low_hyperpoint_[2]=local_mbr.low_.elements_[0];
        query_hymbr.low_hyperpoint_[3]=local_mbr.low_.elements_[1];

        query_hymbr.high_hyperpoint_[0]=local_mbr.high_.elements_[0];
        query_hymbr.high_hyperpoint_[1]=local_mbr.high_.elements_[1];
        query_hymbr.high_hyperpoint_[2]=std::numeric_limits<double_t>::max();
        query_hymbr.high_hyperpoint_[3]=std::numeric_limits<double_t>::max();

        double_t result = 0.0;
        for(int i=0;i<QUERY_NUM_TREES_IN_FOREST;i++)
            result+=EstimateOverlapCount(tree_list_[i], local_mbr, query_hymbr);

        return result/QUERY_NUM_TREES_IN_FOREST;
    }

    double_t EstimateOverlapCount(QueryDensEstNode* node, const BoundingRectangle& local_mbr, HyperRectangle& query_hymbr){
        if(IsCompletelyCoveredByOverlapRange(node->hy_mbr_, query_hymbr))
            return node->counts_;

        if(!(node->hy_mbr_.IsThereOverlap(query_hymbr)))
            return 0L;

        if(node->is_leaf_)
            return ExactLeafOverlapCount(node, local_mbr);

        return EstimateOverlapCount(node->children_[0],local_mbr,query_hymbr) + EstimateOverlapCount(node->children_[1],local_mbr,query_hymbr);
    }


    double_t EstimateCountQueryAA(QueryDensEstNode* node, BoundingRectangle& local_mbr, Point& split){
        HyperRectangle query_hymbr;
        query_hymbr.low_hyperpoint_[0]=global_mbr_.low_.elements_[0];
        query_hymbr.low_hyperpoint_[1]=global_mbr_.low_.elements_[1];
        query_hymbr.low_hyperpoint_[2]=local_mbr.low_.elements_[0];
        query_hymbr.low_hyperpoint_[3]=local_mbr.low_.elements_[1];
        

        query_hymbr.high_hyperpoint_[0]=split.elements_[0];
        query_hymbr.high_hyperpoint_[1]=split.elements_[1];
        query_hymbr.high_hyperpoint_[2]=split.elements_[0];
        query_hymbr.high_hyperpoint_[3]=split.elements_[1];

        return EstimateCountHelper(node,query_hymbr);
    }
    

    double_t EstimateCountQueryBB(QueryDensEstNode* node, BoundingRectangle& local_mbr, Point& split){
        HyperRectangle query_hymbr;
        query_hymbr.low_hyperpoint_[0]=split.elements_[0];
        query_hymbr.low_hyperpoint_[1]=global_mbr_.low_.elements_[1];
        query_hymbr.low_hyperpoint_[2]=split.elements_[0];
        query_hymbr.low_hyperpoint_[3]=local_mbr.low_.elements_[1];
    
        query_hymbr.high_hyperpoint_[0]=local_mbr.high_.elements_[0];
        query_hymbr.high_hyperpoint_[1]=split.elements_[1];
        query_hymbr.high_hyperpoint_[2]=global_mbr_.high_.elements_[0];
        query_hymbr.high_hyperpoint_[3]=split.elements_[1];
        return EstimateCountHelper(node,query_hymbr);
    }
    
    double_t EstimateCountQueryCC(QueryDensEstNode* node, BoundingRectangle& local_mbr, Point& split){
        HyperRectangle query_hymbr;
        query_hymbr.low_hyperpoint_[0]=global_mbr_.low_.elements_[0];
        query_hymbr.low_hyperpoint_[1]=split.elements_[1];
        query_hymbr.low_hyperpoint_[2]=local_mbr.low_.elements_[0];
        query_hymbr.low_hyperpoint_[3]=split.elements_[1];
        

        query_hymbr.high_hyperpoint_[0]=split.elements_[0];
        query_hymbr.high_hyperpoint_[1]=local_mbr.high_.elements_[1];
        query_hymbr.high_hyperpoint_[2]=split.elements_[0];
        query_hymbr.high_hyperpoint_[3]=global_mbr_.high_.elements_[1];

        return EstimateCountHelper(node,query_hymbr);
    }


    double_t EstimateCountQueryDD(QueryDensEstNode* node, BoundingRectangle& local_mbr, Point& split){
        HyperRectangle query_hymbr;
        query_hymbr.low_hyperpoint_[0]=split.elements_[0];
        query_hymbr.low_hyperpoint_[1]=split.elements_[1];
        query_hymbr.low_hyperpoint_[2]=split.elements_[0];
        query_hymbr.low_hyperpoint_[3]=split.elements_[1];
        

        query_hymbr.high_hyperpoint_[0]=local_mbr.high_.elements_[0];
        query_hymbr.high_hyperpoint_[1]=local_mbr.high_.elements_[1];
        query_hymbr.high_hyperpoint_[2]=global_mbr_.high_.elements_[0];
        query_hymbr.high_hyperpoint_[3]=global_mbr_.high_.elements_[1];

        return EstimateCountHelper(node,query_hymbr);
    }


    double_t EstimateCountQueryAB(QueryDensEstNode* node, BoundingRectangle& local_mbr, Point& split){
        HyperRectangle query_hymbr;
        query_hymbr.low_hyperpoint_[0]=global_mbr_.low_.elements_[0];
        query_hymbr.low_hyperpoint_[1]=global_mbr_.low_.elements_[1];
        query_hymbr.low_hyperpoint_[2]=split.elements_[0];
        query_hymbr.low_hyperpoint_[3]=local_mbr.low_.elements_[1];
        

        query_hymbr.high_hyperpoint_[0]=split.elements_[0];
        query_hymbr.high_hyperpoint_[1]=split.elements_[1];
        query_hymbr.high_hyperpoint_[2]=global_mbr_.high_.elements_[0];
        query_hymbr.high_hyperpoint_[3]=split.elements_[1];

        return EstimateCountHelper(node,query_hymbr);
    }


    double_t EstimateCountQueryAC(QueryDensEstNode* node, BoundingRectangle& local_mbr, Point& split){
        HyperRectangle query_hymbr;
        query_hymbr.low_hyperpoint_[0]=global_mbr_.low_.elements_[0];
        query_hymbr.low_hyperpoint_[1]=global_mbr_.low_.elements_[1];
        query_hymbr.low_hyperpoint_[2]=local_mbr.low_.elements_[0];
        query_hymbr.low_hyperpoint_[3]=split.elements_[1];
        

        query_hymbr.high_hyperpoint_[0]=split.elements_[0];
        query_hymbr.high_hyperpoint_[1]=split.elements_[1];
        query_hymbr.high_hyperpoint_[2]=split.elements_[0];
        query_hymbr.high_hyperpoint_[3]=global_mbr_.high_.elements_[1];

        return EstimateCountHelper(node,query_hymbr);
    }


    double_t EstimateCountQueryBD(QueryDensEstNode* node, BoundingRectangle& local_mbr, Point& split){
        HyperRectangle query_hymbr;
        query_hymbr.low_hyperpoint_[0]=split.elements_[0];
        query_hymbr.low_hyperpoint_[1]=global_mbr_.low_.elements_[1];
        query_hymbr.low_hyperpoint_[2]=split.elements_[0];
        query_hymbr.low_hyperpoint_[3]=split.elements_[1];
        

        query_hymbr.high_hyperpoint_[0]=local_mbr.high_.elements_[0];
        query_hymbr.high_hyperpoint_[1]=split.elements_[1];
        query_hymbr.high_hyperpoint_[2]=global_mbr_.high_.elements_[0];
        query_hymbr.high_hyperpoint_[3]=global_mbr_.high_.elements_[1];

        return EstimateCountHelper(node,query_hymbr);
    }


    double_t EstimateCountQueryCD(QueryDensEstNode* node, BoundingRectangle& local_mbr, Point& split){
        HyperRectangle query_hymbr;
        query_hymbr.low_hyperpoint_[0]=global_mbr_.low_.elements_[0];
        query_hymbr.low_hyperpoint_[1]=split.elements_[1];
        query_hymbr.low_hyperpoint_[2]=split.elements_[0];
        query_hymbr.low_hyperpoint_[3]=split.elements_[1];
        

        query_hymbr.high_hyperpoint_[0]=split.elements_[0];
        query_hymbr.high_hyperpoint_[1]=local_mbr.high_.elements_[1];
        query_hymbr.high_hyperpoint_[2]=global_mbr_.high_.elements_[0];
        query_hymbr.high_hyperpoint_[3]=global_mbr_.high_.elements_[1];

        return EstimateCountHelper(node,query_hymbr);
    }


    double_t EstimateCountQueryAD(QueryDensEstNode* node, BoundingRectangle& local_mbr, Point& split){
        HyperRectangle query_hymbr;
        query_hymbr.low_hyperpoint_[0]=global_mbr_.low_.elements_[0];
        query_hymbr.low_hyperpoint_[1]=global_mbr_.low_.elements_[1];
        query_hymbr.low_hyperpoint_[2]=split.elements_[0];
        query_hymbr.low_hyperpoint_[3]=split.elements_[1];
        

        query_hymbr.high_hyperpoint_[0]=split.elements_[0];
        query_hymbr.high_hyperpoint_[1]=split.elements_[1];
        query_hymbr.high_hyperpoint_[2]=global_mbr_.high_.elements_[0];
        query_hymbr.high_hyperpoint_[3]=global_mbr_.high_.elements_[1];

        return EstimateCountHelper(node,query_hymbr);
    }
};


#endif
