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


#define QUERY_NUM_TREES_IN_FOREST 10


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
        for(size_t i =0;i<4;i++)
            result *= (high_hyperpoint_[i]-low_hyperpoint_[i]);
        return result;
    }

    /* Calculates the ration of overlap between two mbrs*/
    double_t RatioOfOverlap(const HyperRectangle& other_hymbr){
        double_t area_of_node = Area();
        double_t area_of_overlap = 1.0;
        for(int i=0;i<4;i++)
            area_of_overlap *= std::min(high_hyperpoint_[i],other_hymbr.high_hyperpoint_[i]) - std::max(low_hyperpoint_[i],other_hymbr.low_hyperpoint_[i]) ;
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
    QueryDensEstTree(std::vector<Query> queries, size_t granularity, BoundingRectangle& global_mbr){ 

        granularity_ = granularity;
        global_mbr_ = global_mbr;

        HyperRectangle hymbr(global_mbr);

        for(int i=0;i<QUERY_NUM_TREES_IN_FOREST;i++)
        {   
            std::cout<<"\t Building the "<<i+1<<"th QueryDensEstTree"<<"\n";
            tree_list_[i] = new QueryDensEstNode();
            BuildTree(tree_list_[i],queries.begin(),queries.end(), hymbr);
            // std::cin.get();
        }
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


        node->split_dim_ = rand()%4;
        node->counts_ = std::distance(queries_begin,queries_end);
        node->hy_mbr_ = hymbr;

        // std::cout<<deb<<" Count:"<<node->counts_<<"  Split Dim:"<<node->split_dim_<<"  "; 
        // hymbr.Print();
        // std::cin.get();

        if(std::distance(queries_begin,queries_end)<=granularity_)
            return;

        node->is_leaf_ = false;

        std::sort(queries_begin,queries_end,BoundingRectSortOrderer(node->split_dim_));  
        // for(auto i=0;i<20;i++) {
        //     cout<<" ### ";
        //     (*(queries_begin+i)).low_.Print();
        //     (*(queries_begin+i)).high_.Print();
        // }

        // node->split_location_ = ( node->hy_mbr_.low_hyperpoint_[node->split_dim_] + node->hy_mbr_.high_hyperpoint_[node->split_dim_] )/2.0;


        if(node->split_dim_<2)
            node->split_location_ = (*(queries_begin+(node->counts_/2))).low_.elements_[node->split_dim_];
        else
            node->split_location_ = (*(queries_begin+(node->counts_/2))).high_.elements_[node->split_dim_-2];

        /* Custom comparator function to find the first position in the vector<Query> with split_dim element greater than split_location */
        auto bs_comp = [node](BoundingRectangle& pt1_iter,double_t split_location) { 
            return  (node->split_dim_<2) ? (pt1_iter.low_.elements_[node->split_dim_]< split_location) : (pt1_iter.high_.elements_[node->split_dim_-2] < split_location); };

        auto split_iter = std::lower_bound (queries_begin, queries_end, node->split_location_,bs_comp);

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


        /**
        * @brief IF were are a leaf node return the value equivalent to ratio of overlap region between 
        *        query region and node region times the number of points present in the node.
        */
        if(node->is_leaf_)
            return  node->counts_* node->hy_mbr_.RatioOfOverlap(query_hymbr);
        

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
