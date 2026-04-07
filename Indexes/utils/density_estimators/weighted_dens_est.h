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


#ifndef WEIGHTED_DENS_EST_TREE_H
#define WEIGHTED_DENS_EST_TREE_H


#include<vector>
#include<algorithm>
#include<functional>
#include<iostream>
#include<cstdlib>
#include"../bounding_rectangle.h"
#include"../point.h"
#include"../query.h"
#include"../sort_tools.h"
#include"../constants.h"
#include"dens_est_node.h"


#define NUM_WEIGHTED_TREES_IN_FOREST 3



/**
 * @brief Root class of density estimation.
 * 
 */
class WeightedDensEstTree{
    public:
    DensEstNode* tree_list_[NUM_WEIGHTED_TREES_IN_FOREST];
    double_t granularity_;
    
    /**
    * @brief Construct a new Dens Est Tree object.
    *   Given the data and the amount of granularity, partition the tree until you reach that granularity.
    */
    WeightedDensEstTree(std::vector<WrappedPoint>& data, std::vector<Query>& queries, double_t granularity, BoundingRectangle mbr){ 

        granularity_ = granularity;
        
        for(int i=0;i<NUM_WEIGHTED_TREES_IN_FOREST;i++)
        {   
            tree_list_[i] = new DensEstNode();
            BuildTree(tree_list_[i],data.begin(),data.end(),mbr);

        }
    }


    /** Recursively release the forest roots. */
    ~WeightedDensEstTree(){ 

        for(int i=0;i<NUM_WEIGHTED_TREES_IN_FOREST;i++)
            delete tree_list_[i];

    }

    





    /** Recursively build one weighted density-estimation tree. */
    /**
    * @brief A helper function to build the tree.
    */
    void BuildTree(DensEstNode* node, std::vector<WrappedPoint>::iterator data_begin, std::vector<WrappedPoint>::iterator data_end, BoundingRectangle mbr){
        
        node->split_dim_ = rand()%Constants::DIM;
        std::sort(data_begin,data_end,SortOrderer(node->split_dim_));

        node->mbr_ = mbr;
        node->counts_ = CalculateWeightedCount(data_begin,data_end);

        /* If the weightage of the points in here is less than granularity*/
        if(node->counts_<granularity_)
            return;
        

        node->is_leaf_ = false;

        
        node->split_location_ = node->counts_/2.0;

        /* Custom comparator function to find the first position in the vector<Point> with split_dim element greater than split_location */
        auto bs_comp = [node](WrappedPoint& pt1_iter,double_t split_location) { return pt1_iter.temp_cum_num_queries_overlapping_< split_location; };
        auto split_iter = std::lower_bound (data_begin, data_end, node->split_location_,bs_comp);

        BoundingRectangle child0_mbr = mbr, child1_mbr = mbr;
        child0_mbr.high_.elements_[node->split_dim_] = node->split_location_;
        child1_mbr.low_.elements_[node->split_dim_] = node->split_location_;

        node->children_[0] = new DensEstNode();
        BuildTree(node->children_[0],data_begin,split_iter,child0_mbr);

        node->children_[1] = new DensEstNode();
        BuildTree(node->children_[1],split_iter,data_end, child1_mbr);


        node->counts_ = node->children_[0]->counts_+node->children_[1]->counts_;
        return;
        
    }


    /** Compute cumulative query-overlap weights for one iterator range. */
    double_t CalculateWeightedCount(std::vector<WrappedPoint>::iterator data_begin, std::vector<WrappedPoint>::iterator data_end){
        double_t result=0;
        for(auto it=data_begin;it!=data_end;it++){
            result+=it->num_queries_overlapping_;
            it->temp_cum_num_queries_overlapping_=result;
        }
        return result;
    }




    /** Estimate the weighted count that falls inside `mbr`. */
    /**
    * @brief The actual exposed function that estimates the counts of data in a given box.
    *        - It estimates the number of points returned by both the trees.
    *        - Averages it to return the values.
    */
    double_t EstimateCount(BoundingRectangle mbr){
        double_t estimate = 0;
        for(int i=0;i<NUM_WEIGHTED_TREES_IN_FOREST;i++){
            estimate += EstimateCountHelper(tree_list_[i],mbr); 
            // std::cout<<"\t\t EstimateCountHelper:: Estimate after "<<i+1<<" trees:"<<(estimate/(i+1))<<"\n";
        }
        return estimate/NUM_WEIGHTED_TREES_IN_FOREST; 
    }


    

    /** Recursive helper for weighted range-count estimation. */
    /**
    * @brief The recursive function that goes through the density estimation tree and calculates
    *        counts within a range query.
    */
    double_t EstimateCountHelper(DensEstNode* node,BoundingRectangle mbr){

        

        
        /* IF the node region is completely inside the query range return the whole count.*/
        if(mbr.IsCompletelyCovering(node->mbr_)){
            // std::cout<<"\t\t EstimateCountHelper:: Leaf with complete coverage:"<<node->counts_<<"\n";
            // node->mbr_.Print();
            // mbr.Print();
            return node->counts_;
        }

        /* Returning if the current tree node has no overlap with query region.*/
        if(!(node->mbr_.IsThereOverlap(mbr)))
            return 0L;


            /* return the value equivalent to ratio of overlap region between 
            *        query region and node region times the number of points present in the node.*/
        if(node->is_leaf_){
            // std::cout<<"\t\t EstimateCountHelper:: Leaf with partial overlap. counts:"<<node->counts_<<" ratio:"<<node->mbr_.RatioOfOverlap(mbr)<<" result:"<<node->counts_* node->mbr_.RatioOfOverlap(mbr)<<"\n";
            // node->mbr_.Print();
            // mbr.Print();
            return  node->counts_* node->mbr_.RatioOfOverlap(mbr);
        }

        /* recurse for lower level nodes if non of the above criteria are met*/
        return EstimateCountHelper(node->children_[0],mbr) + EstimateCountHelper(node->children_[1],mbr);
        
    }


};


#endif
