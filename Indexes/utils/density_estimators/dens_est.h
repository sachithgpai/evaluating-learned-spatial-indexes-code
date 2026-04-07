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


#ifndef DENS_EST_TREE_H
#define DENS_EST_TREE_H


#include<vector>
#include<algorithm>
#include<functional>
#include<iostream>
#include<cstdlib>
#include<string>
#include"../bounding_rectangle.h"
#include"../point.h"
#include"../sort_tools.h"
#include"../constants.h"
#include"../query.h"
#include"dens_est_node.h"


#define NUM_TREES_IN_FOREST 3






/**
 * @brief Root class of density estimation.
 * 
 */
class DensEstTree{
    public:
    DensEstNode* tree_list_[NUM_TREES_IN_FOREST];
    size_t granularity_;
    
    /**
    * @brief Construct a new Dens Est Tree object.
    *   Given the data and the amount of granularity, partition the tree until you reach that granularity.
    */
    DensEstTree(std::vector<Point>& data, size_t granularity, BoundingRectangle mbr, bool debug=false){ 

        granularity_ = granularity;

        for(int i=0;i<NUM_TREES_IN_FOREST;i++)
        {   
            tree_list_[i] = new DensEstNode();


            tree_list_[i]->split_dim_ = rand()%Constants::DIM;
            tree_list_[i]->counts_ = data.size();
            tree_list_[i]->mbr_ = mbr;
            BuildTree(tree_list_[i],data.begin(),data.end());

        }
    }

    

    /** Recursively build one tree in the estimation forest. */
    /**
    * @brief A helper function to build the tree.
    * 
    * @param node 
    * @param data 
    * @param granularity 
    * @param order_x `true` = split along x
    */
    void BuildTree(DensEstNode* node, std::vector<Point>::iterator data_begin, std::vector<Point>::iterator data_end){




        if(node->counts_<=granularity_){
            return;
        }

        node->is_leaf_ = false;

        std::sort(data_begin,data_end,SortOrderer(node->split_dim_));
        
        node->split_location_ = (node->mbr_.high_.elements_[node->split_dim_]+node->mbr_.low_.elements_[node->split_dim_])/2.0;

        /* Custom comparator function to find the first position in the vector<Point> with split_dim element greater than split_location */
        auto bs_comp = [node](Point& pt1_iter,double_t split_location) { return pt1_iter.elements_[node->split_dim_]< split_location; };
        auto split_iter = std::lower_bound (data_begin, data_end, node->split_location_,bs_comp);




        
        node->children_[0] = new DensEstNode();
        node->children_[0]->split_dim_ = rand()%Constants::DIM;
        node->children_[0]->counts_ = std::distance(data_begin,split_iter);
        node->children_[0]->mbr_ = node->mbr_;


        node->children_[1] = new DensEstNode();
        node->children_[1]->split_dim_ = rand()%Constants::DIM;
        node->children_[1]->counts_ = std::distance(split_iter,data_end);
        node->children_[1]->mbr_ = node->mbr_;

        if(std::distance(data_begin,split_iter)==0 || std::distance(split_iter,data_end)==0) // No splitting happened.
            return;


        // BoundingRectangle child0_mbr = mbr, child1_mbr = mbr;
        node->children_[0]->mbr_.high_.elements_[node->split_dim_] = node->split_location_;
        node->children_[1]->mbr_.low_.elements_[node->split_dim_] = node->split_location_;

        BuildTree(node->children_[0],data_begin,split_iter);

        BuildTree(node->children_[1],split_iter,data_end);

        return;
        
    }


    /** Estimate the point count that falls inside `mbr`. */
    /**
    * @brief The actual exposed function that estimates the counts of data in a given box.
    *        - It estimates the number of points returned by both the trees.
    *        - Averages it to return the values.
    */
    double_t EstimateCount(BoundingRectangle mbr){
        double_t estimate = 0;
        for(int i=0;i<NUM_TREES_IN_FOREST;i++){
            estimate += EstimateCountHelper(tree_list_[i],mbr); 
            // std::cout<<"\t\t EstimateCountHelper:: Estimate after "<<i+1<<" trees:"<<(estimate/(i+1))<<"\n";
        }
        return estimate/NUM_TREES_IN_FOREST; 
    }


    

    /** Recursive helper for `EstimateCount`. */
    /**
    * @brief The recursive function that goes through the density estimation tree and calculates
    *        counts within a range query.
    */
    double_t EstimateCountHelper(DensEstNode* node,BoundingRectangle mbr){
        
        /* IF the node region is completely inside the query range return the whole count.*/
        if(mbr.IsCompletelyCovering(node->mbr_)){
            return node->counts_;
        }

        /* Returning if the current tree node has no overlap with query region.*/
        if(!(node->mbr_.IsThereOverlap(mbr)))
            return 0L;


            /* return the value equivalent to ratio of overlap region between 
            *        query region and node region times the number of points present in the node.*/
        if(node->is_leaf_){
            return  node->counts_* node->mbr_.RatioOfOverlap(mbr);
        }

        /* recurse for lower level nodes if non of the above criteria are met*/
        return EstimateCountHelper(node->children_[0],mbr) + EstimateCountHelper(node->children_[1],mbr);
        
    }


};


#endif
