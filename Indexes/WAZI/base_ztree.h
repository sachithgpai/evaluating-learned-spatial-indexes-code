/**
 * @file Base_ztree.h
 * @author Sachith (sachith.pai@helsinki.fi)
 * @brief File to train a Base tree.
 * @version 0.1
 * @date 2022-05-04
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#ifndef BASE_ZTREE_H
#define BASE_ZTREE_H

#include<cstdlib>
#include<vector>
#include<algorithm>
#include<iterator>
#include"ztree.h"
#include"../utils/point.h"
#include"../utils/sort_tools.h"
#include"../utils/constants.h"


/**
 * Median-split baseline ZTree builder.
 */
class BaseZTree: public ZTree{
    public:

    /** Build the baseline ZTree from a dataset. */
    BaseZTree(std::vector<Point> &dataset,bool skipping=false):ZTree(skipping){
        num_datapoints_ = dataset.size();
        double_t data_low_x,data_low_y,data_high_x,data_high_y;

        std::sort(dataset.begin(),dataset.end(),SortOrderer(SortX));
        data_low_x=dataset[0].elements_[0];
        data_high_x=dataset[num_datapoints_-1].elements_[0];


        std::sort(dataset.begin(),dataset.end(),SortOrderer(SortY));
        data_low_y=dataset[0].elements_[1];
        data_high_y=dataset[num_datapoints_-1].elements_[1];
    


        root_=new ZtreeNode();
        node_cnt_++;
        root_->mbr_.low_=Point(data_low_x,data_low_y);
        root_->mbr_.high_=Point(data_high_x,data_high_y);


        BuildTree(root_,dataset.begin(),dataset.end());

        // CreateLocalModels(root_);
        // SetLeafPageMask(root_);

        SetLeafPageMaskCreateLocalModels(root_);


        if(is_skipping_aware_)
            CalculateFwdPointers();
        BulkLoadData(dataset);

        
    }

    /** Recursively build the baseline ZTree structure. */
    /**
     * @brief Function takes the data points and build a Ztree. 
     * Should be identical for all versions of Ztree
     */
    void BuildTree(ZtreeNode *node, std::vector<Point>::iterator it_data_begin, std::vector<Point>::iterator it_data_end){
        
        node->partition_ = FindOptimalSplitPoint(node,it_data_begin,it_data_end);
        // node->partition_.Print();
        // std::cin.get();
        if(node->is_leaf_){
            node->pages_in_subtree_=1;
            return;
        }



        for(size_t i=0;i<4;i++)
            node->children_[i]= new ZtreeNode(node->node_depth_+1);
        node_cnt_+=4;

        std::vector<Point>::iterator it_A_data_begin=it_data_begin,it_B_data_begin,it_C_data_begin,it_D_data_begin;
        std::vector<Point>::iterator it_A_data_end,it_B_data_end,it_C_data_end,it_D_data_end=it_data_end;

        ComparatorPointPartition x_partition_predicate(node->partition_,0);
        ComparatorPointPartition y_partition_predicate(node->partition_,1);
        

        it_B_data_begin = std::partition(it_data_begin,it_data_end,x_partition_predicate);
        it_C_data_end = it_B_data_begin;

        it_C_data_begin = std::partition(it_A_data_begin,it_C_data_end,y_partition_predicate);
        it_A_data_end = it_C_data_begin;

        it_D_data_begin = std::partition(it_B_data_begin,it_D_data_end,y_partition_predicate);
        it_B_data_end = it_D_data_begin;


        node->children_[0]->mbr_ = BoundingRectangle(
                                Point(node->mbr_.low_.elements_[0], node->mbr_.low_.elements_[1]),
                                Point(node->partition_.elements_[0], node->partition_.elements_[1]));
        BuildTree(node->children_[0],it_A_data_begin,it_A_data_end);

        node->children_[1]->mbr_ = BoundingRectangle(
                                Point(node->partition_.elements_[0], node->mbr_.low_.elements_[1]),
                                Point(node->mbr_.high_.elements_[0], node->partition_.elements_[1]));
        BuildTree(node->children_[1],it_B_data_begin,it_B_data_end);

        node->children_[2]->mbr_ = BoundingRectangle(
                                Point(node->mbr_.low_.elements_[0], node->partition_.elements_[1]),
                                Point(node->partition_.elements_[0], node->mbr_.high_.elements_[1]));
        BuildTree(node->children_[2],it_C_data_begin,it_C_data_end);

        node->children_[3]->mbr_ = BoundingRectangle(
                                Point(node->partition_.elements_[0], node->partition_.elements_[1]),
                                Point(node->mbr_.high_.elements_[0], node->mbr_.high_.elements_[1]));
        BuildTree(node->children_[3],it_D_data_begin,it_D_data_end);
        

        node->pages_in_subtree_ = node->children_[0]->pages_in_subtree_+ 
                                    node->children_[1]->pages_in_subtree_+
                                    node->children_[2]->pages_in_subtree_+
                                    node->children_[3]->pages_in_subtree_;
        
        return;
    }

    /** Pick the median split point in each dimension for one node. */
    Point FindOptimalSplitPoint(ZtreeNode *node,
                            std::vector<Point>::iterator it_data_begin,std::vector<Point>::iterator it_data_end){

        uint32_t num_data_here = std::distance(it_data_begin,it_data_end);

        if(num_data_here<=BLOCK_SIZE){
            node->is_leaf_ = true;
            return Point(-1.0,-1.0);
        }

        Point split = Point(-1.0,-1.0);
        size_t mid_ix = num_data_here/2;

        for(size_t d = 0;d<Constants::DIM;d++){        
            std::sort(it_data_begin,it_data_end,SortOrderer(d));
            split.elements_[d]=(*(it_data_begin+mid_ix)).elements_[d];//-Constants::EPSILON_ERR;
        }
        
        return split;
    }

};


 #endif
