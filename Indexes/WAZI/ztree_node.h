
/**
 * @file node.h
 * @author Sachith Pai (sachith.pai@helsinki.fi)
 * @brief This file contains the nodes to be used in our tree.
 * @version 0.1
 * @date 2022-04-25
 * 
 */


#ifndef ZTREE_NODE_H
#define ZTREE_NODE_H


#include<vector>
#include<cmath>
#include<iostream>
#include<algorithm>
#include<list>
#include"../utils/point.h"
#include"../utils/bounding_rectangle.h"
#include"ztree_leaflist_metadata.h"

/**
 * @brief Node for intermidiate node of a ZTree.
 */
class ZtreeNode{
    public:
        bool is_leaf_;
        bool ordering_;
        Point partition_;
        BoundingRectangle mbr_;
        ZtreeNode* children_[4];
        size_t leaf_id_{};
        uint32_t pages_in_subtree_{},node_depth_;

        
        ZtreeNode(uint32_t depth=0):node_depth_(depth){
            ordering_ = false;
            is_leaf_ = false;
        }

        void Print(){
            std::cout<<" (";
            for(auto i=0;i<Constants::DIM;i++)
                std::cout<<mbr_.low_.elements_[i]<<((i+1==Constants::DIM)?" ) ":" , ");

            std::cout<<" (";
            for(auto i=0;i<Constants::DIM;i++)
                std::cout<<mbr_.high_.elements_[i]<<((i+1==Constants::DIM)?" ) ":" , ");


            std::cout<<"   X  (";
            for(auto i=0;i<Constants::DIM;i++)
                std::cout<<partition_.elements_[i]<<((i+1==Constants::DIM)?" ) ":" , ");

            std::cout<<"\n";
        }

        
};





#endif
