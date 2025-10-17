#ifndef KDTREE_NODE_H
#define KDTREE_NODE_H

#include"../utils/bounding_rectangle.h"
#include<vector>

class KDTreeNode{
    public:
        BoundingRectangle mbr_;
        KDTreeNode* children_[2];
        
        size_t local_block_id_;
        bool is_leaf_;

        size_t split_dim_;
        double_t split_value_;

        KDTreeNode(){
            is_leaf_=true;
            local_block_id_=0;
        }

        KDTreeNode(const BoundingRectangle& mbr):mbr_(mbr){
            is_leaf_=true;
            local_block_id_=0;
        }


        ~KDTreeNode(){
            delete children_[0];
            delete children_[1];
        }
        
};


#endif