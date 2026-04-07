#ifndef KDTREE_NODE_H
#define KDTREE_NODE_H

#include"../utils/bounding_rectangle.h"
#include<vector>

/**
 * Node used by the binary KD-tree variants.
 */
class KDTreeNode{
    public:
        BoundingRectangle mbr_;
        KDTreeNode* children_[2];
        
        size_t local_block_id_;
        bool is_leaf_;

        size_t split_dim_;
        double_t split_value_;

        /** Create an empty leaf node. */
        KDTreeNode(){
            is_leaf_=true;
            local_block_id_=0;
        }

        /** Create an empty leaf node pre-seeded with an MBR. */
        KDTreeNode(const BoundingRectangle& mbr):mbr_(mbr){
            is_leaf_=true;
            local_block_id_=0;
        }


        /** Recursively delete the child subtrees. */
        ~KDTreeNode(){
            delete children_[0];
            delete children_[1];
        }
        
};


#endif
