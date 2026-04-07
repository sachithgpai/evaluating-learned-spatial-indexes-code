#ifndef DENS_EST_NODE_H
#define DENS_EST_NODE_H

#include"../bounding_rectangle.h"
#include<cstdlib>

/**
 * @brief Node used by the density-estimation trees.
 */
class DensEstNode{
    public:
    BoundingRectangle mbr_;
    double_t counts_;
    size_t split_dim_;
    double_t split_location_;
    DensEstNode* children_[2];
    bool is_leaf_;

    /** Initialize an empty leaf node. */
    DensEstNode(){
        counts_ = 0;
        children_[0]=NULL;
        children_[1]=NULL;
        is_leaf_ = true;
    }

    /** Recursively delete the child subtrees. */
    ~DensEstNode(){
        delete children_[0];
        delete children_[1];
    }
    
};

#endif
