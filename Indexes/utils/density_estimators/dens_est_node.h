#ifndef DENS_EST_NODE_H
#define DENS_EST_NODE_H

#include"../bounding_rectangle.h"
#include<cstdlib>

/**
 * @brief Class to hold all the node information.
 * 
 */
class DensEstNode{
    public:
    BoundingRectangle mbr_;
    double_t counts_;
    size_t split_dim_;
    double_t split_location_;
    DensEstNode* children_[2];
    bool is_leaf_;

    DensEstNode(){
        counts_ = 0;
        children_[0]=NULL;
        children_[1]=NULL;
        is_leaf_ = true;
    }


    ~DensEstNode(){
        delete children_[0];
        delete children_[1];
    }
    
};

#endif