#ifndef RTREE_NODE_H
#define RTREE_NODE_H

#include"../utils/bounding_rectangle.h"
#include"../utils/density_estimators/weighted_dens_est.h"

#include<vector>
#include<tuple>

/**
 * Node used by the R-tree family.
 */
class RTreeNode{
    public:
        BoundingRectangle mbr_;
        std::vector<RTreeNode*> children_;
        
        size_t local_block_id_;
        bool is_leaf_;
        bool cached_scan_cost_valid_;
        double_t cached_scan_cost_;

        /** Create an empty leaf node. */
        RTreeNode(){
            is_leaf_=true;
            local_block_id_=0;
            cached_scan_cost_valid_=false;
            cached_scan_cost_=0.0;
        }

        /** Recursively delete the child subtrees. */
        ~RTreeNode(){
            for(auto child: children_)
                delete child;
        }

        
};


/**
 * Temporary statistics bundle used while choosing an insertion/split target.
 */
class NodeStatsPointInsert{
    public:
        size_t child_pointer_id_;
        BoundingRectangle expanded_mbr_;

        double_t volume_{};
        double_t delta_volume_{};

        double_t perim_{};
        double_t delta_perim_{};

        double_t delta_overlap_volume_{};               // Sum of volume overlap change between current cell and other siblings

        double_t cost_{};
        double_t delta_cost_{};

        /** Default-construct an empty statistics bundle. */
        NodeStatsPointInsert(){}

        /** Build a statistics bundle from already computed scalar metrics. */
        NodeStatsPointInsert(size_t cid, double_t v, double_t dv, double_t p, double_t dp, double_t dov): child_pointer_id_(cid), volume_(v), delta_volume_(dv), perim_(p), delta_perim_(dp), delta_overlap_volume_(dov){}

        /** Construct the R*-tree insertion metrics for one child candidate. */
        NodeStatsPointInsert(size_t cid, BoundingRectangle& mbr, Point& insert_pnt): child_pointer_id_(cid),expanded_mbr_(mbr){
            volume_= mbr.Area();
            perim_ = mbr.Perimeter();

            expanded_mbr_.UpdateBoundingBoxWithPoint(insert_pnt);
            delta_volume_ = expanded_mbr_.Area()-volume_;
            delta_perim_ = expanded_mbr_.Perimeter()-perim_;
            
        }

        /** Construct the query-weighted insertion metrics for RWTree. */
        NodeStatsPointInsert(size_t cid, BoundingRectangle& mbr, Point& insert_pnt,WeightedDensEstTree * weighted_datapoint_density_estimator_): child_pointer_id_(cid),expanded_mbr_(mbr){
            volume_= mbr.Area();
            perim_ = mbr.Perimeter();
            cost_ = weighted_datapoint_density_estimator_->EstimateCount(mbr);

            expanded_mbr_.UpdateBoundingBoxWithPoint(insert_pnt);
            delta_volume_ = expanded_mbr_.Area()-volume_;
            delta_perim_ = expanded_mbr_.Perimeter()-perim_;
            delta_cost_ = weighted_datapoint_density_estimator_->EstimateCount(expanded_mbr_)-cost_;
            
        }

};  

#endif
