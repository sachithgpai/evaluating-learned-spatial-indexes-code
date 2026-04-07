#ifndef KDTREE_H
#define KDTREE_H

#include<vector>
#include<utility>
#include<tuple>
#include<iterator>
#include<string>
#include<array>

#include"../utils/local_model.h"
#include"../utils/query.h"
#include"kdtree_node.h"
#include"../utils/sort_tools.h"
#include"../utils/constants.h"

#include<algorithm>

/**
 * Binary KD-tree with block-based leaves and scan-based refinement.
 */
class KDTree{
    public:
        KDTreeNode* root_;
        BlockStore block_store_;
        size_t node_count_{};

        /** Create an empty KD-tree shell. */
        KDTree(){}

        /** Build a KD-tree from an in-memory point set. */
        KDTree(std::vector<Point> data){
            root_ = new KDTreeNode();
            root_->mbr_.SetToSpanWholeSpace();
            node_count_++;
            BuildKDTree(root_,data.begin(),data.end(),0);
            block_store_.FinishedConstruction();
        }

        /** Recursively build the KD-tree rooted at `node`. */
        void  BuildKDTree(KDTreeNode* node, std::vector<Point>::iterator it_data_begin,std::vector<Point>::iterator it_data_end,size_t sort_dim){
            
            int num_datapoints = std::distance(it_data_begin,it_data_end);
            // std::cout<<"Node with "<<num_datapoints<<" points ";
            // node->mbr_.Print();

            if(num_datapoints<=BLOCK_SIZE){ 
                std::sort(it_data_begin, it_data_end, SortOrderer(Constants::LEAF_SORT_DIM));
                node->local_block_id_ = block_store_.InsertNewBlock(it_data_begin,it_data_end);
                return;
            }
            node->is_leaf_ = false;


            std::sort(it_data_begin, it_data_end, SortOrderer(sort_dim));
            double_t low = it_data_begin->elements_[sort_dim], high = std::prev(it_data_end)->elements_[sort_dim];
            node->split_value_= (high+low)/2.0;
            node->split_dim_=sort_dim;

            Point bs_obj;
            bs_obj.elements_[sort_dim] =  node->split_value_;

            auto bs_comp = [sort_dim](Point& pt1_iter,Point pt2) { return pt1_iter.elements_[sort_dim]< pt2.elements_[sort_dim]; };
            std::vector<Point>::iterator it_data_split = lower_bound(it_data_begin,it_data_end,bs_obj,bs_comp);

            node->children_[0] = new KDTreeNode(node->mbr_);
            node->children_[0]->mbr_.high_.elements_[sort_dim] = node->split_value_;
            BuildKDTree(node->children_[0],it_data_begin,it_data_split,(sort_dim+1)%Constants::DIM);

            node->children_[1] = new KDTreeNode(node->mbr_);
            node->children_[1]->mbr_.low_.elements_[sort_dim] = node->split_value_;
            BuildKDTree(node->children_[1],it_data_split,it_data_end,(sort_dim+1)%Constants::DIM);

            node_count_+=2;
        }

        /** Execute a range query by projection followed by block scanning. */
        std::vector<Point> RangeQuery(Query& query){

            std::vector<size_t> projected_cell_ids;
            Projection(projected_cell_ids,query,root_);

            std::vector<Point> result_vec;
            Scan(projected_cell_ids,query,result_vec);

            return result_vec;
        }

        /** Collect the leaf blocks whose KD-tree regions overlap the query. */
        void Projection(std::vector<size_t> &projected_cell_ids, Query& query, KDTreeNode* node){
            if(node->is_leaf_){
                projected_cell_ids.push_back(node->local_block_id_);
                return;
            }
            if(query.IsThereOverlap(node->children_[0]->mbr_))
                Projection(projected_cell_ids,query,node->children_[0]);
            
            if(query.IsThereOverlap(node->children_[1]->mbr_))
                Projection(projected_cell_ids,query,node->children_[1]);
        }

        /** Scan the projected blocks and append matches into `result_vec`. */
        void Scan(std::vector<size_t> &projected_cell_ids, Query& query, std::vector<Point> &result_vec){
            std::sort(projected_cell_ids.begin(),projected_cell_ids.end());
            block_store_.FilterPointsFromBlocksForQuery(query, projected_cell_ids, result_vec);
        }

};

#endif
