#ifndef STRTREE_H
#define STRTREE_H


#include<vector>
#include<utility>
#include<tuple>
#include<iterator>
#include<string>
#include<array>
#include<fstream>


#include"../utils/local_model.h"
#include"../utils/query.h"
#include"rtree_node.h"
#include"rtree_base.h"
#include"../utils/sort_tools.h"
#include"../utils/constants.h"


/**
 * Sort-Tile-Recursive style R-tree builder with block-based leaves.
 */
class STRTree: public RTreeBASE {
    public:


        /** Create an empty STRTree shell. */
        STRTree(){}

        /** Build an STRTree from an in-memory point set. */
        STRTree(std::vector<Point> data){
            root_ = BuildSTRTree(data.begin(),data.end(),0);
            block_store_.FinishedConstruction();
        }

        /** Reconstruct an STRTree from the serialized format. */
        STRTree(string filename){
            root_ = new RTreeNode();
            std::ifstream fin(filename);
            ReadTree(fin,root_);
            fin.close();

            block_store_.FinishedConstruction();
        }


        /** Recursively build and return one STRTree subtree. */
        RTreeNode* BuildSTRTree(std::vector<Point>::iterator it_data_begin,std::vector<Point>::iterator it_data_end,size_t sort_dim){

            RTreeNode* node = new RTreeNode();
            node_cnt_++;

            int num_datapoints = std::distance(it_data_begin,it_data_end);
            
            if(num_datapoints<=BLOCK_SIZE){ 
                std::sort(it_data_begin, it_data_end, SortOrderer(Constants::LEAF_SORT_DIM));
                node->local_block_id_ = block_store_.InsertNewBlock(it_data_begin,it_data_end);
                node->mbr_=block_store_.FetchBoundingBoxForBlock(node->local_block_id_);
                return node;
            }

            node->is_leaf_=false;
            std::sort(it_data_begin, it_data_end, SortOrderer(sort_dim));



            /*If the next split creates too small nodes reduce the number of splits to avoid tiny pages*/
            int num_elements_per_child = int(num_datapoints/STR_BRANCH_FACTOR);
            int left_over_elements = num_datapoints%STR_BRANCH_FACTOR;
            int branch_factor = STR_BRANCH_FACTOR;

            if(num_elements_per_child<BLOCK_SIZE){
                branch_factor = int(num_datapoints/BLOCK_SIZE)+1;
                num_elements_per_child = int(num_datapoints/branch_factor);
                left_over_elements = num_datapoints%branch_factor;
            }

            for(int j=0,itr_start=0,itr_end=0;j<branch_factor;j++,left_over_elements--){

                itr_end += num_elements_per_child +(left_over_elements>0);
                itr_end = std::min(itr_end,num_datapoints);

                RTreeNode* child = BuildSTRTree(it_data_begin+itr_start,it_data_begin+itr_end,(sort_dim+1)%Constants::DIM);
                node->children_.push_back(child);
                node->mbr_.UpdateBoundingBoxWithBoundingBox(child->mbr_);
                itr_start=itr_end;
            }
            return node;
        }

};

#endif
