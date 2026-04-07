#ifndef CURTREE_H
#define CURTREE_H


#include<vector>
#include<utility>
#include<tuple>
#include<iterator>
#include<string>
#include<array>
#include<iomanip>


#include"../utils/local_model.h"
#include"../utils/query.h"
#include"rtree_node.h"
#include"rtree_base.h"
#include"../utils/sort_tools.h"
#include"../utils/density_estimators/weighted_dens_est.h"


/**
 * Query-weighted recursive partitioning tree used by the CUR baseline.
 */
class CURTree: public RTreeBASE{
    public:


        /** Create an empty CURTree shell. */
        CURTree(){}
        
        /** Reconstruct a CURTree from the serialized format. */
        CURTree(string filename){
            root_ = new RTreeNode();
            std::ifstream fin(filename);
            ReadTree(fin,root_);
            fin.close();

            block_store_.FinishedConstruction();
        }

        /** Build a CURTree from data plus the training query workload. */
        CURTree(std::vector<Point> data, std::vector<Query> queries){
            std::cout << std::setprecision(2) << std::fixed;

            node_cnt_=0;

            std::vector<WrappedPoint> wrapped_data = WeightPointsWithQuery(data,queries);
            root_ = BuildTree(wrapped_data.begin(),wrapped_data.end(),0);

            block_store_.FinishedConstruction();
        }
        /** Recursively build a CURTree subtree over weighted points. */
        RTreeNode* BuildTree(std::vector<WrappedPoint>::iterator vec_begin,std::vector<WrappedPoint>::iterator vec_end,size_t sort_dim){
            
            RTreeNode* node = new RTreeNode();
            node_cnt_++;
            int num_datapoints = std::distance(vec_begin,vec_end);
            // std::cout<<" num_data_points "<<num_datapoints<<"\n"<<"\t";
            // (*vec_begin).Print();
            // std::cin.get();
             


            if(num_datapoints<=BLOCK_SIZE){ //if the points fit within one block
                std::vector<Point> block_data;
                for(auto it=vec_begin;it!=vec_end;it++) block_data.push_back(it->ExtractPoint());

                node->local_block_id_ = block_store_.InsertNewBlock(block_data.begin(),block_data.end());
                node->mbr_=block_store_.FetchBoundingBoxForBlock(node->local_block_id_);
                return node;
            }


            node->is_leaf_=false;


            /* If the number of pages from current split is less than 10 split evenly.*/
            if(num_datapoints<=BLOCK_SIZE*CUR_BRANCH_FACTOR){
                int num_elements_per_child = int(num_datapoints/CUR_BRANCH_FACTOR);
                int left_over_elements = num_datapoints%CUR_BRANCH_FACTOR;
                int branch_factor = CUR_BRANCH_FACTOR;

                if(num_elements_per_child<BLOCK_SIZE){
                    branch_factor = int(num_datapoints/BLOCK_SIZE)+1;
                    num_elements_per_child = int(num_datapoints/branch_factor);
                    left_over_elements = num_datapoints%branch_factor;
                }

                for(int j=0,itr_start=0,itr_end=0;j<branch_factor;j++,left_over_elements--){

                    itr_end += num_elements_per_child +(left_over_elements>0);
                    itr_end = std::min(itr_end,num_datapoints);

                    RTreeNode* child = BuildTree(vec_begin+itr_start,vec_begin+itr_end,(sort_dim+1)%Constants::DIM);
                    node->children_.push_back(child);
                    node->mbr_.UpdateBoundingBoxWithBoundingBox(child->mbr_);
                    itr_start=itr_end;
                }
                return node;
            }


            /*If the next split creates too small nodes reduce the number of splits to avoid tiny pages*/
            int num_elements_per_child = int(num_datapoints/CUR_BRANCH_FACTOR);
            int left_over_elements = num_datapoints%CUR_BRANCH_FACTOR;
            int branch_factor = CUR_BRANCH_FACTOR;
            if(num_elements_per_child<BLOCK_SIZE){
                branch_factor = int(num_datapoints/BLOCK_SIZE)+1;
                num_elements_per_child = int(num_datapoints/branch_factor);
                left_over_elements = num_datapoints%branch_factor;
            }

            std::sort(vec_begin, vec_end, WrappedPointSortOrderer(sort_dim));

            double_t weighted_num_datapoints = 0;
            std::vector<double_t>  cumulative_weight_arr;
            for(auto it=vec_begin;it!=vec_end;it++) {
                weighted_num_datapoints+=it->num_queries_overlapping_;
                cumulative_weight_arr.push_back(weighted_num_datapoints);
            }

            double_t weight_per_child = weighted_num_datapoints/branch_factor;

            // std::cout<<"BuildTree "<<num_datapoints<<" Weight per child:"<< weight_per_child<<"/"<<weighted_num_datapoints<<" BranchFactor:"<<branch_factor<<"\n";

            double_t running_weight = 0;
            int itr_end=0;
            for(int j=0,itr_start=0;j<branch_factor;j++){
                
                running_weight+=weight_per_child;
                itr_end = BinarySearch<double_t>(cumulative_weight_arr,running_weight,itr_start,cumulative_weight_arr.size()-1)+1;

                RTreeNode* child = BuildTree(vec_begin+itr_start,vec_begin+itr_end,(sort_dim+1)%Constants::DIM);
                node->children_.push_back(child);
                node->mbr_.UpdateBoundingBoxWithBoundingBox(child->mbr_);
                itr_start=itr_end;
            }

            return node;
        }

};

#endif
