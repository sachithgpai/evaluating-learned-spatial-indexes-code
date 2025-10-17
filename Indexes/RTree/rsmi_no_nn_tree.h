// ! UNUSED. thiw is was using a local model. New version saves it like an RTreeBase.

#ifndef RSMI_NO_NN_TREE_H
#define RSMI_NO_NN_TREE_H

#include<iomanip>
#include<iostream>
#include"../utils/bounding_rectangle.h"
#include"../utils/local_model.h"


class RSMINodeNoNN{
    public:
        BoundingRectangle mbr_;
        std::vector<RSMINodeNoNN*> children_;
        
        size_t local_model_id_;
        bool is_leaf_;

        RSMINodeNoNN(){
            is_leaf_=true;
            local_model_id_=0;
        }

        ~RSMINodeNoNN(){
            for(auto child: children_)
                delete child;
        }

};



/*
An Rtree base class where common Rtree functions like Range query, Write-tree and read-tree are implemented.
*/
class RSMITreeNoNN{
    public:
        RSMINodeNoNN* root_;
        std::vector<LocalModel> cell_list_; 

        BlockStore block_store_;
        size_t node_cnt_{};


        RSMITreeNoNN(){}

        RSMITreeNoNN(string filename){
            std::cout<<"Loading RSMI RTree from "<<filename<<"\n";
            root_ = new RSMINodeNoNN();
            std::ifstream fin(filename);
            ReadTree(fin,root_);
            fin.close();
            block_store_.FinishedConstruction();
        }


        ~RSMITreeNoNN(){
            delete root_;
        }



        
        void ReadTree(std::ifstream& fin,RSMINodeNoNN* node){
            size_t num_children, local_model_size;
            double_t x,y;
            fin>>num_children>>node->mbr_.low_.elements_[0]>>node->mbr_.low_.elements_[1]>>node->mbr_.high_.elements_[0]>>node->mbr_.high_.elements_[1];

            // std::cout<<num_children<<" "; node->mbr_.Print();
            // std::cin.get();

            if(num_children==0){// same as if(node->is_leaf_) TODO:
                fin>>local_model_size;
                std::vector<Point> local_model_data;
                for(size_t i=0;i<local_model_size;i++){
                    fin>>x>>y;
                    local_model_data.emplace_back(x,y);
                }

                cell_list_.emplace_back(LocalModel(local_model_data.begin(),local_model_data.end(),block_store_));
                
                node->local_model_id_=cell_list_.size()-1;
                node->mbr_=cell_list_[node->local_model_id_].local_model_mbr_;

            }
            else{
                node->is_leaf_=false;
                for(size_t i=0;i<num_children;i++){
                    RSMINodeNoNN* new_child = new RSMINodeNoNN();
                    ReadTree(fin,new_child);
                    node->children_.push_back(new_child);
                }
            }
            
        }



        std::vector<Point> RangeQuery(Query& query){
            std::vector<size_t> projected_cells;
            Projection(projected_cells,query,root_);

            std::vector<size_t> refined_blocks;
            Refinement(refined_blocks,query,projected_cells);

            std::vector<Point> result_vec;
            Scan(refined_blocks,query,result_vec);
            return result_vec;
        }

        void Projection(std::vector<size_t> &projected_cells, Query& query, RSMINodeNoNN* node){
            if(node->is_leaf_){
                projected_cells.push_back(node->local_model_id_);
                return;
            }

            for(auto &child: node->children_)
                if(query.IsThereOverlap(child->mbr_))
                    Projection(projected_cells,query,child);
        }

        void Refinement(std::vector<size_t> &refined_blocks, Query &query, std::vector<size_t> &projected_cells)
        {
            for(auto& cell: projected_cells)
                cell_list_[cell].RefinedBlocksForQuery(query,block_store_,refined_blocks);
        } 


        void Scan(std::vector<size_t> &refined_blocks, Query& query, std::vector<Point> &result_vec){
            std::sort(refined_blocks.begin(),refined_blocks.end());
            block_store_.FilterPointsFromBlocksForQuery(query,refined_blocks,result_vec);
        }

};

#endif