#ifndef RSMI_TREE_H
#define RSMI_TREE_H

#include <torch/script.h> // One-stop header.
#include <memory>
#include<vector>
#include<utility>
#include<tuple>
#include<iterator>
#include<string>
#include<array>
#include<algorithm>
#include<fstream>
#include <cmath>
#include<cassert>

#include"../utils/bounding_rectangle.h"
#include"../utils/local_model.h"

class RSMINode{
    public:
        BoundingRectangle mbr_;
        std::vector<RSMINode*> children_;

        /* Since there might be random missing nodes due to trained model split we use a map to remap the children.*/
        std::map<int,int> child_map_;              
        torch::jit::script::Module module_;
        size_t num_buckets_;              // the original training tried to split data this many buckets

        size_t local_model_id_;
        bool is_leaf_;

        RSMINode(){
            is_leaf_=true;
            local_model_id_=0;
        }

        ~RSMINode(){
            for(auto child: children_)
                delete child;
        }

        
};


class RSMITree{
    public:
        RSMINode* root_;
        std::vector<LocalModel> cell_list_; 
        BlockStore block_store_;
        size_t node_cnt_{};

        RSMITree(){}

        RSMITree(std::string foldername,std::vector<Point>& data){
            root_ = new RSMINode();


            /* Reading the tree structure and train pytorch models*/
            std::ifstream fin(foldername+"tree.txt");
            LoadRSMITree(fin, root_, foldername);
            fin.close();


            /* Inserting points into the tree*/
            InsertDataIntoRSMI(root_,data);

            block_store_.FinishedConstruction();
        }

        size_t LoadRSMITree(std::ifstream& fin,RSMINode* node,std::string foldername){
            std::string torchscript_filename;
            size_t num_children,child_num;
            double_t a,b,c,d;
            fin>>num_children>>node->num_buckets_>>torchscript_filename>>child_num>>a>>b>>c>>d;


            if(num_children!=0){
                node->module_ = torch::jit::load(foldername+torchscript_filename+".pt");

                node->is_leaf_=false;
                for(size_t i=0;i<num_children;i++){
                    RSMINode* new_child = new RSMINode();
                    node->child_map_[LoadRSMITree(fin,new_child,foldername)]=i;
                    node->children_.push_back(new_child);
                }
            }
            return child_num;
        }


        void InsertDataIntoRSMI(RSMINode* node,std::vector<Point>& data){
            
            
            /*If at leaf node put all points into local model*/
            if(node->is_leaf_){
                cell_list_.emplace_back(LocalModel(data.begin(),data.end(),block_store_));
                
                node->local_model_id_=cell_list_.size()-1;
                node->mbr_=cell_list_[node->local_model_id_].local_model_mbr_;
                return;
            }



            /* Otherwise split data into as many leaves as required.*/
            std::vector<std::vector<Point>> split_data(node->children_.size());

            for(auto& pt: data){
                std::vector<torch::jit::IValue> inputs{torch::tensor({pt.elements_[0], pt.elements_[1]})};
                // double_t nn_output = node->module_.forward(inputs).toTensor().item<double_t>();
                // int bucket = std::round(nn_output*(node->num_buckets_-1));
                // bucket = (bucket<0)?0:bucket;
                // bucket = (bucket>=node->num_buckets_-1)?node->num_buckets_-1:bucket;
                // auto nn_output=node->module_.forward(inputs);//.toTensor(); 
                // // std::cout<<nn_output.sizes()<<"\n";

                // std::cout<<nn_output.argmax().item().toInt()<<"\n";
                // std::cin.get();


                int bucket = node->module_.forward(inputs).toTensor().argmax().item().toInt();

                //There is a problem if the mapped node was not present during construction.
                assert( node->child_map_.contains(bucket));

                
                int child_num = node->child_map_[bucket];
                
                split_data[child_num].push_back(pt);



            }

            node->mbr_.SetToDefault();
            for(int child_num=0;child_num<node->children_.size();child_num++){
                InsertDataIntoRSMI(node->children_[child_num],split_data[child_num]);
                node->mbr_.UpdateBoundingBoxWithBoundingBox(node->children_[child_num]->mbr_);
            }

        }

        /* TODO: Range query required functions. */

        std::vector<Point> RangeQuery(Query& query){
            std::vector<size_t> projected_cells;
            Projection(projected_cells,query,root_);

            std::vector<size_t> refined_blocks;
            Refinement(refined_blocks,query,projected_cells);

            std::vector<Point> result_vec;
            Scan(refined_blocks,query,result_vec);
            return result_vec;
        }

        void Projection(std::vector<size_t> &projected_cells, Query& query, RSMINode* node){
            if(node->is_leaf_){
                projected_cells.push_back(node->local_model_id_);
                return;
            }

            std::vector<torch::jit::IValue> inputs_low{torch::tensor({query.low_.elements_[0], query.low_.elements_[1]})};
            // double_t nn_output_low = node->module_.forward(inputs_low).toTensor().item<double_t>();
            // int start_bucket = std::round(nn_output_low*(node->num_buckets_-1));
            // start_bucket = (start_bucket<0)?0:start_bucket;
            // start_bucket = (start_bucket>=node->num_buckets_-1)?node->num_buckets_-1:start_bucket;
            int start_bucket = node->module_.forward(inputs_low).toTensor().argmax().item().toInt();

            std::vector<torch::jit::IValue> inputs_high{torch::tensor({query.high_.elements_[0], query.high_.elements_[1]})};
            // double_t nn_output_high = node->module_.forward(inputs_high).toTensor().item<double_t>();
            // int end_bucket = std::round(nn_output_high*(node->num_buckets_-1));
            // end_bucket = (end_bucket<0)?0:end_bucket;
            // end_bucket = (end_bucket>=node->num_buckets_-1)?node->num_buckets_-1:end_bucket;
            int end_bucket = node->module_.forward(inputs_high).toTensor().argmax().item().toInt();




            for(size_t bucket_num = start_bucket;bucket_num<=end_bucket;bucket_num++)
                if(node->child_map_.contains(bucket_num))
                    Projection(projected_cells,query,node->children_[node->child_map_[bucket_num]]);
        }

        std::vector<Point> AccurateRangeQuery(Query& query){
            std::vector<size_t> projected_cells;
            AccurateProjection(projected_cells,query,root_);

            std::vector<size_t> refined_blocks;
            Refinement(refined_blocks,query,projected_cells);

            std::vector<Point> result_vec;
            Scan(refined_blocks,query,result_vec);
            return result_vec;
        }

        void AccurateProjection(std::vector<size_t> &projected_cells, Query& query, RSMINode* node){
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