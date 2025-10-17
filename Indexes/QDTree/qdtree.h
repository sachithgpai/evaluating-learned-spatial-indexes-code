#ifndef QDTREE_H
#define QDTREE_H


#include<vector>
#include<utility>
#include<tuple>
#include<iterator>
#include<string>
#include<array>
#include<cstdlib>
#include<cassert>
#include<iostream>


#include"../utils/local_model.h"
#include"../utils/query.h"
#include"../utils/sort_tools.h"
#include"../utils/density_estimators/dens_est.h"
#include"../KDTree/kdtree.h"


class QDTree:public KDTree{
    public:
        ComparatorPointPartition temp_comparator_point_;  // dummy point for use in partition() with BulkLoad


        QDTree(std::vector<Point> data,size_t TARGET_BLOCK_SIZE=BLOCK_SIZE,bool disk_backup = true){
            root_ = new KDTreeNode();
            root_->mbr_.SetToSpanWholeSpace();

            srand(time(NULL)); 
            BuildRandomQdTree(root_,data.begin(),data.end(),TARGET_BLOCK_SIZE);
            if(disk_backup) block_store_.FinishedConstruction();
        }

        

        /*Function to create a random QDTree (Same as BuildKDTree with random dim for split and pick median)*/
        void BuildRandomQdTree(KDTreeNode *node,std::vector<Point>::iterator it_data_begin,std::vector<Point>::iterator it_data_end,size_t TARGET_BLOCK_SIZE=BLOCK_SIZE){

            int num_datapoints = std::distance(it_data_begin,it_data_end);
            if(num_datapoints<=TARGET_BLOCK_SIZE){ 
                std::sort(it_data_begin, it_data_end, SortOrderer(Constants::LEAF_SORT_DIM));
                node->local_block_id_ = block_store_.InsertNewBlock(it_data_begin,it_data_end);
                return;
            }
            node->is_leaf_ = false;


            // Random flip dimension.
           size_t sort_dim =rand()%2; 

            // sort data along that dimension.
            std::sort(it_data_begin, it_data_end, SortOrderer(sort_dim));
            // pick mid point to create predicate
            double_t low = it_data_begin->elements_[sort_dim], high = std::prev(it_data_end)->elements_[sort_dim];
            node->split_value_ = (high+low)/2.0;
            node->split_dim_ = sort_dim;

            Point bs_obj;
            bs_obj.elements_[sort_dim] =  node->split_value_;

            auto bs_comp = [sort_dim](Point& pt1_iter,Point pt2) { return pt1_iter.elements_[sort_dim]< pt2.elements_[sort_dim]; };
            std::vector<Point>::iterator it_data_split = lower_bound(it_data_begin,it_data_end,bs_obj,bs_comp);
            node->children_[0] = new KDTreeNode(node->mbr_);
            node->children_[0]->mbr_.high_.elements_[node->split_dim_] = node->split_value_;
            BuildRandomQdTree(node->children_[0],it_data_begin,it_data_split);

            node->children_[1] = new KDTreeNode(node->mbr_);
            node->children_[1]->mbr_.low_.elements_[node->split_dim_] = node->split_value_;
            BuildRandomQdTree(node->children_[1],it_data_split,it_data_end);

            node_count_+=2;
        }

        // Load QD-tree from a file
        QDTree(std::vector<Point> data,std::string filename){

            root_ = new KDTreeNode();
            root_->mbr_.SetToSpanWholeSpace();
            node_count_++;

            LoadQDTree(filename);
            BulkLoadData(root_,data.begin(),data.end());

            block_store_.FinishedConstruction();
        }

        void LoadQDTree(std::string file){
            std::ifstream fin(file);
            std::cout<<"Open file"<<file<<"\n";

            double_t split_val;
            uint64_t  id;
            size_t split_dim;
            
            while (!fin.eof())
            {
                fin>>id>>split_dim>>split_val;
                InsertNodeToTree(id,split_dim,split_val);
            }
        }


        /**
         * @brief Helper function to  `loadTree` to insert each node
         */
        void InsertNodeToTree(uint64_t id,size_t split_dim,double_t split_val){
            std::vector<uint32_t> path;            
            while(id!=0){
                path.push_back((id-1)%2);
                id=(id-1)/2;
            }
            std::reverse(path.begin(), path.end());

            KDTreeNode* curr_node = root_;
            for(uint32_t childId : path)
                curr_node = curr_node->children_[childId];
            
            curr_node->split_dim_ = split_dim;
            curr_node->split_value_ = split_val;
            curr_node->is_leaf_ = false;

            curr_node->children_[0]= new KDTreeNode(curr_node->mbr_);
            curr_node->children_[0]->mbr_.high_.elements_[split_dim]=split_val;

            curr_node->children_[1]= new KDTreeNode(curr_node->mbr_);
            curr_node->children_[1]->mbr_.low_.elements_[split_dim]=split_val;

            node_count_+=2;
        
        }

        /** 
         * @brief Helper function to bulk load data.
         */
        void BulkLoadData(KDTreeNode *node,std::vector<Point>::iterator it_data_begin,std::vector<Point>::iterator it_data_end){
            
            if(node->is_leaf_){
                BuildRandomQdTree(node,it_data_begin,it_data_end);
                return;
            }

            
            
            temp_comparator_point_.SetDimSplitValue(node->split_dim_,node->split_value_);
            std::vector<Point>::iterator it_0_data_begin=it_data_begin,it_0_data_end=std::partition(it_data_begin,it_data_end,temp_comparator_point_);
            std::vector<Point>::iterator it_1_data_begin=it_0_data_end,it_1_data_end=it_data_end;

            BulkLoadData(node->children_[0],it_0_data_begin,it_0_data_end);
            BulkLoadData(node->children_[1],it_1_data_begin,it_1_data_end);

        }




        void SaveQDTree(std::ofstream& fout,size_t node_id, KDTreeNode* node){

            fout<<node_id<<" "<<node->split_dim_<<" "<<node->split_value_<<"\n";
            if(node->children_[0]->is_leaf_==false)
                SaveQDTree(fout, (node_id*2)+1,node->children_[0]);
            if(node->children_[1]->is_leaf_==false)
                SaveQDTree(fout, (node_id*2)+2,node->children_[1]);
            
        }

        
};





// Random sampling based QD-Tree
void QDTreeTrainerRandomSearch(std::vector<Point> data,std::vector<Query> &queries,std::string filename,double_t sampling_ratio = 0.1){
    
    srand(time(NULL)); 

    double_t best_cost = std::numeric_limits<double_t>::max(),temp_cost =0;
    int since_last_improvement=0,max_samples=500,sampled_data_size = data.size()*sampling_ratio;

    std::vector<Point> sampled_data(data.begin(),data.begin()+sampled_data_size);
    while(since_last_improvement<25 && max_samples>0){
        QDTree qd_tree_obj(sampled_data,BLOCK_SIZE*sampling_ratio,false); // setting bool disk_backup = false

        std::vector<size_t> projected_cell_ids;
        for(auto& query:queries)
            qd_tree_obj.Projection(projected_cell_ids,query,qd_tree_obj.root_);

        temp_cost = qd_tree_obj.block_store_.NumOfPointsInBlocks(projected_cell_ids);
        
        if(temp_cost < best_cost){
            std::cout<<(500-max_samples)<<" Improved cost from "<< best_cost<<" to "<<temp_cost<<" \n";
            best_cost = temp_cost;
            ofstream best_tree(filename,ios_base::out);
            qd_tree_obj.SaveQDTree(best_tree,0,qd_tree_obj.root_);
            best_tree.close();
            since_last_improvement=0;

        }
        since_last_improvement++;
        max_samples--;

    }

}

#endif