/**
 * @file Ztree.h
 * @author Sachith Pai (sachith.pai@helsinki.fi)
 * @brief 
 * @version 0.1
 * @date 2022-01-02
 * 
 * @copyright Copyright (c) 2022
 * 
 */


#ifndef ZTREE_H
#define ZTREE_H

#include<vector>
#include<cmath>
#include<iostream>
#include<algorithm>
#include<fstream>
#include<string>
#include<map>
#include<list>
#include<tuple>
#include<chrono>
#include"ztree_node.h"
#include"../utils/sort_tools.h"
#include"../utils/query.h"
#include <cassert>
 
// Use (void) to silence unused warnings.
#define assertm(exp, msg) std::assert(((void)msg, exp))



/**
 * Base quadtree-style Z-order index with optional skipping-aware traversal.
 */
class ZTree{
    public:
        ZtreeNode *root_;
        bool is_skipping_aware_{};
        uint32_t num_datapoints_,page_cnt_,node_cnt_,tree_height_;

        std::vector<ZtreeLeaflistMetadata> leaf_list_;
        BlockStore block_store_;


        uint32_t metric_points_scanned_{};
        uint32_t metric_mbrs_checked_{};

        /** Construct an empty ZTree shell. */
        ZTree(bool skipping_awareness = false){
            num_datapoints_ = 0;
            root_ = NULL;
            page_cnt_=0;
            node_cnt_ = 0;
            tree_height_ = 0;
            is_skipping_aware_ = skipping_awareness;
        }
        /** Descend the tree to the leaf that owns `new_point`. */
        /**
         * @brief Return pointer to the leaf node for `new_point`
         * NOTE: The ordering is not important now as the nodes are stored in default order.
         */
        ZtreeNode* LeafNodeForPoint(Point new_point){
            ZtreeNode* curr_node = root_;
            while(!(curr_node->is_leaf_)){
                bool bit_x = new_point.elements_[0]>=curr_node->partition_.elements_[0];
                bool bit_y = new_point.elements_[1]>=curr_node->partition_.elements_[1];
                int children_num=((bit_y<<1) + bit_x);
                curr_node = curr_node->children_[children_num];
            }
            return curr_node;
        }
        /** Bulk-load the leaf blocks after the tree structure is fixed. */
        /** TODO:
         * @brief Given a set of data points, this function iteratively partitions the points and inserts
         * them to respective pages.
         */
        void BulkLoadData(std::vector<Point>& dataset){
            num_datapoints_ = dataset.size();
            BulkLoadData(root_,dataset.begin(),dataset.end());
            block_store_.FinishedConstruction();
        }
        /** Recursive helper that bulk-loads one subtree into block pages. */
        /** TODO:
         * @brief Helper function to bulk load data.
         */
        void BulkLoadData(ZtreeNode *node,std::vector<Point>::iterator it_data_begin,std::vector<Point>::iterator it_data_end){

            if(node->is_leaf_){
                node->leaf_id_ = block_store_.InsertNewBlock(it_data_begin,it_data_end);
                return;
            }
            std::vector<Point>::iterator it_A_data_begin=it_data_begin,it_B_data_begin,it_C_data_begin,it_D_data_begin;
            std::vector<Point>::iterator it_A_data_end,it_B_data_end,it_C_data_end,it_D_data_end=it_data_end;

            ComparatorPointPartition x_partition_predicate(node->partition_,0);
            ComparatorPointPartition y_partition_predicate(node->partition_,1);
            
            it_B_data_begin = std::partition(it_data_begin,it_data_end,x_partition_predicate);
            it_C_data_end = it_B_data_begin;

            it_C_data_begin = std::partition(it_A_data_begin,it_C_data_end,y_partition_predicate);
            it_A_data_end = it_C_data_begin;

            it_D_data_begin = std::partition(it_B_data_begin,it_D_data_end,y_partition_predicate);
            it_B_data_end = it_D_data_begin;

            BulkLoadData(node->children_[0],it_A_data_begin,it_A_data_end);
            if(node->ordering_==false){
                BulkLoadData(node->children_[1],it_B_data_begin,it_B_data_end);
                BulkLoadData(node->children_[2],it_C_data_begin,it_C_data_end);
            }
            else{
                BulkLoadData(node->children_[2],it_C_data_begin,it_C_data_end);
                BulkLoadData(node->children_[1],it_B_data_begin,it_B_data_end);
            }
            BulkLoadData(node->children_[3],it_D_data_begin,it_D_data_end);

        }

        /** Save the current tree structure in its preorder text format. */
        void WriteTree(std::string file){
            std::ofstream fout(file);
            fout<<root_->mbr_.low_.elements_[0]<<" "<<root_->mbr_.low_.elements_[1]<<" "<<root_->mbr_.high_.elements_[0]<<" "<<root_->mbr_.high_.elements_[1];
            fout<<"\n0 "<<root_->partition_.elements_[0]<<" "<<root_->partition_.elements_[1]<<" "<<root_->ordering_;

            for(uint32_t i=0;i<4;i++)
                if(root_->children_[i]!=NULL)
                    WriteSubTree(fout,root_->children_[i],i+1);
        }
        /** Recursive helper for serializing one subtree. */
        /** DONE
         * @brief Helper recursive function for `WriteTree`
         */
        void WriteSubTree(std::ofstream& fout,ZtreeNode* node,uint64_t id){
            if(node->is_leaf_)
                return; 

            fout<<"\n"<<id<<" "<<node->partition_.elements_[0]<<" "<<node->partition_.elements_[1]<<" "<<node->ordering_;
            for(uint32_t i=0;i<4;i++)
                if(node->children_[i]!=NULL)
                    WriteSubTree(fout,node->children_[i],id*4+i+1);
        }

        /** Load a serialized tree structure from disk. */
        /**
         * @brief A function to load the Z-Index from a given file
         */
        void ReadTree(std::string file){
            root_= new ZtreeNode();
            node_cnt_++;

            std::ifstream fin(file);
            double_t lowx,lowy,hix,hiy;
            fin>>lowx>>lowy>>hix>>hiy;                          // first line consists of mbr of root.
            root_->mbr_=BoundingRectangle(Point(lowx,lowy),Point(hix,hiy));
            double_t midx,midy;
            uint64_t  id;
            uint32_t ord;
            
            while (!fin.eof())
            {
                fin>>id>>midx>>midy>>ord;                   // line consists of node-id(with missing nodes) and split+order
                InsertNodeToTree(id,midx,midy,ord);
            }
            UpdatePagesInSubtree(root_);
            SetLeafPageMaskCreateLocalModels(root_,is_skipping_aware_);

            if(is_skipping_aware_)
                CalculateFwdPointers();
        }

        /** Insert one serialized node entry into the in-memory tree. */
        /**
         * @brief Helper function to  `ReadTree` to insert each node
         */
        void InsertNodeToTree(uint64_t id,double_t midx,double_t midy,uint32_t ord){


            // store the path to insert location of new node.
            std::vector<uint32_t> path;                 
            while(id!=0){           
                path.push_back((id-1)%4);
                id=id/4-(id%4==0);
            }
            std::reverse(path.begin(), path.end());


            // find the node to be inserted. The node will already exist in tree.
            ZtreeNode* curr_node = root_;
            BoundingRectangle curr_region = root_->mbr_;
            for(uint32_t childId : path){
                switch (childId){
                    case 0:
                        curr_region.high_.elements_[0] = curr_node->partition_.elements_[0];
                        curr_region.high_.elements_[1] = curr_node->partition_.elements_[1];
                        break;
                    case 1:
                        curr_region.low_.elements_[0] = curr_node->partition_.elements_[0];
                        curr_region.high_.elements_[1] = curr_node->partition_.elements_[1];
                        break;    
                    case 2:
                        curr_region.high_.elements_[0] = curr_node->partition_.elements_[0];
                        curr_region.low_.elements_[1] = curr_node->partition_.elements_[1];
                        break;
                    case 3:
                        curr_region.low_.elements_[0] = curr_node->partition_.elements_[0];
                        curr_region.low_.elements_[1] = curr_node->partition_.elements_[1];
                        break;       
                    default:
                        break;
                }
                curr_node = curr_node->children_[childId];
            }


            // fill the node with split+order and create children.
            curr_node->partition_ = Point(midx,midy);
            curr_node->mbr_ = curr_region;
            curr_node->ordering_ = ord;
            curr_node->is_leaf_ = false;
            for(size_t i=0;i<4;i++){
                curr_node->children_[i]= new ZtreeNode(curr_node->node_depth_+1);
                curr_node->children_[i]->is_leaf_=true;
            }
            node_cnt_+=4;

            curr_node->children_[0]->mbr_ = BoundingRectangle(
                                Point(curr_node->mbr_.low_.elements_[0], curr_node->mbr_.low_.elements_[1]),
                                Point(curr_node->partition_.elements_[0], curr_node->partition_.elements_[1]));

            curr_node->children_[1]->mbr_ = BoundingRectangle(
                                Point(curr_node->partition_.elements_[0], curr_node->mbr_.low_.elements_[1]),
                                Point(curr_node->mbr_.high_.elements_[0], curr_node->partition_.elements_[1]));
        
            curr_node->children_[2]->mbr_ = BoundingRectangle(
                                Point(curr_node->mbr_.low_.elements_[0], curr_node->partition_.elements_[1]),
                                Point(curr_node->partition_.elements_[0], curr_node->mbr_.high_.elements_[1]));
        
            curr_node->children_[3]->mbr_ = BoundingRectangle(
                                Point(curr_node->partition_.elements_[0], curr_node->partition_.elements_[1]),
                                Point(curr_node->mbr_.high_.elements_[0], curr_node->mbr_.high_.elements_[1]));
        
        }
        /** Recompute the number of pages contained in each subtree. */
        void UpdatePagesInSubtree(ZtreeNode* node){
            if(node->is_leaf_){
                node->pages_in_subtree_=1;
                return;
            }

            UpdatePagesInSubtree(node->children_[0]);
            UpdatePagesInSubtree(node->children_[1]);
            UpdatePagesInSubtree(node->children_[2]);
            UpdatePagesInSubtree(node->children_[3]);

            node->pages_in_subtree_ = node->children_[0]->pages_in_subtree_+ 
                                    node->children_[1]->pages_in_subtree_+
                                    node->children_[2]->pages_in_subtree_+
                                    node->children_[3]->pages_in_subtree_;
        }

        /**/
        size_t SetLeafPageMaskCreateLocalModels(ZtreeNode* node, bool dummy_nodes_not_inserted_yet =false, uint64_t page_mask=0){

            /*If leaf then create */
            if(node->is_leaf_){
                page_cnt_++;
                leaf_list_.push_back(ZtreeLeaflistMetadata(BoundingRectangle(node->mbr_.low_,node->mbr_.high_)));
                node->leaf_id_ = leaf_list_.size()-1;
                leaf_list_[node->leaf_id_].InitializeFwdIters(node->leaf_id_+1);
                leaf_list_[node->leaf_id_].page_mask_ = page_mask+1ULL ;
                tree_height_ = std::max((node->node_depth_)+1,tree_height_);
                return node->leaf_id_;
            }


            // size_t inserted_dummy_node_id=0;
            // bool was_dummy_node_inserted=false;
            // if(node->pages_in_subtree_>256 && dummy_nodes_not_inserted_yet && is_skipping_aware_){
            //     dummy_nodes_not_inserted_yet=false;
            //     leaf_list_.push_back(ZtreeLeaflistMetadata(BoundingRectangle(node->mbr_.low_,node->mbr_.high_)));
            //     was_dummy_node_inserted =true;
            //     inserted_dummy_node_id = leaf_list_.size()-1;
            // }

            if(node->children_[0])
                node->leaf_id_ =  SetLeafPageMaskCreateLocalModels(node->children_[0],dummy_nodes_not_inserted_yet,page_mask|(0ULL<<(62-2*(node->node_depth_))));

            if(node->children_[1+(node->ordering_)])
                SetLeafPageMaskCreateLocalModels(node->children_[1+(node->ordering_)],dummy_nodes_not_inserted_yet,page_mask|(1ULL<<(62-2*(node->node_depth_))));

            if(node->children_[2-(node->ordering_)])
                SetLeafPageMaskCreateLocalModels(node->children_[2-(node->ordering_)],dummy_nodes_not_inserted_yet,page_mask|(2ULL<<(62-2*(node->node_depth_))));

            if(node->children_[3])
                SetLeafPageMaskCreateLocalModels(node->children_[3],dummy_nodes_not_inserted_yet,page_mask|(3ULL<<(62-2*(node->node_depth_))));
            

            // if(was_dummy_node_inserted)
            //     leaf_list_[inserted_dummy_node_id].page_mask_ = leaf_list_[node->leaf_id_].page_mask_-1;

            return node->leaf_id_;

        }


        void CalculateFwdPointers(){
            assert(is_skipping_aware_ && "Trying to calculate FwdPointer in skipping agnostic setting. The fwd_iters are not initialized in pages.");

            // iteratively calculate the forward pointers for each page starting from the last page.
            for(size_t curr_page_id= leaf_list_.size()-1;curr_page_id>0;curr_page_id--)
                CalculateFwdPointersHelper(curr_page_id);
        }


        /**
         * @brief Function that takes a page iterator and builds the four fwd pointers required 
         */
        void CalculateFwdPointersHelper(size_t curr_page_id){
            leaf_list_[curr_page_id].InitializeFwdIters(curr_page_id+1);

            /** @brief While the page pointed to by prev_page's fwd_iter_1_ has high_.x_ less than or equal 
             *       to prev_page's high_.x_; */
            while(leaf_list_[curr_page_id].fwd_iter_1_ < leaf_list_.size() && 
                    leaf_list_[leaf_list_[curr_page_id].fwd_iter_1_].mbr_.high_.elements_[0] <= leaf_list_[curr_page_id].mbr_.high_.elements_[0]){
                
                leaf_list_[curr_page_id].fwd_iter_1_ = leaf_list_[leaf_list_[curr_page_id].fwd_iter_1_].fwd_iter_1_;
            }

            // std::cout<<"\t SamplZTree::CalculateFwdPointers::CalculateFwdPointersHelper;   curr_page_id [fwd_iter_1_]:"<<leaf_list_[curr_page_id].fwd_iter_1_<<"\n";

            while(leaf_list_[curr_page_id].fwd_iter_2_ < leaf_list_.size() &&
                    leaf_list_[leaf_list_[curr_page_id].fwd_iter_2_].mbr_.high_.elements_[1] <= leaf_list_[curr_page_id].mbr_.high_.elements_[1]){                
                leaf_list_[curr_page_id].fwd_iter_2_ = leaf_list_[leaf_list_[curr_page_id].fwd_iter_2_].fwd_iter_2_;
            }
            // std::cout<<"\t SamplZTree::CalculateFwdPointers::CalculateFwdPointersHelper;   curr_page_id [fwd_iter_2_]:"<<leaf_list_[curr_page_id].fwd_iter_2_<<"\n";


            /** @brief While he page pointed to by prev_page's fwd_iter_1_ has low_.x_ higher than or equal 
             *       to prev_page's low_.x_;
             */
            while(leaf_list_[curr_page_id].fwd_iter_3_ < leaf_list_.size() && 
                leaf_list_[leaf_list_[curr_page_id].fwd_iter_3_].mbr_.low_.elements_[0] >= leaf_list_[curr_page_id].mbr_.low_.elements_[0]){                
                leaf_list_[curr_page_id].fwd_iter_3_ = leaf_list_[leaf_list_[curr_page_id].fwd_iter_3_].fwd_iter_3_;
            }
            // std::cout<<"\t SamplZTree::CalculateFwdPointers::CalculateFwdPointersHelper;   curr_page_id [fwd_iter_3_]:"<<leaf_list_[curr_page_id].fwd_iter_3_<<"\n";


            while(leaf_list_[curr_page_id].fwd_iter_4_ < leaf_list_.size() && 
                leaf_list_[leaf_list_[curr_page_id].fwd_iter_4_].mbr_.low_.elements_[1] >= leaf_list_[curr_page_id].mbr_.low_.elements_[1]){                
                leaf_list_[curr_page_id].fwd_iter_4_ = leaf_list_[leaf_list_[curr_page_id].fwd_iter_4_].fwd_iter_4_;
            }
            // std::cout<<"\t SamplZTree::CalculateFwdPointers::CalculateFwdPointersHelper;   curr_page_id [fwd_iter_4_]:"<<leaf_list_[curr_page_id].fwd_iter_4_<<"\n";

        }


        


        /**
         * @brief A function to perform a range query and return the query results along 
         *        with stats of range query processing.
         */
        std::vector<Point> RangeQuery(Query & query){

            std::vector<size_t> refined_blocks;
            Projection(refined_blocks,query);

            // std::cout<<"Num refined blocks: "<<refined_blocks.size()<<"\n";
            std::vector<Point> result_vec;
            Scan(refined_blocks,query,result_vec);
            return result_vec;

            return result_vec;
        }

        void Projection(std::vector<size_t> &projected_cells,Query & query){
            if(is_skipping_aware_)
                ProjectionWithSkipping(projected_cells,query);
            else
                ProjectionBasic(projected_cells,query);
        }

        void ProjectionBasic(std::vector<size_t> &projected_cells,Query & query){
            
            size_t start_iter = (LeafNodeForPoint(query.low_))->leaf_id_;
            size_t end_iter = (LeafNodeForPoint(query.high_))->leaf_id_;
            end_iter++;


            for(size_t it=start_iter;it<=end_iter;it++,metric_mbrs_checked_++)
                if(query.IsThereOverlap(leaf_list_[it].mbr_))
                    projected_cells.push_back(it);
            
        }

        void ProjectionWithSkipping(std::vector<size_t> &projected_cells,Query & query){
            std::vector<Point> queryResult;
            size_t start_iter = (LeafNodeForPoint(query.low_))->leaf_id_;
            size_t end_iter = (LeafNodeForPoint(query.high_))->leaf_id_;


            size_t it=start_iter;
            size_t next_iter,temp_iter;
            uint64_t furthest_mask,next_possible_mask;
            uint8_t mask;
            while(it<leaf_list_.size()){  
                metric_mbrs_checked_++;  
                mask = leaf_list_[it].CheckFwdIterCases(query);
                
                if(mask){ 
                    //IF mask non zero then 
                    next_iter = it+1;
                    furthest_mask = leaf_list_[it].page_mask_;
                    if(mask & 1){
                        temp_iter = leaf_list_[it].fwd_iter_1_;
                        while(temp_iter < leaf_list_.size() && leaf_list_[temp_iter].mbr_.high_.elements_[0] < query.low_.elements_[0]) 
                            temp_iter=leaf_list_[temp_iter].fwd_iter_1_; 

                        if(temp_iter == leaf_list_.size()) break;
                        if(furthest_mask < leaf_list_[temp_iter].page_mask_){
                            furthest_mask = leaf_list_[temp_iter].page_mask_;
                            next_iter = temp_iter;
                        }
                    }

                    else if(mask & 4){
                        temp_iter = leaf_list_[it].fwd_iter_3_;
                        while(temp_iter < leaf_list_.size() && leaf_list_[temp_iter].mbr_.low_.elements_[0] > query.high_.elements_[0]) 
                            temp_iter=leaf_list_[temp_iter].fwd_iter_3_; 
            
                        if(temp_iter == leaf_list_.size()) break;
                        if(furthest_mask < leaf_list_[temp_iter].page_mask_){
                            furthest_mask = leaf_list_[temp_iter].page_mask_;
                            next_iter = temp_iter;
                        }
                    }


                    if(mask & 2){
                        temp_iter = leaf_list_[it].fwd_iter_2_;
                        while(temp_iter < leaf_list_.size() && leaf_list_[temp_iter].mbr_.high_.elements_[1] < query.low_.elements_[1]) 
                            temp_iter=leaf_list_[temp_iter].fwd_iter_2_;
            
                        if(temp_iter == leaf_list_.size()) break;
                        if(furthest_mask< leaf_list_[temp_iter].page_mask_){
                            furthest_mask =  leaf_list_[temp_iter].page_mask_;
                            next_iter = temp_iter;
                        }
                    }

                    else if(mask & 8){
                        temp_iter = leaf_list_[it].fwd_iter_4_;
                        while(temp_iter < leaf_list_.size() && leaf_list_[temp_iter].mbr_.low_.elements_[1] > query.high_.elements_[1]) 
                            temp_iter=leaf_list_[temp_iter].fwd_iter_4_;
            
                        if(temp_iter == leaf_list_.size()) break;
                        if(furthest_mask< leaf_list_[temp_iter].page_mask_){
                            furthest_mask =  leaf_list_[temp_iter].page_mask_;
                            next_iter = temp_iter;
                        }
                    }

                    it = next_iter;
                }

                else{
                    projected_cells.push_back(it);
                    it++;
                }

                if(leaf_list_[it].page_mask_ > leaf_list_[end_iter].page_mask_)
                    break;

            }
        }

        void Scan(std::vector<size_t> &projected_cell_ids, Query& query, std::vector<Point> &result_vec){
            std::sort(projected_cell_ids.begin(),projected_cell_ids.end());
            block_store_.FilterPointsFromBlocksForQuery(query, projected_cell_ids, result_vec);
        }


        /**
         * @brief Function to traverse down the tree and fetch the page num for a given point.
         */
        uint64_t GetPageMask(ZtreeNode* node,Point point){
            if(node->is_leaf_)
                return leaf_list_[node->leaf_id_].page_mask_;
        
            bool bit_x = point.elements_[0]>=node->partition_.elements_[0],bit_y = point.elements_[1]>=node->partition_.elements_[1];
            int children_num=((bit_y<<1) + bit_x);             // We don't need to account for ordering because SetLeafNum does the ordering.
            return GetPageMask(node->children_[children_num],point);
        }


        size_t ModelSize(){
            return sizeof(ZtreeNode)*node_cnt_ + sizeof(ZTree) + page_cnt_*sizeof(ZtreeLeaflistMetadata);
        }

        


  

        // ********************** Unused functions ******************

        // /**
        //  *  
        //  * @brief Function to print all pages of within my array
        //  * 
        //  */
        // void PrintPages(){
        //     std::cout<<"Inside Print Pages\n";
        //     std::cout<<"page_cnt_  "<<page_cnt_<<"("<<leaf_list_.size()<<") tree_height_  "<<tree_height_<<" \n";
        //     int curr_page_num =0;
        //     for(auto& pg: leaf_list_){
        //         std::cout<<"\n No:"<<curr_page_num++<<"\t mask:"<<pg.page_mask_<<"\t("<<pg.mbr_.low_.elements_[0]<<','<<pg.mbr_.low_.elements_[1]<<") ("<<pg.mbr_.high_.elements_[0]<<','<<pg.mbr_.high_.elements_[1]<<") [size="<<pg.local_data_.size()<<"]:  [ "<<pg.fwd_iter_1_<<", "<<pg.fwd_iter_2_<<", "<<pg.fwd_iter_3_<<", "<<pg.fwd_iter_4_<<"]";

        //         if (pg.fwd_iter_1_ == leaf_list_.size())
        //             std::cout<<"\n\t\tLEFT  PageList End()";
        //         else
        //             std::cout<<"\n\t\tLEFT  mask:"<<leaf_list_[pg.fwd_iter_1_].page_mask_<<"\t("<<leaf_list_[pg.fwd_iter_1_].mbr_.low_.elements_[0]<<','<<leaf_list_[pg.fwd_iter_1_].mbr_.low_.elements_[1]<<") ("<<leaf_list_[pg.fwd_iter_1_].mbr_.high_.elements_[0]<<','<<leaf_list_[pg.fwd_iter_1_].mbr_.high_.elements_[1]<<") [size="<<leaf_list_[pg.fwd_iter_1_].local_data_.size()<<"]: ";

        //         if (pg.fwd_iter_2_ == leaf_list_.size())
        //             std::cout<<"\n\t\tBELOW PageList End()";
        //         else
        //             std::cout<<"\n\t\tBELOW mask:"<<leaf_list_[pg.fwd_iter_2_].page_mask_<<"\t("<<leaf_list_[pg.fwd_iter_2_].mbr_.low_.elements_[0]<<','<<leaf_list_[pg.fwd_iter_2_].mbr_.low_.elements_[1]<<") ("<<leaf_list_[pg.fwd_iter_2_].mbr_.high_.elements_[0]<<','<<leaf_list_[pg.fwd_iter_2_].mbr_.high_.elements_[1]<<") [size="<<leaf_list_[pg.fwd_iter_2_].local_data_.size()<<"]: ";

        //         if (pg.fwd_iter_3_ == leaf_list_.size())
        //             std::cout<<"\n\t\tRIGHT PageList End()";
        //         else
        //             std::cout<<"\n\t\tRIGHT mask:"<<leaf_list_[pg.fwd_iter_3_].page_mask_<<"\t("<<leaf_list_[pg.fwd_iter_3_].mbr_.low_.elements_[0]<<','<<leaf_list_[pg.fwd_iter_3_].mbr_.low_.elements_[1]<<") ("<<leaf_list_[pg.fwd_iter_3_].mbr_.high_.elements_[0]<<','<<leaf_list_[pg.fwd_iter_3_].mbr_.high_.elements_[1]<<") [size="<<leaf_list_[pg.fwd_iter_3_].local_data_.size()<<"]: ";

        //         if (pg.fwd_iter_4_ == leaf_list_.size())
        //             std::cout<<"\n\t\tABOVE PageList End()";
        //         else
        //             std::cout<<"\n\t\tABOVE mask:"<<leaf_list_[pg.fwd_iter_4_].page_mask_<<"\t("<<leaf_list_[pg.fwd_iter_4_].mbr_.low_.elements_[0]<<','<<leaf_list_[pg.fwd_iter_4_].mbr_.low_.elements_[1]<<") ("<<leaf_list_[pg.fwd_iter_4_].mbr_.high_.elements_[0]<<','<<leaf_list_[pg.fwd_iter_4_].mbr_.high_.elements_[1]<<") [size="<<leaf_list_[pg.fwd_iter_4_].local_data_.size()<<"]: ";
        //     }
        // }

};
#endif
