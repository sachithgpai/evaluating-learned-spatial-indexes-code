

#ifndef RTREE_BASE_H
#define RTREE_BASE_H

#include"rtree_node.h"
#include <iomanip>
#define MINFILL (BLOCK_SIZE>>2)

/*
An Rtree base class where common Rtree functions like Range query, Write-tree and read-tree are implemented.
*/
class RTreeBASE{
    public:
        RTreeNode* root_;
        BlockStore block_store_;
        size_t node_cnt_{};

        RTreeBASE(){}
        
        RTreeBASE(string filename){
            root_ = new RTreeNode();
            std::ifstream fin(filename);
            ReadTree(fin,root_);
            fin.close();

            block_store_.FinishedConstruction();
        }

        ~RTreeBASE(){
            delete root_;
        }

        void WriteTree(std::ofstream& fout,RTreeNode* node){
            fout<<std::setprecision(9)<<node->children_.size()<<" "<<node->mbr_.low_.elements_[0]<<" "<<node->mbr_.low_.elements_[1]<<" "<<node->mbr_.high_.elements_[0]<<" "<<node->mbr_.high_.elements_[1]<<"\n";

            if(node->is_leaf_){
                fout<<block_store_.block_point_counts_[node->local_block_id_]<<"\n";
                for(auto pnt: block_store_.FetchPointsInBlock(node->local_block_id_))
                    fout<<std::setprecision(9)<<pnt.elements_[0]<<" "<<pnt.elements_[1]<<"\n";
            }
            else
                for(auto child: node->children_)
                    WriteTree(fout,child);
        }


        
        void ReadTree(std::ifstream& fin,RTreeNode* node){
            size_t num_children, block_size;
            double_t x,y;
            fin>>num_children>>node->mbr_.low_.elements_[0]>>node->mbr_.low_.elements_[1]>>node->mbr_.high_.elements_[0]>>node->mbr_.high_.elements_[1];

            if(num_children==0){// same as if(node->is_leaf_) TODO:
                
                node->local_block_id_ = block_store_.CreateEmptyBlock();
                fin>>block_size;
                for(size_t i=0;i<block_size;i++){
                    fin>>x>>y;
                    Point pnt(x,y);
                    block_store_.InsertNewPointInBlock(pnt,node->local_block_id_);
                }
            }
            else{
                node->is_leaf_=false;
                for(size_t i=0;i<num_children;i++){
                    RTreeNode* new_child = new RTreeNode();
                    ReadTree(fin,new_child);
                    node->children_.push_back(new_child);
                }
            }
            
        }



        std::vector<Point> RangeQuery(Query& query){
            std::vector<size_t> refined_blocks;
            Projection(refined_blocks,query,root_);

            std::vector<Point> result_vec;
            Scan(refined_blocks,query,result_vec);
            return result_vec;
        }

        void Projection(std::vector<size_t> &refined_blocks, Query& query, RTreeNode* node){
            if(node->is_leaf_){
                refined_blocks.push_back(node->local_block_id_);
                return;
            }

            for(auto &child: node->children_)
                if(query.IsThereOverlap(child->mbr_))
                    Projection(refined_blocks,query,child);
        }


        void Scan(std::vector<size_t> &refined_blocks, Query& query, std::vector<Point> &result_vec){
            std::sort(refined_blocks.begin(),refined_blocks.end());
            block_store_.FilterPointsFromBlocksForQuery(query,refined_blocks,result_vec);
        }


        /* PRINTING AND DEBUG FUNCTIONS*/
        void PrintTree(RTreeNode* node,string deb_string="|-> "){
            
            std::cout<<deb_string<<((node->is_leaf_)?" LEAF ":"")<<((node->is_leaf_)?(node->local_block_id_):0); 
            node->mbr_.Print();
            if(node->is_leaf_)
                return;
            for(auto child: node->children_)
                PrintTree(child,deb_string+"|-> ");
        }
};

#endif