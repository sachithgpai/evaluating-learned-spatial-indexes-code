#ifndef RWTREE_H
#define RWTREE_H


#include<vector>
#include<utility>
#include<tuple>
#include<iterator>
#include<string>
#include<array>
#include<random>

#include"../utils/local_model.h"
#include"../utils/query.h"
#include"rtree_node.h"
#include"rtree_base.h"
#include"../utils/sort_tools.h"
#include"../utils/density_estimators/weighted_dens_est.h"


#define RW_MINBRANCH (RW_BRANCH_FACTOR>>2)


/**
 * Query-weighted incremental R-tree variant that relies on a weighted density
 * estimator during split and insertion decisions.
 */
class RWTree: public RTreeBASE{
    public:
        WeightedDensEstTree* weighted_datapoint_density_estimator_{};

        /** Create an empty RWTree shell. */
        RWTree(){}

        /** Reconstruct an RWTree from the serialized format. */
        RWTree(string filename){
            root_ = new RTreeNode();
            std::ifstream fin(filename);
            ReadTree(fin,root_);
            fin.close();
            block_store_.FinishedConstruction();
        }

        /** Build an RWTree from data and the training query workload. */
        RWTree(std::vector<Point> data, std::vector<Query> queries){


            BoundingRectangle mbr;
            for(auto &pnt:data) mbr.UpdateBoundingBoxWithPoint(pnt);

            std::random_device rd;  // Will be used to obtain a seed for the random number engine
            std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
            std::uniform_real_distribution<> uni_dist(0.0, 1.0);
            std::vector<Point> uniform_data;

            for(int i=0;i<data.size();i++) uniform_data.emplace_back(uni_dist(gen),uni_dist(gen));
            std::vector<WrappedPoint> wrapped_uniform_data = WeightPointsWithQuery(uniform_data,queries);

            double_t total_weight_from_queries=0;
            for(auto &uni_pnt:wrapped_uniform_data) total_weight_from_queries+= uni_pnt.num_queries_overlapping_;
            std::cout<<"RWTree::WeightPointsWithQuery::total_weight_from_queries:"<<total_weight_from_queries<<"\n";


            weighted_datapoint_density_estimator_ = new WeightedDensEstTree(wrapped_uniform_data,queries,total_weight_from_queries/10,mbr);
            std::cout<<"RWTree::WeightPointsWithQuery:: FINISHED CONSTRUCTING WeightedDensEstTree"<<"\n";

            BuildRWTree(data);
            block_store_.FinishedConstruction();
        }

        /** Release the owned weighted density estimator. */
        ~RWTree(){
            if(weighted_datapoint_density_estimator_ != NULL )
                delete weighted_datapoint_density_estimator_;
        }
        /** Bootstrap the incremental RWTree construction from the initial block. */
        void BuildRWTree(std::vector<Point>& data){

            root_ = new RTreeNode();
            node_cnt_++;

            // create initial child node and fill it with BLOCK SIZE elements
            RTreeNode* initial_child = new RTreeNode();
            initial_child->local_block_id_ = block_store_.InsertNewBlock(data.begin(),data.begin()+BLOCK_SIZE);
            initial_child->mbr_=block_store_.FetchBoundingBoxForBlock(initial_child->local_block_id_);

            // Insert the initial_child into children of root. Now tree has root and one leaf.
            root_->is_leaf_=false;
            root_->children_.push_back(initial_child);
            root_->mbr_.UpdateBoundingBoxWithBoundingBox(initial_child->mbr_);
         
            // Insert rest of the points iteratively.
            for(size_t it=BLOCK_SIZE;it<data.size();it++)
                InsertPoint(data[it]);

            return;
        }

        /** Insert one point and maintain the incremental tree invariants. */
        void InsertPoint(Point pnt){

            // ChooseSubtree to find the leaf node to insert. Also get all the nodes that are accessed in the way.
            std::vector<RTreeNode*> parents;
            RTreeNode* leaf_node_to_insert_point = ChooseSubtree(root_,pnt,parents);
            std::reverse(parents.begin(),parents.end()); // make the parents array bottom up.

            // insert pnt into leaf_node_to_insert_point. IF it overflows, split it into two. 
            size_t &block_id = leaf_node_to_insert_point->local_block_id_,new_block_id;
            if(block_store_.InsertNewPointInBlock(pnt,block_id)>BLOCK_SIZE){
                // Take all points from given block and put back half of the points. 
                std::vector<Point> temp_arr = block_store_.FetchPointsInBlock(block_id);
                size_t split_pos = SplitPointsIntoTwo(temp_arr);
                leaf_node_to_insert_point->mbr_ = block_store_.ReassignPointsInBlock(block_id,temp_arr.begin(),temp_arr.begin()+split_pos);
                

                // Create a new leaf node and place back.
                RTreeNode* new_node = new RTreeNode();
                new_node->local_block_id_ = block_store_.InsertNewBlock(temp_arr.begin()+split_pos,temp_arr.end());
                new_node->mbr_=block_store_.FetchBoundingBoxForBlock(new_node->local_block_id_);
                parents[0]->children_.push_back(new_node);
                parents[0]->mbr_.UpdateBoundingBoxWithBoundingBox(new_node->mbr_);

            }
            else{
                leaf_node_to_insert_point->mbr_.UpdateBoundingBoxWithPoint(pnt);
                parents[0]->mbr_.UpdateBoundingBoxWithBoundingBox(leaf_node_to_insert_point->mbr_);
            }

            
            // split the internal nodes bottom up.
            size_t parent_to_split =0;
            while(parent_to_split<parents.size()){
                
                //split child nodes if it needs to be split
                if(parents[parent_to_split]->children_.size() > RW_BRANCH_FACTOR){

                    size_t split_location = SplitNodesIntoTwo(parents[parent_to_split]->children_);

                    // create a new node and insert half the nodes into it
                    RTreeNode* new_node = new RTreeNode();
                    new_node->children_.assign(parents[parent_to_split]->children_.begin()+split_location,parents[parent_to_split]->children_.end());
                    for(auto& child: new_node->children_)
                        new_node->mbr_.UpdateBoundingBoxWithBoundingBox(child->mbr_);
                    new_node->is_leaf_=false;

                    // update the current node by removing the secoond half of nodes and updating the bounding box.
                    parents[parent_to_split]->mbr_.SetToDefault();
                    parents[parent_to_split]->children_.resize(split_location);
                    for(auto& child: parents[parent_to_split]->children_)
                        parents[parent_to_split]->mbr_.UpdateBoundingBoxWithBoundingBox(child->mbr_);


                    if(parent_to_split==parents.size()-1){              // if the rood node is overflowing create a new root node.
                        root_ = new RTreeNode();
                        root_->is_leaf_=false;
                        root_->children_.push_back(parents[parent_to_split]);
                        root_->mbr_.UpdateBoundingBoxWithBoundingBox(parents[parent_to_split]->mbr_);
                        root_->mbr_.UpdateBoundingBoxWithBoundingBox(new_node->mbr_);
                        root_->children_.push_back(new_node);

                    }
                    else{
                        parents[parent_to_split+1]->mbr_.UpdateBoundingBoxWithBoundingBox(parents[parent_to_split]->mbr_);
                        parents[parent_to_split+1]->mbr_.UpdateBoundingBoxWithBoundingBox(new_node->mbr_);
                        parents[parent_to_split+1]->children_.push_back(new_node);
                    }
                }
                else if(parent_to_split!=0)
                    parents[parent_to_split]->mbr_.UpdateBoundingBoxWithBoundingBox(parents[parent_to_split-1]->mbr_);

                parent_to_split++;
            }

        }

        /** Split an overflowing internal node and return the split position. */
        size_t SplitNodesIntoTwo(std::vector<RTreeNode*>& temp_arr){

            double_t min_split_cost = std::numeric_limits<double_t>::max();
            size_t argmin_split_cost, sort_dim, split_using_low_or_high;
            std::vector<double_t> perimeters;            
            BoundingRectangle temp_mbr;



            for(int ord=0;ord<2;ord++){

                // Comparing against low
                auto sort_nodes_low = [ord](const RTreeNode* n1,const RTreeNode* n2) { return n1->mbr_.low_.elements_[ord]<n2->mbr_.low_.elements_[ord];};
                perimeters.clear();
                perimeters.resize(temp_arr.size()-2*RW_MINBRANCH+1,0.0);
                temp_mbr.SetToDefault();
                std::sort(temp_arr.begin(), temp_arr.end(), sort_nodes_low);
                
                for(int i=0;i<RW_MINBRANCH-1;i++)
                    temp_mbr.UpdateBoundingBoxWithBoundingBox(temp_arr[i]->mbr_);

                for(int i=RW_MINBRANCH-1;i<temp_arr.size()-RW_MINBRANCH;i++){
                    temp_mbr.UpdateBoundingBoxWithBoundingBox(temp_arr[i]->mbr_);
                    perimeters[i-(RW_MINBRANCH-1)]+=weighted_datapoint_density_estimator_->EstimateCount(temp_mbr);
                }

                temp_mbr.SetToDefault();

                for(int i=temp_arr.size()-1;i>temp_arr.size()-RW_MINBRANCH;i--)
                    temp_mbr.UpdateBoundingBoxWithBoundingBox(temp_arr[i]->mbr_);  

                for(int i=temp_arr.size()-RW_MINBRANCH;i>=RW_MINBRANCH;i--){
                    temp_mbr.UpdateBoundingBoxWithBoundingBox(temp_arr[i]->mbr_);
                    perimeters[i-RW_MINBRANCH]+=weighted_datapoint_density_estimator_->EstimateCount(temp_mbr);
                    if(perimeters[i-RW_MINBRANCH]<min_split_cost){
                        sort_dim=ord;
                        argmin_split_cost=i;
                        min_split_cost=perimeters[i-RW_MINBRANCH];
                        split_using_low_or_high=0;
                    }
                }
                

                auto sort_nodes_high = [ord](const RTreeNode* n1,const RTreeNode* n2) { return n1->mbr_.high_.elements_[ord]<n2->mbr_.high_.elements_[ord];};
                perimeters.clear();
                perimeters.resize(temp_arr.size()-2*RW_MINBRANCH+1,0.0);
                std::sort(temp_arr.begin(), temp_arr.end(), sort_nodes_high);
                
                for(int i=0;i<RW_MINBRANCH-1;i++)
                    temp_mbr.UpdateBoundingBoxWithBoundingBox(temp_arr[i]->mbr_);

                for(int i=RW_MINBRANCH-1;i<temp_arr.size()-RW_MINBRANCH;i++){
                    temp_mbr.UpdateBoundingBoxWithBoundingBox(temp_arr[i]->mbr_);
                    perimeters[i-(RW_MINBRANCH-1)]+=weighted_datapoint_density_estimator_->EstimateCount(temp_mbr);
                }

                temp_mbr.SetToDefault();

                for(int i=temp_arr.size()-1;i>temp_arr.size()-RW_MINBRANCH;i--)
                    temp_mbr.UpdateBoundingBoxWithBoundingBox(temp_arr[i]->mbr_);  

                for(int i=temp_arr.size()-RW_MINBRANCH;i>=RW_MINBRANCH;i--){
                    temp_mbr.UpdateBoundingBoxWithBoundingBox(temp_arr[i]->mbr_);
                    perimeters[i-RW_MINBRANCH]+=weighted_datapoint_density_estimator_->EstimateCount(temp_mbr);
                    if(perimeters[i-RW_MINBRANCH]<min_split_cost){
                        sort_dim=ord;
                        argmin_split_cost=i;
                        min_split_cost=perimeters[i-RW_MINBRANCH];
                        split_using_low_or_high=1;
                    }
                }
                
                
            }



            if(split_using_low_or_high){
                auto sorter = [sort_dim](const RTreeNode* n1,const RTreeNode* n2) { return n1->mbr_.high_.elements_[sort_dim]<n2->mbr_.high_.elements_[sort_dim];};
                std::sort(temp_arr.begin(), temp_arr.end(), sorter);
            }

            else{
                auto sorter = [sort_dim](const RTreeNode* n1,const RTreeNode* n2) { return n1->mbr_.low_.elements_[sort_dim]<n2->mbr_.low_.elements_[sort_dim];};
                std::sort(temp_arr.begin(), temp_arr.end(), sorter);
            }
                
            return argmin_split_cost;


        }


        /* Function splits the points into two. The function sorts the array according to best sort dim and returns the location of split*/
        size_t SplitPointsIntoTwo(std::vector<Point>& temp_arr){

            double_t min_split_cost = std::numeric_limits<double_t>::max();
            size_t argmin_split_cost, sort_dim;
            std::vector<double_t> costs_at_split;
            costs_at_split.resize(temp_arr.size()-2*MINFILL+1,0.0);
            BoundingRectangle temp_mbr;



            std::sort(temp_arr.begin(), temp_arr.end(), SortOrderer(SortX));
            temp_mbr.SetToDefault();

            for(int i=0;i<MINFILL-1;i++)
                temp_mbr.UpdateBoundingBoxWithPoint(temp_arr[i]);

            for(int i=MINFILL-1;i<temp_arr.size()-MINFILL;i++){
                temp_mbr.UpdateBoundingBoxWithPoint(temp_arr[i]);
                costs_at_split[i-(MINFILL-1)]+=weighted_datapoint_density_estimator_->EstimateCount(temp_mbr);
            }


            temp_mbr.SetToDefault();
            for(int i=temp_arr.size()-1;i>temp_arr.size()-MINFILL;i--)
                temp_mbr.UpdateBoundingBoxWithPoint(temp_arr[i]);

            for(int i=temp_arr.size()-MINFILL;i>=MINFILL;i--){
                temp_mbr.UpdateBoundingBoxWithPoint(temp_arr[i]);
                costs_at_split[i-MINFILL]+=weighted_datapoint_density_estimator_->EstimateCount(temp_mbr);
                if(costs_at_split[i-MINFILL]<min_split_cost){
                    sort_dim=SortX;
                    argmin_split_cost=i;
                    min_split_cost=costs_at_split[i-MINFILL];
                }
            }

            costs_at_split.clear();
            costs_at_split.resize(temp_arr.size()-2*MINFILL+1,0.0);
            std::sort(temp_arr.begin(), temp_arr.end(), SortOrderer(SortY));
            temp_mbr.SetToDefault();

            for(int i=0;i<MINFILL-1;i++)
                temp_mbr.UpdateBoundingBoxWithPoint(temp_arr[i]);

            for(int i=MINFILL-1;i<temp_arr.size()-MINFILL;i++){
                temp_mbr.UpdateBoundingBoxWithPoint(temp_arr[i]);
                costs_at_split[i-(MINFILL-1)]+=weighted_datapoint_density_estimator_->EstimateCount(temp_mbr);
            }


            temp_mbr.SetToDefault();
            for(int i=temp_arr.size()-1;i>temp_arr.size()-MINFILL;i--)
                temp_mbr.UpdateBoundingBoxWithPoint(temp_arr[i]); 

            for(int i=temp_arr.size()-MINFILL;i>=MINFILL;i--){
                temp_mbr.UpdateBoundingBoxWithPoint(temp_arr[i]);
                costs_at_split[i-MINFILL]+=weighted_datapoint_density_estimator_->EstimateCount(temp_mbr);
                if(costs_at_split[i-MINFILL]<min_split_cost){
                    sort_dim=SortY;
                    argmin_split_cost=i;
                    min_split_cost=costs_at_split[i-MINFILL];
                }
            }


            std::sort(temp_arr.begin(), temp_arr.end(), SortOrderer(sort_dim));
            return argmin_split_cost;
        }



        /* A function to find exact leaf node within which the new point should be inserted. The function appends nodes traversed on the path*/
        RTreeNode* ChooseSubtree(RTreeNode* node, Point& insert_pnt,std::vector<RTreeNode*>& parents){
            if(node->is_leaf_)
                return node;

            parents.push_back(node);

            //For all children compute <if point falls within node>,<area of node>, <delta area of node>


            // a structure to hold stats required to perform choose subtree.
            std::vector<NodeStatsPointInsert> children_stats;       

            // for each child compute delta volume and volume and cost and delta cost
            BoundingRectangle expanded_mbr;
            size_t child_id=0;
            for(auto& child_ptr: node->children_){
                expanded_mbr = child_ptr->mbr_; 
                expanded_mbr.UpdateBoundingBoxWithPoint(insert_pnt);
                children_stats.emplace_back(child_id,child_ptr->mbr_,insert_pnt,weighted_datapoint_density_estimator_);
                child_id++;
            }

            //sort by delta cost and then by cost
            auto nspi_comp = [](const NodeStatsPointInsert& ns1,const NodeStatsPointInsert& ns2) { return std::tie(ns1.delta_cost_, ns1.cost_) < std::tie(ns2.delta_cost_, ns2.cost_);};
            std::sort(children_stats.begin(),children_stats.end(),nspi_comp);

            //IF you find nodes that covers the point to be inserted then pick the one with smallest volume (already achieved by nspi_comp)
            if(children_stats[0].delta_cost_ < Constants::EPSILON_ERR || node->children_[0]->is_leaf_==false)
                return ChooseSubtree(node->children_[children_stats[0].child_pointer_id_],insert_pnt,parents);

            //ELSE find a 
            double_t argmin_delta_overlap_volume = std::numeric_limits<double_t>::max();
            size_t argmin_child_id = 0;
            for(auto& ns:children_stats){
                for(size_t cid=0;cid<node->children_.size();cid++)
                    if(ns.child_pointer_id_!=cid)
                        ns.delta_overlap_volume_+= ns.expanded_mbr_.AreaOfOverlap(node->children_[cid]->mbr_);
            
                if (ns.delta_overlap_volume_ < argmin_delta_overlap_volume){
                    argmin_delta_overlap_volume = ns.delta_overlap_volume_;
                    argmin_child_id = ns.child_pointer_id_;
                }
            }

            return ChooseSubtree(node->children_[argmin_child_id],insert_pnt,parents);
        }
        
};

#endif
