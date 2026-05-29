#ifndef RWTREE_H
#define RWTREE_H


#include<vector>
#include<utility>
#include<tuple>
#include<iterator>
#include<string>
#include<array>
#include<algorithm>
#include<cstdlib>
#include<limits>

#include"../utils/local_model.h"
#include"../utils/query.h"
#include"../utils/density_estimators/query_dens_est.h"
#include"rtree_node.h"
#include"rtree_base.h"
#include"../utils/sort_tools.h"


#define RW_MINBRANCH (RW_BRANCH_FACTOR>>2)


/**
 * Query-weighted incremental R-tree variant that estimates training-workload
 * page-scan cost during split and insertion decisions.
 */
class RWTree: public RTreeBASE{
    public:
        QueryDensEstTree* query_scan_cost_estimator_{NULL};
        size_t query_scan_cost_estimator_granularity_{4};
        size_t choose_subtree_query_top_k_{4};

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

            ConfigureBuildParameters();
            BuildQueryScanCostEstimator(std::move(queries));
            BuildRWTree(data);
            block_store_.FinishedConstruction();
        }

        /** Release the query-density estimator owned by this RWTree. */
        ~RWTree(){
            delete query_scan_cost_estimator_;
        }

        /** Build the 4D query-endpoint density estimator used by the RW cost oracle. */
        void BuildQueryScanCostEstimator(std::vector<Query> queries){
            delete query_scan_cost_estimator_;
            query_scan_cost_estimator_ = new QueryDensEstTree(std::move(queries), query_scan_cost_estimator_granularity_);
        }

        /** Estimate page-scan cost as the number of training queries intersecting `mbr`. */
        double_t EstimateScanCost(const BoundingRectangle& mbr){
            if(query_scan_cost_estimator_!=NULL)
                return query_scan_cost_estimator_->EstimateOverlapCount(mbr);

            return 0.0;
        }

        /** Estimate and memoize the scan cost for a node's current MBR. */
        double_t CachedScanCost(RTreeNode* node){
            if(!node->cached_scan_cost_valid_){
                node->cached_scan_cost_ = EstimateScanCost(node->mbr_);
                node->cached_scan_cost_valid_ = true;
            }
            return node->cached_scan_cost_;
        }

        /** Mark a node's cached scan cost stale after changing its MBR. */
        void InvalidateCachedScanCost(RTreeNode* node){
            if(node!=NULL)
                node->cached_scan_cost_valid_ = false;
        }

        /** Read a positive integer build parameter from the environment. */
        static size_t ReadPositiveSizeTEnv(const char* name, size_t fallback){
            const char* raw_value = std::getenv(name);
            if(raw_value==NULL || raw_value[0]=='\0' || raw_value[0]=='-')
                return fallback;

            char* parse_end = NULL;
            unsigned long long parsed_value = std::strtoull(raw_value, &parse_end, 10);
            if(parse_end==raw_value || parsed_value==0)
                return fallback;

            return static_cast<size_t>(parsed_value);
        }

        /** Configure RW construction knobs, with env vars for quick experiments. */
        void ConfigureBuildParameters(){
            query_scan_cost_estimator_granularity_ = ReadPositiveSizeTEnv(
                "RW_QUERY_GRANULARITY",
                query_scan_cost_estimator_granularity_
            );
            choose_subtree_query_top_k_ = ReadPositiveSizeTEnv(
                "RW_CHOOSE_TOP_K",
                choose_subtree_query_top_k_
            );
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
            InvalidateCachedScanCost(root_);
         
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
            size_t &block_id = leaf_node_to_insert_point->local_block_id_;
            if(block_store_.InsertNewPointInBlock(pnt,block_id)>BLOCK_SIZE){
                // Take all points from given block and put back half of the points. 
                std::vector<Point> temp_arr = block_store_.FetchPointsInBlock(block_id);
                size_t split_pos = SplitPointsIntoTwo(temp_arr);
                leaf_node_to_insert_point->mbr_ = block_store_.ReassignPointsInBlock(block_id,temp_arr.begin(),temp_arr.begin()+split_pos);
                InvalidateCachedScanCost(leaf_node_to_insert_point);
                

                // Create a new leaf node and place back.
                RTreeNode* new_node = new RTreeNode();
                new_node->local_block_id_ = block_store_.InsertNewBlock(temp_arr.begin()+split_pos,temp_arr.end());
                new_node->mbr_=block_store_.FetchBoundingBoxForBlock(new_node->local_block_id_);
                parents[0]->children_.push_back(new_node);
                parents[0]->mbr_.UpdateBoundingBoxWithBoundingBox(new_node->mbr_);
                InvalidateCachedScanCost(parents[0]);

            }
            else{
                leaf_node_to_insert_point->mbr_.UpdateBoundingBoxWithPoint(pnt);
                InvalidateCachedScanCost(leaf_node_to_insert_point);
                parents[0]->mbr_.UpdateBoundingBoxWithBoundingBox(leaf_node_to_insert_point->mbr_);
                InvalidateCachedScanCost(parents[0]);
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
                    InvalidateCachedScanCost(new_node);

                    // update the current node by removing the secoond half of nodes and updating the bounding box.
                    parents[parent_to_split]->mbr_.SetToDefault();
                    parents[parent_to_split]->children_.resize(split_location);
                    for(auto& child: parents[parent_to_split]->children_)
                        parents[parent_to_split]->mbr_.UpdateBoundingBoxWithBoundingBox(child->mbr_);
                    InvalidateCachedScanCost(parents[parent_to_split]);


                    if(parent_to_split==parents.size()-1){              // if the rood node is overflowing create a new root node.
                        root_ = new RTreeNode();
                        root_->is_leaf_=false;
                        root_->children_.push_back(parents[parent_to_split]);
                        root_->mbr_.UpdateBoundingBoxWithBoundingBox(parents[parent_to_split]->mbr_);
                        root_->mbr_.UpdateBoundingBoxWithBoundingBox(new_node->mbr_);
                        root_->children_.push_back(new_node);
                        InvalidateCachedScanCost(root_);

                    }
                    else{
                        parents[parent_to_split+1]->mbr_.UpdateBoundingBoxWithBoundingBox(parents[parent_to_split]->mbr_);
                        parents[parent_to_split+1]->mbr_.UpdateBoundingBoxWithBoundingBox(new_node->mbr_);
                        parents[parent_to_split+1]->children_.push_back(new_node);
                        InvalidateCachedScanCost(parents[parent_to_split+1]);
                    }
                }
                else if(parent_to_split!=0){
                    parents[parent_to_split]->mbr_.UpdateBoundingBoxWithBoundingBox(parents[parent_to_split-1]->mbr_);
                    InvalidateCachedScanCost(parents[parent_to_split]);
                }

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
                    perimeters[i-(RW_MINBRANCH-1)]+=EstimateScanCost(temp_mbr);
                }

                temp_mbr.SetToDefault();

                for(int i=temp_arr.size()-1;i>temp_arr.size()-RW_MINBRANCH;i--)
                    temp_mbr.UpdateBoundingBoxWithBoundingBox(temp_arr[i]->mbr_);  

                for(int i=temp_arr.size()-RW_MINBRANCH;i>=RW_MINBRANCH;i--){
                    temp_mbr.UpdateBoundingBoxWithBoundingBox(temp_arr[i]->mbr_);
                    perimeters[i-RW_MINBRANCH]+=EstimateScanCost(temp_mbr);
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
                temp_mbr.SetToDefault();
                std::sort(temp_arr.begin(), temp_arr.end(), sort_nodes_high);
                
                for(int i=0;i<RW_MINBRANCH-1;i++)
                    temp_mbr.UpdateBoundingBoxWithBoundingBox(temp_arr[i]->mbr_);

                for(int i=RW_MINBRANCH-1;i<temp_arr.size()-RW_MINBRANCH;i++){
                    temp_mbr.UpdateBoundingBoxWithBoundingBox(temp_arr[i]->mbr_);
                    perimeters[i-(RW_MINBRANCH-1)]+=EstimateScanCost(temp_mbr);
                }

                temp_mbr.SetToDefault();

                for(int i=temp_arr.size()-1;i>temp_arr.size()-RW_MINBRANCH;i--)
                    temp_mbr.UpdateBoundingBoxWithBoundingBox(temp_arr[i]->mbr_);  

                for(int i=temp_arr.size()-RW_MINBRANCH;i>=RW_MINBRANCH;i--){
                    temp_mbr.UpdateBoundingBoxWithBoundingBox(temp_arr[i]->mbr_);
                    perimeters[i-RW_MINBRANCH]+=EstimateScanCost(temp_mbr);
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
                costs_at_split[i-(MINFILL-1)]+=EstimateScanCost(temp_mbr);
            }


            temp_mbr.SetToDefault();
            for(int i=temp_arr.size()-1;i>temp_arr.size()-MINFILL;i--)
                temp_mbr.UpdateBoundingBoxWithPoint(temp_arr[i]);

            for(int i=temp_arr.size()-MINFILL;i>=MINFILL;i--){
                temp_mbr.UpdateBoundingBoxWithPoint(temp_arr[i]);
                costs_at_split[i-MINFILL]+=EstimateScanCost(temp_mbr);
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
                costs_at_split[i-(MINFILL-1)]+=EstimateScanCost(temp_mbr);
            }


            temp_mbr.SetToDefault();
            for(int i=temp_arr.size()-1;i>temp_arr.size()-MINFILL;i--)
                temp_mbr.UpdateBoundingBoxWithPoint(temp_arr[i]); 

            for(int i=temp_arr.size()-MINFILL;i>=MINFILL;i--){
                temp_mbr.UpdateBoundingBoxWithPoint(temp_arr[i]);
                costs_at_split[i-MINFILL]+=EstimateScanCost(temp_mbr);
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

            // Cheaply rank all children by geometry, then run the query-cost oracle
            // only on the most promising candidates.
            size_t child_id=0;
            for(auto& child_ptr: node->children_){
                children_stats.emplace_back(child_id,child_ptr->mbr_,insert_pnt);
                child_id++;
            }

            auto cheap_nspi_comp = [](const NodeStatsPointInsert& ns1,const NodeStatsPointInsert& ns2) {
                return std::tie(ns1.delta_volume_, ns1.volume_, ns1.delta_perim_, ns1.perim_) <
                       std::tie(ns2.delta_volume_, ns2.volume_, ns2.delta_perim_, ns2.perim_);
            };
            std::sort(children_stats.begin(),children_stats.end(),cheap_nspi_comp);

            size_t candidate_count = std::min(
                std::max<size_t>(1, choose_subtree_query_top_k_),
                children_stats.size()
            );
            for(size_t candidate_id=0;candidate_id<candidate_count;candidate_id++){
                auto& candidate = children_stats[candidate_id];
                RTreeNode* child_node = node->children_[candidate.child_pointer_id_];
                candidate.cost_ = CachedScanCost(child_node);
                if(candidate.delta_volume_ < Constants::EPSILON_ERR)
                    candidate.delta_cost_ = 0.0;
                else
                    candidate.delta_cost_ = EstimateScanCost(candidate.expanded_mbr_)-candidate.cost_;
            }

            // Sort the candidate set by workload cost first, then cheap geometric tie-breakers.
            auto nspi_comp = [](const NodeStatsPointInsert& ns1,const NodeStatsPointInsert& ns2) {
                return std::tie(ns1.delta_cost_, ns1.cost_, ns1.delta_volume_, ns1.volume_) <
                       std::tie(ns2.delta_cost_, ns2.cost_, ns2.delta_volume_, ns2.volume_);
            };
            std::sort(children_stats.begin(),children_stats.begin()+candidate_count,nspi_comp);

            //IF you find nodes that covers the point to be inserted then pick the one with smallest volume (already achieved by nspi_comp)
            if(children_stats[0].delta_cost_ < Constants::EPSILON_ERR || node->children_[0]->is_leaf_==false)
                return ChooseSubtree(node->children_[children_stats[0].child_pointer_id_],insert_pnt,parents);

            //ELSE find the least-overlapping choice inside the query-ranked candidate set.
            double_t argmin_delta_overlap_volume = std::numeric_limits<double_t>::max();
            size_t argmin_child_id = 0;
            for(size_t candidate_id=0;candidate_id<candidate_count;candidate_id++){
                auto& ns = children_stats[candidate_id];
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
