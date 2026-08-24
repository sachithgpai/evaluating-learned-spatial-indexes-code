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
#include"../utils/build_profile.h"
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
        size_t split_nodes_query_top_k_{4};
        size_t split_points_query_top_k_{16};

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
            // Two phases with nothing shared but the estimator pointer: the first
            // reads only queries, the second only data. The cost-oracle calls the
            // second phase makes are counted rather than timed -- see the note in
            // build_profile.h on why wrapping them in a clock would distort them.
            CurrentBuildProfile().training_queries = queries.size();
            {
                ScopedPhase phase(CurrentBuildProfile().workload_model_s);
                BuildQueryScanCostEstimator(std::move(queries));
            }
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
            CurrentBuildProfile().oracle_calls++;
            ScopedPhase phase(CurrentBuildProfile().workload_oracle_s);
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

        /** Return true when inserting the point would leave this MBR unchanged. */
        bool PointDoesNotExpandMBR(const BoundingRectangle& mbr, const Point& pnt){
            for(size_t dim=0;dim<Constants::DIM;dim++)
                if(pnt.elements_[dim] < mbr.low_.elements_[dim] || pnt.elements_[dim] > mbr.high_.elements_[dim])
                    return false;

            return true;
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
            split_nodes_query_top_k_ = ReadPositiveSizeTEnv(
                "RW_SPLIT_NODES_TOP_K",
                split_nodes_query_top_k_
            );
            split_points_query_top_k_ = ReadPositiveSizeTEnv(
                "RW_SPLIT_POINTS_TOP_K",
                split_points_query_top_k_
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

            struct NodeSplitCandidate{
                size_t split_pos_;
                size_t sort_dim_;
                bool sort_high_;
                BoundingRectangle left_mbr_;
                BoundingRectangle right_mbr_;
                double_t cheap_perimeter_;
                double_t cheap_area_;
                double_t query_cost_;
            };

            std::vector<NodeSplitCandidate> split_candidates;
            split_candidates.reserve(4*(temp_arr.size()-2*RW_MINBRANCH+1));

            // Generate every legal split across low/high ordering in each dimension.
            // The cheap rank uses perimeter first because compact groups tend to reduce
            // future overlap, then area as a tie-breaker. Query cost is paid later only
            // for the best split_nodes_query_top_k_ candidates.
            auto add_candidates_for_order = [&](size_t candidate_sort_dim, bool sort_high) {
                auto sorter = [candidate_sort_dim, sort_high](const RTreeNode* n1,const RTreeNode* n2) {
                    if(sort_high)
                        return n1->mbr_.high_.elements_[candidate_sort_dim]<n2->mbr_.high_.elements_[candidate_sort_dim];
                    return n1->mbr_.low_.elements_[candidate_sort_dim]<n2->mbr_.low_.elements_[candidate_sort_dim];
                };
                std::sort(temp_arr.begin(), temp_arr.end(), sorter);

                std::vector<BoundingRectangle> right_mbrs(temp_arr.size()+1);
                right_mbrs[temp_arr.size()].SetToDefault();
                for(size_t pos=temp_arr.size();pos>0;pos--){
                    right_mbrs[pos-1] = right_mbrs[pos];
                    right_mbrs[pos-1].UpdateBoundingBoxWithBoundingBox(temp_arr[pos-1]->mbr_);
                }

                BoundingRectangle left_mbr;
                left_mbr.SetToDefault();
                for(size_t pos=0;pos<temp_arr.size()-RW_MINBRANCH;pos++){
                    left_mbr.UpdateBoundingBoxWithBoundingBox(temp_arr[pos]->mbr_);
                    size_t split_pos = pos+1;
                    if(split_pos<RW_MINBRANCH)
                        continue;

                    BoundingRectangle right_mbr = right_mbrs[split_pos];
                    split_candidates.push_back({
                        split_pos,
                        candidate_sort_dim,
                        sort_high,
                        left_mbr,
                        right_mbr,
                        left_mbr.Perimeter()+right_mbr.Perimeter(),
                        left_mbr.Area()+right_mbr.Area(),
                        0.0
                    });
                }
            };

            for(size_t ord=0;ord<Constants::DIM;ord++){
                add_candidates_for_order(ord, false);
                add_candidates_for_order(ord, true);
            }

            auto cheap_split_comp = [](const NodeSplitCandidate& lhs, const NodeSplitCandidate& rhs) {
                return std::tie(lhs.cheap_perimeter_, lhs.cheap_area_, lhs.split_pos_) <
                       std::tie(rhs.cheap_perimeter_, rhs.cheap_area_, rhs.split_pos_);
            };
            std::sort(split_candidates.begin(), split_candidates.end(), cheap_split_comp);

            size_t candidate_count = std::min(
                std::max<size_t>(1, split_nodes_query_top_k_),
                split_candidates.size()
            );

            // Refine only the cheap shortlist with the RW workload objective:
            // estimated scans for the left split MBR plus the right split MBR.
            double_t min_split_cost = std::numeric_limits<double_t>::max();
            size_t argmin_candidate = 0;
            for(size_t candidate_id=0;candidate_id<candidate_count;candidate_id++){
                auto& candidate = split_candidates[candidate_id];
                candidate.query_cost_ = EstimateScanCost(candidate.left_mbr_)+
                                        EstimateScanCost(candidate.right_mbr_);
                if(candidate.query_cost_<min_split_cost){
                    min_split_cost = candidate.query_cost_;
                    argmin_candidate = candidate_id;
                }
            }

            const auto& best_candidate = split_candidates[argmin_candidate];
            auto final_sorter = [&best_candidate](const RTreeNode* n1,const RTreeNode* n2) {
                if(best_candidate.sort_high_)
                    return n1->mbr_.high_.elements_[best_candidate.sort_dim_]<n2->mbr_.high_.elements_[best_candidate.sort_dim_];
                return n1->mbr_.low_.elements_[best_candidate.sort_dim_]<n2->mbr_.low_.elements_[best_candidate.sort_dim_];
            };
            std::sort(temp_arr.begin(), temp_arr.end(), final_sorter);

            return best_candidate.split_pos_;


        }


        /* Function splits the points into two. The function sorts the array according to best sort dim and returns the location of split*/
        size_t SplitPointsIntoTwo(std::vector<Point>& temp_arr){

            struct PointSplitCandidate{
                size_t split_pos_;
                size_t sort_dim_;
                BoundingRectangle left_mbr_;
                BoundingRectangle right_mbr_;
                double_t cheap_perimeter_;
                double_t cheap_area_;
                double_t query_cost_;
            };

            std::vector<PointSplitCandidate> split_candidates;
            split_candidates.reserve(2*(temp_arr.size()-2*MINFILL+1));

            // Generate every legal split in X and Y order. The cheap rank uses
            // perimeter first, then area, to shortlist compact candidate pages before
            // invoking the expensive query-density estimator.
            auto add_candidates_for_sort = [&](size_t candidate_sort_dim) {
                std::sort(temp_arr.begin(), temp_arr.end(), SortOrderer(candidate_sort_dim));

                std::vector<BoundingRectangle> right_mbrs(temp_arr.size()+1);
                right_mbrs[temp_arr.size()].SetToDefault();
                for(size_t pos=temp_arr.size();pos>0;pos--){
                    right_mbrs[pos-1] = right_mbrs[pos];
                    right_mbrs[pos-1].UpdateBoundingBoxWithPoint(temp_arr[pos-1]);
                }

                BoundingRectangle left_mbr;
                left_mbr.SetToDefault();
                for(size_t pos=0;pos<temp_arr.size()-MINFILL;pos++){
                    left_mbr.UpdateBoundingBoxWithPoint(temp_arr[pos]);
                    size_t split_pos = pos+1;
                    if(split_pos<MINFILL)
                        continue;

                    BoundingRectangle right_mbr = right_mbrs[split_pos];
                    split_candidates.push_back({
                        split_pos,
                        candidate_sort_dim,
                        left_mbr,
                        right_mbr,
                        left_mbr.Perimeter()+right_mbr.Perimeter(),
                        left_mbr.Area()+right_mbr.Area(),
                        0.0
                    });
                }
            };

            add_candidates_for_sort(SortX);
            add_candidates_for_sort(SortY);

            auto cheap_split_comp = [](const PointSplitCandidate& lhs, const PointSplitCandidate& rhs) {
                return std::tie(lhs.cheap_perimeter_, lhs.cheap_area_, lhs.split_pos_) <
                       std::tie(rhs.cheap_perimeter_, rhs.cheap_area_, rhs.split_pos_);
            };
            std::sort(split_candidates.begin(), split_candidates.end(), cheap_split_comp);

            size_t candidate_count = std::min(
                std::max<size_t>(1, split_points_query_top_k_),
                split_candidates.size()
            );

            // Refine only the top split_points_query_top_k_ cheap candidates with
            // the RW workload cost, avoiding hundreds of EstimateScanCost calls per
            // leaf split.
            double_t min_split_cost = std::numeric_limits<double_t>::max();
            size_t argmin_candidate = 0;
            for(size_t candidate_id=0;candidate_id<candidate_count;candidate_id++){
                auto& candidate = split_candidates[candidate_id];
                candidate.query_cost_ = EstimateScanCost(candidate.left_mbr_)+
                                        EstimateScanCost(candidate.right_mbr_);
                if(candidate.query_cost_<min_split_cost){
                    min_split_cost = candidate.query_cost_;
                    argmin_candidate = candidate_id;
                }
            }

            const auto& best_candidate = split_candidates[argmin_candidate];
            std::sort(temp_arr.begin(), temp_arr.end(), SortOrderer(best_candidate.sort_dim_));
            return best_candidate.split_pos_;
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

            // If a child already covers the point, inserting there does not enlarge
            // that child's MBR. The query-visible scan cost at this level therefore
            // cannot increase, so skip the query-cost oracle and choose the smallest
            // covering child by cheap geometry: area first, then perimeter.
            for(auto& child_stats: children_stats){
                RTreeNode* child_node = node->children_[child_stats.child_pointer_id_];
                if(PointDoesNotExpandMBR(child_node->mbr_, insert_pnt))
                    return ChooseSubtree(child_node,insert_pnt,parents);
            }

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
