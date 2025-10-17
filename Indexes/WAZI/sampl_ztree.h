/**
 * @file sampl_ztree.h
 * @author Sachith (sachith.pai@helsinki.fi)
 * @brief An extenstion to the ZTRee to perform the optimization based on the uniform sampling.
 * @version 0.1
 * @date 2022-04-28
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#ifndef SAMPL_ZTREE_H
#define SAMPL_ZTREE_H

#include<cstdlib>
#include<ctime>
#include<cmath>
#include<vector>
#include<algorithm>
#include<iterator>
#include<utility>
#include<limits>
#include<string>
#include"ztree.h"
#include"../utils/density_estimators/dens_est.h"
#include"../utils/density_estimators/query_dens_est.h"
#include"../utils/point.h"
#include"../utils/sort_tools.h"
#include"../utils/constants.h"


enum QueryCases { AA, BB, CC, DD, AB, AC, BD, CD, AD};

class SamplZTree: public ZTree{
    
    public:

    DensEstTree * datapoint_density_estimator_;
    DensEstTree * query_starts_density_estimator_;
    DensEstTree * query_ends_density_estimator_;

    // QueryDensEstTree* query_density_estimator_;

    size_t random_sample_size_,dens_est_data_gran_,dens_est_query_gran_;
    double_t skip_alpha_;

    SamplZTree(std::vector<Point> &dataset, std::vector<Query> &queries, bool skipping_awareness = true, uint32_t random_sample_size=100,uint32_t dens_est_data_gran=16384, uint32_t dens_est_query_gran=256):ZTree(skipping_awareness),random_sample_size_(random_sample_size),dens_est_data_gran_(dens_est_data_gran),dens_est_query_gran_(dens_est_query_gran){


        num_datapoints_ = dataset.size();
        double_t data_low_x,data_low_y,data_high_x,data_high_y;

        std::sort(dataset.begin(),dataset.end(),SortOrderer(SortX)); // DEV why do we need the data lows and highs
        data_low_x=dataset[0].elements_[0];
        data_high_x=dataset[num_datapoints_-1].elements_[0];

        std::sort(dataset.begin(),dataset.end(),SortOrderer(SortY));
        data_low_y=dataset[0].elements_[1];
        data_high_y=dataset[num_datapoints_-1].elements_[1];

        for(auto& query: queries){ // expand the range of data to queries also.
            data_low_x=min(data_low_x,query.low_.elements_[0]);
            data_high_x=max(data_high_x,query.low_.elements_[0]);

            data_low_y=min(data_low_y,query.low_.elements_[1]);
            data_high_y=max(data_high_y,query.low_.elements_[1]);        
        }

        skip_alpha_ = (skipping_awareness)?0.000005:0.0005;
        root_=new ZtreeNode();
        node_cnt_++;
        root_->mbr_.low_=Point(data_low_x,data_low_y);
        root_->mbr_.high_=Point(data_high_x,data_high_y);

        
        std::srand(std::time(NULL)); // use current time as seed for random generator
        ConstructDensityEstimators(dataset, queries, BoundingRectangle(Point(data_low_x,data_low_y),Point(data_high_x+Constants::EPSILON_ERR, data_high_y+Constants::EPSILON_ERR)));


        BuildTree(root_,dataset.begin(),dataset.end());


        UpdatePagesInSubtree(root_);
        SetLeafPageMaskCreateLocalModels(root_,true);

        if(is_skipping_aware_)
            CalculateFwdPointers();

        BulkLoadData(dataset);

        
    }


    void ConstructDensityEstimators(std::vector<Point> &dataset, std::vector<Query> &queries, BoundingRectangle mbr){

        datapoint_density_estimator_ = new DensEstTree(dataset, dens_est_data_gran_,mbr);
        std::cout<<"No isseus with data estimator"<<"\n";

        /* Splitting up the endpoints and starts of queries*/
        std::vector<Point> rank_query_starts_de;
        std::vector<Point> rank_query_ends_de;
        for(auto &q: queries){
            rank_query_starts_de.push_back(q.low_);

            // if(mbr.CheckPointWithin(q.low_))
            //     continue;
            // else{
            //     std::cout<<"Q.Low_ not within given MBR."<<"\n";
            //     q.low_.Print();
            //     mbr.Print();
            //     std::cin.get();
            // }

            rank_query_ends_de.push_back(q.high_);

            // if(mbr.CheckPointWithin(q.high_))
            //     continue;
            // else{
            //     std::cout<<"Q.High_ not within given MBR."<<"\n";
            //     q.high_.Print();
            //     mbr.Print();
            //     std::cin.get();
            // }
        }

        // std::cout<<"No Points outside MBR"<<"\n";


        query_starts_density_estimator_ = new DensEstTree(rank_query_starts_de,dens_est_query_gran_,mbr);
        std::cout<<"No isseus with querylow estimator"<<"\n";
        query_ends_density_estimator_ = new DensEstTree(rank_query_ends_de,dens_est_query_gran_,mbr);
        std::cout<<"No isseus with queryhigh estimator"<<"\n";
    }


    /**
     * @brief Function takes the data points and build a Ztree. 
     * Should be identical for all versions of Ztree
     */
    void BuildTree(ZtreeNode *node, std::vector<Point>::iterator it_data_begin, std::vector<Point>::iterator it_data_end ){

        FindOptimalSplitPartition(node,it_data_begin,it_data_end); 

        if(node->is_leaf_){
            node->pages_in_subtree_=1;
            return;
        }

        for(size_t i=0;i<4;i++)
            node->children_[i]= new ZtreeNode(node->node_depth_+1);
        node_cnt_+=4;
        
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


        
        node->children_[0]->mbr_ = BoundingRectangle(
                                Point(node->mbr_.low_.elements_[0], node->mbr_.low_.elements_[1]),
                                Point(node->partition_.elements_[0], node->partition_.elements_[1]));
        BuildTree(node->children_[0],it_A_data_begin,it_A_data_end);

        node->children_[1]->mbr_ = BoundingRectangle(
                                Point(node->partition_.elements_[0], node->mbr_.low_.elements_[1]),
                                Point(node->mbr_.high_.elements_[0], node->partition_.elements_[1]));
        BuildTree(node->children_[1],it_B_data_begin,it_B_data_end);

        node->children_[2]->mbr_ = BoundingRectangle(
                                Point(node->mbr_.low_.elements_[0], node->partition_.elements_[1]),
                                Point(node->partition_.elements_[0], node->mbr_.high_.elements_[1]));
        BuildTree(node->children_[2],it_C_data_begin,it_C_data_end);

        node->children_[3]->mbr_ = BoundingRectangle(
                                Point(node->partition_.elements_[0], node->partition_.elements_[1]),
                                Point(node->mbr_.high_.elements_[0], node->mbr_.high_.elements_[1]));
        BuildTree(node->children_[3],it_D_data_begin,it_D_data_end);

        node->pages_in_subtree_ = node->children_[0]->pages_in_subtree_+ 
                                    node->children_[1]->pages_in_subtree_+
                                    node->children_[2]->pages_in_subtree_+
                                    node->children_[3]->pages_in_subtree_;

    }

    /**
     * @brief A function to sample `random_sample_size_` points uniformly from the dataspace covered by
     *      the current node and pick the best point based on the `Objective` function. 
     *      Starts from the default median split.
     * 
     * @param node  
     * @param it_data_begin 
     * @param it_data_end 
     * @return std::pair<Point,bool> 
     */
    void FindOptimalSplitPartition(ZtreeNode *node, std::vector<Point>::iterator it_data_begin,std::vector<Point>::iterator it_data_end){
        
        size_t num_data_here = std::distance(it_data_begin,it_data_end);
        
        if(num_data_here<=BLOCK_SIZE){
            node->is_leaf_ = true;
            node->partition_ = Point(-1.0,-1.0);
            return;
        }



        size_t mid_ix = (num_data_here-1)/2;
        double_t estimate_query_start_here =  query_starts_density_estimator_->EstimateCount(node->mbr_);
        double_t estimate_query_end_here =  query_ends_density_estimator_->EstimateCount(node->mbr_);



        /** IF: There exist enough queries in the region to affect the partitioning.
         *  ELSE: Just Pick the median.*/
        Point split = Point(-1.0,-1.0);
        /* Test the default first */
        std::sort(it_data_begin,it_data_end,SortOrderer(SortX));
        split.elements_[0] = (*(it_data_begin+mid_ix)).elements_[0];// + Constants::EPSILON_ERR;
        std::sort(it_data_begin,it_data_end,SortOrderer(SortY));
        split.elements_[1] = (*(it_data_begin+mid_ix)).elements_[1];// + Constants::EPSILON_ERR;
        bool best_obj_shape= false;

        if((estimate_query_start_here+estimate_query_end_here)>1.0){
        // if((estimate_num_queries_here)>1.0){
            std::pair<long double,bool> default_partition_obj=Objective(node,split);
            long double best_obj_value =default_partition_obj.first;
            best_obj_shape= default_partition_obj.second;

            /* randomly sample splits */
            for(int i=0;i<std::max(random_sample_size_,size_t(std::sqrt(num_data_here)));i++){
                size_t candidate_ix_x = std::rand()%num_data_here;
                size_t candidate_ix_y = std::rand()%num_data_here;

                Point candididate_split= Point((*(it_data_begin+candidate_ix_x)).elements_[0],(*(it_data_begin+candidate_ix_y)).elements_[1]);
                
                std::pair<long double,bool> objective_value = Objective(node,candididate_split);

                if(objective_value.first<best_obj_value){
                    split = candididate_split;
                    best_obj_value=objective_value.first;
                    best_obj_shape=objective_value.second;
                }
                
            }

        }

        node->ordering_ = best_obj_shape;
        node->partition_ = split;
        return;

    }



    /** Objective function based on decoupled queries*/
    std::pair<long double,bool> Objective(ZtreeNode *node,Point split){        
        long double na_raw,nb_raw,nc_raw,nd_raw,ea,eb,ec,ed,sa,sb,sc,sd;
        long double na,nb,nc,nd;


        na_raw = datapoint_density_estimator_->EstimateCount(BoundingRectangle(node->mbr_.low_,split));
        nb_raw = datapoint_density_estimator_->EstimateCount(BoundingRectangle(Point(split.elements_[0],node->mbr_.low_.elements_[1]), Point(node->mbr_.high_.elements_[0],split.elements_[1])));
        nc_raw = datapoint_density_estimator_->EstimateCount(BoundingRectangle(Point(node->mbr_.low_.elements_[0],split.elements_[1]),Point(split.elements_[0],node->mbr_.high_.elements_[1])));
        nd_raw = datapoint_density_estimator_->EstimateCount(BoundingRectangle(split,node->mbr_.high_));


        /* IF the partition creates very small page */
        if(na_raw<BLOCK_SIZE * 0.25 
            || nb_raw<BLOCK_SIZE * 0.25 
            || nc_raw<BLOCK_SIZE * 0.25 
            || nd_raw<BLOCK_SIZE * 0.25 )
            return std::make_pair(std::numeric_limits<long double>::max(),false);


        /**
         * @brief Any given subtree will have x number of pages where x is of the form 
         * 3y+1. So we round the raw counts to the nearest such x to promote better filling
         * of pages.
         */
        if(na_raw>BLOCK_SIZE) na = (3*ceil((na_raw-BLOCK_SIZE)/(3*BLOCK_SIZE))+1)*BLOCK_SIZE;
        else                  na = BLOCK_SIZE;

        if(nb_raw>BLOCK_SIZE) nb = (3*ceil((nb_raw-BLOCK_SIZE)/(3*BLOCK_SIZE))+1)*BLOCK_SIZE;
        else                  nb = BLOCK_SIZE;

        if(nc_raw>BLOCK_SIZE) nc = (3*ceil((nc_raw-BLOCK_SIZE)/(3*BLOCK_SIZE))+1)*BLOCK_SIZE;
        else                  nc = BLOCK_SIZE;

        if(nd_raw>BLOCK_SIZE) nd = (3*ceil((nd_raw-BLOCK_SIZE)/(3*BLOCK_SIZE))+1)*BLOCK_SIZE;
        else                  nd = BLOCK_SIZE;


        // TODO: explain why do we need to add 1 to the result?
        sa = query_starts_density_estimator_->EstimateCount(BoundingRectangle(node->mbr_.low_,split))+1;
        sb = query_starts_density_estimator_->EstimateCount(BoundingRectangle(Point(split.elements_[0],node->mbr_.low_.elements_[1]), Point(node->mbr_.high_.elements_[0],split.elements_[1])))+1;
        sc = query_starts_density_estimator_->EstimateCount(BoundingRectangle(Point(node->mbr_.low_.elements_[0],split.elements_[1]),Point(split.elements_[0],node->mbr_.high_.elements_[1])))+1;
        sd = query_starts_density_estimator_->EstimateCount(BoundingRectangle(split,node->mbr_.high_))+1;

        ea = query_ends_density_estimator_->EstimateCount(BoundingRectangle(node->mbr_.low_,split))+1;
        eb = query_ends_density_estimator_->EstimateCount(BoundingRectangle(Point(split.elements_[0],node->mbr_.low_.elements_[1]), Point(node->mbr_.high_.elements_[0],split.elements_[1])))+1;
        ec = query_ends_density_estimator_->EstimateCount(BoundingRectangle(Point(node->mbr_.low_.elements_[0],split.elements_[1]),Point(split.elements_[0],node->mbr_.high_.elements_[1])))+1;
        ed = query_ends_density_estimator_->EstimateCount(BoundingRectangle(split,node->mbr_.high_))+1;
        

        double_t shape_cost_YX, shape_cost_XY;
        if(is_skipping_aware_){
            // The shape cost terms nb*ec becomes just ec. Same for nc*sb becomes sb.
            shape_cost_YX = na*(eb+ec+ed) + nb*(sa+ed) + nc*(sa+ed) + nd*(sa+sb+sc) +ec+sb;
            shape_cost_XY = na*(eb+ec+ed) + nb*(sa+ed) + nc*(sa+ed) + nd*(sa+sb+sc) +eb+sc;
        }
        else{
            shape_cost_YX = na*(eb+ec+ed) + nb*(sa+skip_alpha_*ec+ed) + nc*(sa+skip_alpha_*sb+ed) + nd*(sa+sb+sc);
            shape_cost_XY = na*(eb+ec+ed) + nb*(sa+skip_alpha_*sc+ed) + nc*(sa+skip_alpha_*eb+ed) + nd*(sa+sb+sc);
        }

        long double lower_level_cost = (na*(sa+ea) + nb*(sb+eb) + nc*(sc+ec) + nd*(sd+ed));
        
        return std::make_pair(lower_level_cost+std::min(shape_cost_YX,shape_cost_XY),
                            (shape_cost_YX>shape_cost_XY));
    }

};

#endif 