#ifndef FLOOD_H
#define FLOOD_H


#include<vector>
#include<utility>
#include<tuple>
#include<iterator>
#include<string>
#include<array>
#include<algorithm>
#include<array>
#include"../utils/local_model.h"
#include"../utils/query.h"
#include"../utils/point.h"
#include"../utils/sort_tools.h"


class FloodIndex{
  public:
      int dim_order_[Constants::DIM];                 // the dimension order. [0] is will have the dimension on which the first sorting is done.
      int num_col_per_dim_[Constants::DIM];           // An array to hold how many splits are present per dim. Index[0] refers to 0th dim
      std::vector<double_t> split_location_per_dim_[Constants::DIM];   // split locations. Index[0] holds splits along 0th dim;
      // std::vector<LocalModel> cell_list_;             // A vector to store all the data associated with a grid cell.
      BlockStore block_store_;


      FloodIndex(int *dim_order,int *split_per_dim){
        std::copy(dim_order, dim_order+Constants::DIM, dim_order_);
        std::copy(split_per_dim, split_per_dim+Constants::DIM, num_col_per_dim_); 
      }   


      FloodIndex(std::array<int,Constants::DIM> dim_order,std::array<int,Constants::DIM> split_per_dim){
        std::copy(dim_order.begin(), dim_order.end(), dim_order_);
        std::copy(split_per_dim.begin(), split_per_dim.end(), num_col_per_dim_); 
      }
      
      void LoadElements(std::vector<Point> &data,bool disk_backup = true){  
        LearnSplitPointsPerDim(data);
        BoundingRectangle mbr;
        LoadElementsHelper(data.begin(),data.end(),0,mbr);
        if(disk_backup) block_store_.FinishedConstruction();
      }



      /**
       * @brief A function to perform a range query and return the query results along 
       *        with stats of range query processing. */
      std::vector<Point> RangeQuery(Query & query){

        std::vector<size_t> refined_blocks;
        Projection(refined_blocks,query);

        std::vector<Point> result_vec;
        Scan(refined_blocks,query,result_vec);

        return result_vec;
      }

      void Projection(std::vector<size_t> &projected_cells, Query &query){
        
        // the range of splits to follow for the index.
        std::vector<int> col_low(Constants::DIM,0); 
        std::vector<int> col_high(Constants::DIM,0);

        for(size_t i=0;i<Constants::DIM;i++){
          int d=dim_order_[i];
          col_low[d]=BinarySearch<double_t>(split_location_per_dim_[d],query.low_.elements_[d]);
          col_high[d]=BinarySearch<double_t>(split_location_per_dim_[d],query.high_.elements_[d]);
        }

        CollectCells(0,0,col_low,col_high,projected_cells);
      }



      void Scan( std::vector<size_t> &refined_blocks, Query &query, std::vector<Point> &result_vec){
        block_store_.FilterPointsFromBlocksForQuery(query,refined_blocks,result_vec);
      }




  private:

      /* A recursive function to step through and filter all the cells based on the columns*/
      void CollectCells(int dim_order_pos,int running_cell_id,const std::vector<int>& col_low, const std::vector<int>& col_high, std::vector<size_t>& collected_cells){
        if(dim_order_pos==Constants::DIM){
          collected_cells.push_back(running_cell_id);
          return;
        }

        int d=dim_order_[dim_order_pos];
        int multiplier = 1;       // multiplier for splits of all the lower priority dimensions
        for(size_t d_ix=dim_order_pos+1;d_ix<Constants::DIM;d_ix++) multiplier*= num_col_per_dim_[dim_order_[d_ix]];
        for(int ix = col_low[d];ix<=col_high[d];ix++)
          CollectCells(dim_order_pos+1,running_cell_id+ix*multiplier,col_low,col_high,collected_cells);
        return;
      }

      /* Learn adaptive split locations for the flood index  */
      void LearnSplitPointsPerDim(std::vector<Point> &data){
        for(size_t d=0;d<Constants::DIM;d++){
          if(num_col_per_dim_[d]==0) continue;       
          
          split_location_per_dim_[d].assign(num_col_per_dim_[d], std::numeric_limits<double_t>::min()); 
          std::sort(data.begin(),data.end(),SortOrderer(d));


          // use num_elements_per_col and left_over_elements to evenly split the data.
          int num_elements_per_col = int(data.size()/num_col_per_dim_[d]);
          int left_over_elements = data.size()%num_col_per_dim_[d];
          
          for(int j=0,location=0;j<num_col_per_dim_[d];j++,left_over_elements--){
            location+=num_elements_per_col + (left_over_elements>0);
            location = std::min(size_t(location),data.size()-1);
            split_location_per_dim_[d][j] = data[location].elements_[d];//+Constants::EPSILON_ERR;
          }
        }
      }


      void LoadElementsHelper(std::vector<Point>::iterator vec_begin,std::vector<Point>::iterator vec_end,int priority_order_pos,BoundingRectangle &mbr){

        
        if(priority_order_pos == (Constants::DIM)){ //if it is the final sort dimension
          // std::cout<<"New Cell. ("<<mbr.low_.elements_[0]<<","<<mbr.low_.elements_[1]<<"),("<<mbr.high_.elements_[0]<<","<<mbr.high_.elements_[1]<<")";
          // cell_list_.emplace_back(LocalModel(vec_begin,vec_end,block_store_,mbr));
          auto temp = block_store_.InsertNewBlock(vec_begin,vec_end);
          return;
        }


        int curr_dim = dim_order_[priority_order_pos];
        std::sort(vec_begin, vec_end, SortOrderer(curr_dim));

        typename std::vector<Point>::iterator col_begin_itr = vec_begin;
        Point bs_obj;
        for(int j=0;j<num_col_per_dim_[curr_dim];j++){
          bs_obj.elements_[curr_dim] = split_location_per_dim_[curr_dim][j];
          auto bs_comp = [curr_dim](Point& pt1_iter,Point pt2) { return pt1_iter.elements_[curr_dim]< pt2.elements_[curr_dim]; };
          typename std::vector<Point>::iterator col_end_itr = lower_bound(col_begin_itr,vec_end,bs_obj,bs_comp);
          mbr.low_.elements_[curr_dim]=col_begin_itr->elements_[curr_dim];
          mbr.high_.elements_[curr_dim]=std::prev(col_end_itr)->elements_[curr_dim];
          LoadElementsHelper(col_begin_itr,col_end_itr,priority_order_pos+1,mbr);
          col_begin_itr=col_end_itr;
        }
      }    
};



#endif
