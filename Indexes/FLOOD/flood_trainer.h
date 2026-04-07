#ifndef FLOOD_TRAINER_H
#define FLOOD_TRAINER_H

#include<algorithm>
#include<numeric>
#include<fstream>
#include<utility>
#include<chrono>
#include<limits>
#include<random>
#include"flood.h"


/**
 * Random-search the FLOOD configuration space using an in-memory dataset/query
 * workload and return the best split order plus split counts.
 */

std::pair<std::array<int,Constants::DIM>, std::array<int,Constants::DIM>> FloodTrainerRandomSearch(std::vector<Point> &datapoints,std::vector<Query> &queries){
    int since_last_improvement=0,max_samples=500;
    double_t best_time =  std::numeric_limits<double_t>::max();
    std::array<int, 2> best_dim_ord;
    std::array<int, 2> best_split_per_dim;
    while(since_last_improvement<25 && max_samples>0){
        int l = 2 + std::rand()%2000;  
        int m = 2 + std::rand()%2000;
        int n = std::rand()%2;
        std::array<int, 2> dim_ord{{n, (n+1)%2}};
        std::array<int, 2> split_per_dim{{l,m}};


        FloodIndex floodobj(dim_ord,split_per_dim);
        floodobj.LoadElements(datapoints,false);

        auto start = std::chrono::system_clock::now();
        for(int i=0;i<queries.size()*0.05;i++)
            floodobj.RangeQuery(queries[i]);
        auto end = std::chrono::system_clock::now();


        double_t elapsed_seconds = std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count();
        if(elapsed_seconds<best_time){
            best_time = elapsed_seconds;
            best_dim_ord = dim_ord;
            best_split_per_dim = split_per_dim;
            since_last_improvement=0;
        }
        since_last_improvement++;
        max_samples--;
    } 
    return std::make_pair(best_dim_ord,best_split_per_dim);

}

/**
 * Convenience overload that loads points and queries from files before running
 * the same random search procedure.
 */
std::pair<std::array<int,Constants::DIM>, std::array<int,Constants::DIM>> FloodTrainerRandomSearch(std::string datapoint_filename,std::string query_filename){
    auto start = std::chrono::system_clock::now();
    std::ifstream point_file;
    point_file.open(datapoint_filename);
    std::vector<Point> point_array;
    double_t a,b,c,d;
    while(!point_file.eof()){
        point_file>>a>>b;
        point_array.emplace_back(a,b);
    }
    point_file.close();


    Query query;
    for(size_t i=0;i<Constants::DIM;i++) query.dim_used_[i]=true;
    std::ifstream query_file;
    std::vector<Query> query_array;
    query_file.open(query_filename);
    while(!query_file.eof()){
        query_file>>a>>b>>c>>d;
        query_array.emplace_back(a,b,c,d);
    }
    query_file.close();

    std::mt19937 rng(std::random_device{}());
    std::shuffle(query_array.begin(),query_array.end(),rng);
    auto end = std::chrono::system_clock::now();
    
    return FloodTrainerRandomSearch(point_array,query_array);
}





#endif
