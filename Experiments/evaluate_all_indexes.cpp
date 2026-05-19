#include<vector>
#include<iostream>
#include<fstream>
#include<chrono>
#include<string>
#include<algorithm>
#include<cmath>
#include<cstdlib>
#include<filesystem>
#include<iomanip>
#include<sstream>
#include<stdexcept>
#include<thread>
#include<random>

// All the models.
// #include <torch/script.h> // One-stop header..
// #include"../Indexes/RTree/rsmi_tree.h"
// #include"../Indexes/RTree/rtree_base.h"
#include"../Indexes/RTree/str_tree.h"
#include"../Indexes/RTree/rstar_tree.h"
#include"../Indexes/RTree/rw_tree.h"
#include"../Indexes/RTree/cur_tree.h"

#include"../Indexes/WAZI/base_ztree.h"
#include"../Indexes/WAZI/sampl_ztree.h"
#include"../Indexes/WAZI/zm_index.h"


#include"../Indexes/KDTree/kdtree.h"
#include"../Indexes/QDTree/qdtree.h"


#include"../Indexes/FLOOD/flood.h"
#include"../Indexes/FLOOD/flood_trainer.h"  


#include"../Indexes/utils/json.hpp"



using namespace std;
using json = nlohmann::json;   // using this to dump various logs.


string configured_experiment_path() {
    const char* env_path = getenv("EXPERIMENT_CONFIG");
    if(env_path != nullptr && string(env_path).size() > 0) {
        return string(env_path);
    }
    return (filesystem::path(PROJECT_ROOT) / "experiment_config.json").string();
}

json load_project_config() {
    string config_path = configured_experiment_path();
    ifstream config_file(config_path, ios::in);
    if(!config_file.is_open()) {
        throw runtime_error("Unable to open experiment config: " + config_path);
    }

    json project_config;
    config_file >> project_config;
    return project_config;
}

string configured_experiment_name(const json& project_config) {
    const char* env_name = getenv("EXPERIMENT_NAME");
    if(env_name != nullptr && string(env_name).size() > 0) {
        return string(env_name);
    }
    if(project_config.contains("default_experiment")) {
        return project_config.at("default_experiment").get<string>();
    }
    return "synthetic";
}

const json& experiment_config_for(
    const json& project_config,
    const string& experiment_name
) {
    if(
        !project_config.contains("experiments")
        || !project_config.at("experiments").contains(experiment_name)
    ) {
        throw runtime_error("Unknown experiment in config: " + experiment_name);
    }
    return project_config.at("experiments").at(experiment_name);
}

string fraction_to_selectivity_tag(double fraction) {
    long long scaled = llround(fraction * 1000000.0);
    stringstream tag;
    tag << setw(5) << setfill('0') << scaled;
    return tag.str();
}

vector<string> selectivity_tags_from_config(const json& experiment_config) {
    if(
        !experiment_config.contains("target_fractions")
        || !experiment_config.at("target_fractions").is_array()
        || experiment_config.at("target_fractions").empty()
    ) {
        throw runtime_error("Experiment config must define a non-empty target_fractions array.");
    }

    vector<string> selectivity_tags;
    for(const auto& fraction : experiment_config.at("target_fractions")) {
        selectivity_tags.push_back(fraction_to_selectivity_tag(fraction.get<double>()));
    }
    return selectivity_tags;
}

int query_entropy_variants_from_config(const json& experiment_config) {
    if(experiment_config.contains("query_entropy_variants")) {
        return experiment_config.at("query_entropy_variants").get<int>();
    }
    if(experiment_config.contains("num_query_scales")) {
        return experiment_config.at("num_query_scales").get<int>();
    }
    throw runtime_error(
        "Experiment config must define query_entropy_variants or num_query_scales."
    );
}

template <typename T>
void shuffle_vector(vector<T>& values) {
    static thread_local mt19937 rng(random_device{}());
    shuffle(values.begin(), values.end(), rng);
}

int main(int argc, char* argv[]){

    if(argc != 7 && argc != 8){
        cerr<<"Usage: "<<argv[0]
            <<" <dataset_name> <data_sample_num> <data_ent_id> <block_size> <query_ent_id> <selectivity_id> [result_file]"<<endl;
        return 1;
    }

    vector<string> selectivities_arr;
    int query_entropy_variants = 0;
    try {
        json project_config = load_project_config();
        string experiment_name = configured_experiment_name(project_config);
        const json& experiment_config =
            experiment_config_for(project_config, experiment_name);
        selectivities_arr = selectivity_tags_from_config(experiment_config);
        query_entropy_variants = query_entropy_variants_from_config(experiment_config);
        if(query_entropy_variants < 1) {
            throw runtime_error("Configured query entropy variant count must be >= 1.");
        }
    } catch(const exception& error) {
        cerr<<"Failed to load experiment config: "<<error.what()<<endl;
        return 1;
    }

    // std::mt19937_64 eng{std::random_device{}()};  // or seed however you want
    // std::uniform_int_distribution<> dist{0, 100};
    // std::this_thread::sleep_for(std::chrono::seconds{dist(eng)});


    string dataset_folder_name= string(argv[1]);
    int data_sample_num =  atoi(argv[2]);
    int data_ent_id = atoi(argv[3]);
    BLOCK_SIZE = atoi(argv[4]);
    int query_ent_id = atoi(argv[5]);
    int selectivity_id = atoi(argv[6]);
    if(selectivity_id < 0 || selectivity_id >= static_cast<int>(selectivities_arr.size())){
        cerr<<"Invalid selectivity_id: "<<selectivity_id<<endl;
        return 1;
    }
    string selectivity= selectivities_arr[selectivity_id];
    string line_num =
        (argc == 8)
            ? string(argv[7])
            : "P_"+to_string(BLOCK_SIZE)+"_D_"+to_string(data_sample_num)+"_DE_"+
                to_string(data_ent_id)+"_Q_"+to_string(query_ent_id)+"_S_"+selectivity+".jsonl";


    cout<<dataset_folder_name<<" "<<data_sample_num<<" "<<data_ent_id<<" "<<BLOCK_SIZE<<" "<<query_ent_id<<" "<<selectivity<<" "<<PROJECT_ROOT<<"\n";


    /*Reading the dataset and the entropy values*/
    vector<Point> datapoints;
    double_t a, b, c, d;
    ifstream pointsfile(PROJECT_ROOT+"Datasets/"+dataset_folder_name+"/"+to_string(data_sample_num)+"/datapoints/"+to_string(data_ent_id),ios::in);
    while (pointsfile >> a >> b)
        datapoints.push_back(Point(a,b));
    pointsfile.close();
    cout<<"Finished loading points |D|:"<<datapoints.size()<<"\n";

    vector<double_t> data_entropy;
    ifstream data_entropy_file(PROJECT_ROOT+"Datasets/"+dataset_folder_name+"/"+to_string(data_sample_num)+"/datapoints/entropy_values",ios::in);
    while (data_entropy_file >> a >> b)
        data_entropy.push_back(b);
    data_entropy_file.close();

    vector<Query> countbased_queries;
    ifstream countbased_queriesfile(PROJECT_ROOT+"Datasets/"+dataset_folder_name+"/"+to_string(data_sample_num)+"/queries/otherDist/"+to_string(data_ent_id)+"_"+selectivity+"_countbased_"+to_string(query_ent_id),ios::in);
    while (countbased_queriesfile >> a >> b >> c >> d)
        countbased_queries.push_back(Query(Point(a,b),Point(c,d)));
    countbased_queriesfile.close();
    cout<<"Finished loading countbased_queries |Q|:"<<countbased_queries.size()<<"\n";


    vector<double_t> query_entropy;
    ifstream query_entropy_file(PROJECT_ROOT+"Datasets/"+dataset_folder_name+"/"+to_string(data_sample_num)+"/queries/entropy_values",ios::in);
    while (query_entropy_file >> a >> b >>c)
        query_entropy.push_back(c);
    query_entropy_file.close();


    /* Query-agnostic indexes tree name */
    string query_agnostic_tree_name = "P_"+to_string(BLOCK_SIZE)+"_D_"+to_string(data_sample_num)+"_DE_"+to_string(data_ent_id);
    string tree_name = "P_"+to_string(BLOCK_SIZE)+"_D_"+to_string(data_sample_num)+"_DE_"+to_string(data_ent_id)+"_Q_"+to_string(query_ent_id)+"_S_"+selectivity;


    

    vector<Point> result_vec;
    size_t result_size;
    json log_json;
    log_json["block_size"] = BLOCK_SIZE;
    log_json["data_sample_num"] = data_sample_num;
    log_json["dataset_entropy_id"] = data_ent_id;
    if(data_ent_id < 1 || data_ent_id > static_cast<int>(data_entropy.size())){
        cerr<<"Invalid data_ent_id "<<data_ent_id
            <<" for "<<data_entropy.size()<<" data entropy rows."<<endl;
        return 1;
    }
    log_json["dataset_entropy"] = data_entropy[data_ent_id-1];
    log_json["query_entropy_id"] = query_ent_id;
    if(query_ent_id < 1 || query_ent_id > query_entropy_variants){
        cerr<<"Invalid query_ent_id "<<query_ent_id
            <<" for "<<query_entropy_variants
            <<" configured query entropy variants."<<endl;
        return 1;
    }
    size_t query_entropy_index =
        static_cast<size_t>((data_ent_id - 1) * query_entropy_variants + query_ent_id - 1);
    if(query_entropy_index >= query_entropy.size()){
        cerr<<"Invalid query_ent_id "<<query_ent_id
            <<" for "<<query_entropy_variants
            <<" configured query entropy variants and "
            <<query_entropy.size()<<" query entropy rows."<<endl;
        return 1;
    }
    log_json["query_entropy"] = query_entropy[query_entropy_index];
    log_json["selectivity"] = selectivity;
    
    std::vector<json> list_of_results;

    cout<<dataset_folder_name<<" "<<data_sample_num<<" "<<data_ent_id<<" "<<BLOCK_SIZE<<" "<<query_ent_id<<" "<<selectivity<<" WAZI Started"<<"\n";

    
    shuffle_vector(datapoints);
    shuffle_vector(countbased_queries);
    {   //############# WAZI #################
        // Training
        auto train_start = std::chrono::high_resolution_clock::now();
        SamplZTree wazi_obj(datapoints,countbased_queries);
        auto train_end = std::chrono::high_resolution_clock::now();
        double_t wazi_tree_build_time = chrono::duration_cast<chrono::nanoseconds>(train_end - train_start).count()/1000000000.0;

        log_json["model"]="WAZI";
        log_json["build_time"] = wazi_tree_build_time;

        //################## Query Processing Count Based ##################
        log_json["area_or_count_based"]="count";
        {    
            result_size=0;
            auto eval_start = std::chrono::high_resolution_clock::now();
            for(auto &query: countbased_queries){
                result_vec = wazi_obj.RangeQuery(query);  
                result_size+=result_vec.size();
            }
            auto eval_end = std::chrono::high_resolution_clock::now();

            log_json["result_size"]=result_size;
            log_json["query_latency"] = chrono::duration_cast<chrono::nanoseconds>(eval_end - eval_start).count()/countbased_queries.size();
        }

        { // Extra Query Processing metrics
            vector<vector<size_t>> refined_blocks(countbased_queries.size());


            auto refinement_start = std::chrono::high_resolution_clock::now();
            for(int q_id=0;q_id<countbased_queries.size();q_id++)
                wazi_obj.Projection(refined_blocks[q_id],countbased_queries[q_id]);
            auto refinement_end = std::chrono::high_resolution_clock::now();
            
            double_t number_of_refined_blocks=0;
            double_t number_of_points_scanned=0;
            for(int q_id=0;q_id<countbased_queries.size();q_id++){
                number_of_refined_blocks+=refined_blocks[q_id].size();
                number_of_points_scanned+=wazi_obj.block_store_.NumOfPointsInBlocks(refined_blocks[q_id]);
            }


            log_json["refinement_latency"] = chrono::duration_cast<chrono::nanoseconds>(refinement_end - refinement_start).count()/countbased_queries.size();
            log_json["number_of_refined_blocks"] = number_of_refined_blocks/countbased_queries.size();
            log_json["number_blocks_in_blockstore"] = wazi_obj.block_store_.block_list_.size();
            log_json["number_of_points_scanned"] = number_of_points_scanned/countbased_queries.size();
            log_json["block_size_quantiles"] = wazi_obj.block_store_.QuantilesOfBlockSizes();

        }

        {
            wazi_obj.block_store_.use_memory_mapped_data=true;
            size_t disk_backed_result_size=0;
            auto eval_start = std::chrono::high_resolution_clock::now();
            for(auto &query: countbased_queries){
                result_vec = wazi_obj.RangeQuery(query);  
                disk_backed_result_size+=result_vec.size();
            }
            auto eval_end = std::chrono::high_resolution_clock::now();

            log_json["disk_backed_result_size"]=disk_backed_result_size;
            log_json["disk_backed_query_latency"] = chrono::duration_cast<chrono::nanoseconds>(eval_end - eval_start).count()/countbased_queries.size();
        }
        list_of_results.push_back(log_json);
    }

    cout<<dataset_folder_name<<" "<<data_sample_num<<" "<<data_ent_id<<" "<<BLOCK_SIZE<<" "<<query_ent_id<<" "<<selectivity<<" WAZI Finished"<<"\n";




    cout<<dataset_folder_name<<" "<<data_sample_num<<" "<<data_ent_id<<" "<<BLOCK_SIZE<<" "<<query_ent_id<<" "<<selectivity<<" ZIndexStarted"<<"\n";
    
    shuffle_vector(datapoints);
    shuffle_vector(countbased_queries);
    {   //############# ZIndex #################
        // Training
        auto train_start = std::chrono::high_resolution_clock::now();
        BaseZTree zindex_obj(datapoints);
        auto train_end = std::chrono::high_resolution_clock::now();
        double_t zindex_tree_build_time = chrono::duration_cast<chrono::nanoseconds>(train_end - train_start).count()/1000000000.0;

        log_json["model"]="ZIndex";
        log_json["build_time"] = zindex_tree_build_time;

        //################## Query Processing Count Based ##################
        log_json["area_or_count_based"]="count";
        {
            result_size=0;
            auto eval_start = std::chrono::high_resolution_clock::now();
            for(auto &query: countbased_queries){
                result_vec = zindex_obj.RangeQuery(query);  
                result_size+=result_vec.size();
            }
            auto eval_end = std::chrono::high_resolution_clock::now();

            log_json["result_size"]=result_size;
            log_json["query_latency"] = chrono::duration_cast<chrono::nanoseconds>(eval_end - eval_start).count()/countbased_queries.size();
        }

        { // Extra Query Processing metrics
            vector<vector<size_t>> refined_blocks(countbased_queries.size());


            auto refinement_start = std::chrono::high_resolution_clock::now();
            for(int q_id=0;q_id<countbased_queries.size();q_id++)
                zindex_obj.Projection(refined_blocks[q_id],countbased_queries[q_id]);
            auto refinement_end = std::chrono::high_resolution_clock::now();
            
            double_t number_of_refined_blocks=0;
            double_t number_of_points_scanned=0;
            for(int q_id=0;q_id<countbased_queries.size();q_id++){
                number_of_refined_blocks+=refined_blocks[q_id].size();
                number_of_points_scanned+=zindex_obj.block_store_.NumOfPointsInBlocks(refined_blocks[q_id]);
            }

            log_json["refinement_latency"] = chrono::duration_cast<chrono::nanoseconds>(refinement_end - refinement_start).count()/countbased_queries.size();
            log_json["number_of_refined_blocks"] = number_of_refined_blocks/countbased_queries.size();
            log_json["number_blocks_in_blockstore"] = zindex_obj.block_store_.block_list_.size();
            log_json["number_of_points_scanned"] = number_of_points_scanned/countbased_queries.size();
            log_json["block_size_quantiles"] = zindex_obj.block_store_.QuantilesOfBlockSizes();

        }

        {
            zindex_obj.block_store_.use_memory_mapped_data=true;
            size_t disk_backed_result_size=0;
            auto eval_start = std::chrono::high_resolution_clock::now();
            for(auto &query: countbased_queries){
                result_vec = zindex_obj.RangeQuery(query);  
                disk_backed_result_size+=result_vec.size();
            }
            auto eval_end = std::chrono::high_resolution_clock::now();

            log_json["disk_backed_result_size"]=disk_backed_result_size;
            log_json["disk_backed_query_latency"] = chrono::duration_cast<chrono::nanoseconds>(eval_end - eval_start).count()/countbased_queries.size();
        }

        list_of_results.push_back(log_json);
    }
    cout<<dataset_folder_name<<" "<<data_sample_num<<" "<<data_ent_id<<" "<<BLOCK_SIZE<<" "<<query_ent_id<<" "<<selectivity<<" ZIndex Finished"<<"\n";



    cout<<dataset_folder_name<<" "<<data_sample_num<<" "<<data_ent_id<<" "<<BLOCK_SIZE<<" "<<query_ent_id<<" "<<selectivity<<" ZM-Index Started"<<"\n";
    
    shuffle_vector(datapoints);
    shuffle_vector(countbased_queries);
    {   //############# ZM-Index #################
        // Training
        auto train_start = std::chrono::high_resolution_clock::now();
        ZMIndex zmindex_obj(datapoints);
        auto train_end = std::chrono::high_resolution_clock::now();
        double_t zmindex_tree_build_time = chrono::duration_cast<chrono::nanoseconds>(train_end - train_start).count()/1000000000.0;
        std::cout<<"Finished building ZM"<<std::endl;
        log_json["model"]="ZM";
        log_json["build_time"] = zmindex_tree_build_time;



        //################## Query Processing Count Based ##################
        log_json["area_or_count_based"]="count";
        {
            /* The rank space mapping of data could be bottle neck. To avoid that we perform rankspace mapping of the countbased_queries offline.*/
            vector<WrappedPoint> query_lows;
            vector<WrappedPoint> query_highs;
            WrappedPoint query_low_rankspace, query_high_rankspace;
            for(auto &query: countbased_queries){
                query_low_rankspace = zmindex_obj.rank_space_map_->Transform(query.low_);
                query_low_rankspace.curve_value_ = compute_Z_value(query_low_rankspace);
                query_lows.push_back(query_low_rankspace);

                query_high_rankspace = zmindex_obj.rank_space_map_->Transform(query.high_);
                query_high_rankspace.curve_value_ = compute_Z_value(query_high_rankspace);
                query_highs.push_back(query_high_rankspace);

            }
            


            result_size=0;
            auto eval_start = std::chrono::high_resolution_clock::now();
            for(int i=0;i<countbased_queries.size();i++){
                result_vec = zmindex_obj.RangeQuery(countbased_queries[i],query_lows[i],query_highs[i]); 
                result_size+=result_vec.size();
            }
            auto eval_end = std::chrono::high_resolution_clock::now();

            log_json["result_size"]=result_size;
            log_json["query_latency"] = chrono::duration_cast<chrono::nanoseconds>(eval_end - eval_start).count()/countbased_queries.size();


            // Extra Query Processing metrics
            vector<vector<size_t>> projected_blocks(countbased_queries.size());
            vector<vector<size_t>> refined_blocks(countbased_queries.size());


            auto refinement_start = std::chrono::high_resolution_clock::now();
            for(int q_id=0;q_id<countbased_queries.size();q_id++){
                zmindex_obj.Projection(projected_blocks[q_id],query_lows[q_id],query_highs[q_id]);
                zmindex_obj.Refinement(refined_blocks[q_id],countbased_queries[q_id],projected_blocks[q_id]);
            }
            auto refinement_end = std::chrono::high_resolution_clock::now();
            
            double_t number_of_refined_blocks=0;
            double_t number_of_points_scanned=0;
            for(int q_id=0;q_id<countbased_queries.size();q_id++){
                number_of_refined_blocks+=refined_blocks[q_id].size();
                number_of_points_scanned+=zmindex_obj.block_store_.NumOfPointsInBlocks(refined_blocks[q_id]);
            }


            log_json["refinement_latency"] = chrono::duration_cast<chrono::nanoseconds>(refinement_end - refinement_start).count()/countbased_queries.size();
            log_json["number_of_refined_blocks"] = number_of_refined_blocks/countbased_queries.size();
            log_json["number_blocks_in_blockstore"] = zmindex_obj.block_store_.block_list_.size();
            log_json["number_of_points_scanned"] = number_of_points_scanned/countbased_queries.size();
            log_json["block_size_quantiles"] = zmindex_obj.block_store_.QuantilesOfBlockSizes();

            std::cout<<"Starting Disk based queries"<<std::endl;

            zmindex_obj.block_store_.use_memory_mapped_data=true;
            size_t disk_backed_result_size=0;
            auto eval_start2 = std::chrono::high_resolution_clock::now();
            for(int i=0;i<countbased_queries.size();i++){
                result_vec = zmindex_obj.RangeQuery(countbased_queries[i],query_lows[i],query_highs[i]); 
                disk_backed_result_size+=result_vec.size();
            }
            auto eval_end2 = std::chrono::high_resolution_clock::now();

            log_json["disk_backed_result_size"]=disk_backed_result_size;
            log_json["disk_backed_query_latency"] = chrono::duration_cast<chrono::nanoseconds>(eval_end2 - eval_start2).count()/countbased_queries.size();
        }

        list_of_results.push_back(log_json);
    }
    cout<<dataset_folder_name<<" "<<data_sample_num<<" "<<data_ent_id<<" "<<BLOCK_SIZE<<" "<<query_ent_id<<" "<<selectivity<<" ZM-Index Finished"<<"\n";
    
    
    cout<<dataset_folder_name<<" "<<data_sample_num<<" "<<data_ent_id<<" "<<BLOCK_SIZE<<" "<<query_ent_id<<" "<<selectivity<<" GRID Started"<<"\n";
    
    shuffle_vector(datapoints);
    shuffle_vector(countbased_queries);
    {   //############# GRID #################
        // Training
        auto train_start = std::chrono::high_resolution_clock::now();
        int num_splits_uniform_grid = int(sqrt(datapoints.size()/BLOCK_SIZE));
        std::array<int, 2> split_per_dim{num_splits_uniform_grid, num_splits_uniform_grid};
        std::array<int, 2> dim_order{0, 1};
        FloodIndex unigrid_obj( dim_order,split_per_dim);
        unigrid_obj.LoadElements(datapoints);
        auto train_end = std::chrono::high_resolution_clock::now();
        double_t unigrid_build_time = chrono::duration_cast<chrono::nanoseconds>(train_end - train_start).count()/1000000000.0;

        log_json["model"]="GRID";
        log_json["build_time"] = unigrid_build_time;


        //################## Query Processing Count Based ##################
        log_json["area_or_count_based"]="count";
        {
            result_size=0;
            auto eval_start = std::chrono::high_resolution_clock::now();
            for(auto &query: countbased_queries){
                result_vec = unigrid_obj.RangeQuery(query); 
                result_size+=result_vec.size();
            }
            auto eval_end = std::chrono::high_resolution_clock::now();

            log_json["result_size"]=result_size;
            log_json["query_latency"] = chrono::duration_cast<chrono::nanoseconds>(eval_end - eval_start).count()/countbased_queries.size();

            vector<vector<size_t>> refined_blocks(countbased_queries.size());


            auto refinement_start = std::chrono::high_resolution_clock::now();
            for(int q_id=0;q_id<countbased_queries.size();q_id++)
                unigrid_obj.Projection(refined_blocks[q_id],countbased_queries[q_id]);

            auto refinement_end = std::chrono::high_resolution_clock::now();
            
            double_t number_of_refined_blocks=0;
            double_t number_of_points_scanned=0;
            for(int q_id=0;q_id<countbased_queries.size();q_id++){
            number_of_refined_blocks+=refined_blocks[q_id].size();
            number_of_points_scanned+=unigrid_obj.block_store_.NumOfPointsInBlocks(refined_blocks[q_id]);
            }


            log_json["refinement_latency"] = chrono::duration_cast<chrono::nanoseconds>(refinement_end - refinement_start).count()/countbased_queries.size();
            log_json["number_of_refined_blocks"] = number_of_refined_blocks/countbased_queries.size();
            log_json["number_blocks_in_blockstore"] = unigrid_obj.block_store_.block_list_.size();
            log_json["number_of_points_scanned"] = number_of_points_scanned/countbased_queries.size();
            log_json["block_size_quantiles"] = unigrid_obj.block_store_.QuantilesOfBlockSizes();

        }

        {
            unigrid_obj.block_store_.use_memory_mapped_data=true;
            size_t disk_backed_result_size=0;
            auto eval_start = std::chrono::high_resolution_clock::now();
            for(auto &query: countbased_queries){
                result_vec = unigrid_obj.RangeQuery(query);  
                disk_backed_result_size+=result_vec.size();
            }
            auto eval_end = std::chrono::high_resolution_clock::now();

            log_json["disk_backed_result_size"]=disk_backed_result_size;
            log_json["disk_backed_query_latency"] = chrono::duration_cast<chrono::nanoseconds>(eval_end - eval_start).count()/countbased_queries.size();
        }
        list_of_results.push_back(log_json);
    }
    cout<<dataset_folder_name<<" "<<data_sample_num<<" "<<data_ent_id<<" "<<BLOCK_SIZE<<" "<<query_ent_id<<" "<<selectivity<<" GRID Finished"<<"\n";
    
    
    
    
    cout<<dataset_folder_name<<" "<<data_sample_num<<" "<<data_ent_id<<" "<<BLOCK_SIZE<<" "<<query_ent_id<<" "<<selectivity<<" FLOOD Started"<<"\n";
    
    shuffle_vector(datapoints);
    shuffle_vector(countbased_queries);
    {   //############# FLOOD #################
        // Training
        auto train_start = std::chrono::high_resolution_clock::now();
        auto flood_config = FloodTrainerRandomSearch(datapoints,countbased_queries);
        FloodIndex flood_obj(flood_config.first,flood_config.second);
        flood_obj.LoadElements(datapoints);
        auto train_end = std::chrono::high_resolution_clock::now();


        double_t flood_build_time = chrono::duration_cast<chrono::nanoseconds>(train_end - train_start).count()/1000000000.0;

        log_json["model"]="FLOOD";
        log_json["build_time"] = flood_build_time;

        //################## Query Processing Count Based ##################
        log_json["area_or_count_based"]="count";

        result_size=0;
        auto eval_start = std::chrono::high_resolution_clock::now();
        for(auto &query: countbased_queries){
            result_vec = flood_obj.RangeQuery(query); 
            result_size+=result_vec.size();
        }
        auto eval_end = std::chrono::high_resolution_clock::now();

        log_json["result_size"]=result_size;
        log_json["query_latency"] = chrono::duration_cast<chrono::nanoseconds>(eval_end - eval_start).count()/countbased_queries.size();


        { // Extra Query Processing metrics
            vector<vector<size_t>> refined_blocks(countbased_queries.size());


            auto refinement_start = std::chrono::high_resolution_clock::now();
            for(int q_id=0;q_id<countbased_queries.size();q_id++)
                flood_obj.Projection(refined_blocks[q_id],countbased_queries[q_id]);
            auto refinement_end = std::chrono::high_resolution_clock::now();
            
            double_t number_of_refined_blocks=0;
            double_t number_of_points_scanned=0;
            for(int q_id=0;q_id<countbased_queries.size();q_id++){
            number_of_refined_blocks+=refined_blocks[q_id].size();
            number_of_points_scanned+=flood_obj.block_store_.NumOfPointsInBlocks(refined_blocks[q_id]);
            }


            log_json["refinement_latency"] = chrono::duration_cast<chrono::nanoseconds>(refinement_end - refinement_start).count()/countbased_queries.size();
            log_json["number_of_refined_blocks"] = number_of_refined_blocks/countbased_queries.size();
            log_json["number_blocks_in_blockstore"] = flood_obj.block_store_.block_list_.size();
            log_json["number_of_points_scanned"] = number_of_points_scanned/countbased_queries.size();
            log_json["block_size_quantiles"] = flood_obj.block_store_.QuantilesOfBlockSizes();

        }

        {
            flood_obj.block_store_.use_memory_mapped_data=true;
            size_t disk_backed_result_size=0;
            auto eval_start = std::chrono::high_resolution_clock::now();
            for(auto &query: countbased_queries){
                result_vec = flood_obj.RangeQuery(query);  
                disk_backed_result_size+=result_vec.size();
            }
            auto eval_end = std::chrono::high_resolution_clock::now();

            log_json["disk_backed_result_size"]=disk_backed_result_size;
            log_json["disk_backed_query_latency"] = chrono::duration_cast<chrono::nanoseconds>(eval_end - eval_start).count()/countbased_queries.size();
        }

        list_of_results.push_back(log_json);
    }
    cout<<dataset_folder_name<<" "<<data_sample_num<<" "<<data_ent_id<<" "<<BLOCK_SIZE<<" "<<query_ent_id<<" "<<selectivity<<" FLOOD Finished"<<"\n";
    
    
    
    cout<<dataset_folder_name<<" "<<data_sample_num<<" "<<data_ent_id<<" "<<BLOCK_SIZE<<" "<<query_ent_id<<" "<<selectivity<<" STR Started"<<"\n";
    
    shuffle_vector(datapoints);
    shuffle_vector(countbased_queries);
    {   /* ##########################    STR   ######################################*/
        // Training
        auto train_start = std::chrono::high_resolution_clock::now();
        STRTree str_tree_obj(datapoints);
        auto train_end = std::chrono::high_resolution_clock::now();
        double_t str_tree_build_time = chrono::duration_cast<chrono::nanoseconds>(train_end - train_start).count()/1000000000.0;

        log_json["model"]="STR";
        log_json["build_time"] = str_tree_build_time;

        //################## Query Processing Count Based ##################
        {
            log_json["area_or_count_based"]="count";
            result_size=0;
            auto eval_start = std::chrono::high_resolution_clock::now();
            for(auto &query: countbased_queries){
                result_vec = str_tree_obj.RangeQuery(query); 
                result_size+=result_vec.size();
            }
            auto eval_end = std::chrono::high_resolution_clock::now();

            log_json["result_size"]=result_size;
            log_json["query_latency"] = chrono::duration_cast<chrono::nanoseconds>(eval_end - eval_start).count()/countbased_queries.size();

            // Extra Query Processing metrics
            vector<vector<size_t>> refined_blocks(countbased_queries.size());


            auto refinement_start = std::chrono::high_resolution_clock::now();
            for(int q_id=0;q_id<countbased_queries.size();q_id++)
            str_tree_obj.Projection(refined_blocks[q_id],countbased_queries[q_id],str_tree_obj.root_);
            auto refinement_end = std::chrono::high_resolution_clock::now();
            
            double_t number_of_refined_blocks=0;
            double_t number_of_points_scanned=0;
            for(int q_id=0;q_id<countbased_queries.size();q_id++){
            number_of_refined_blocks+=refined_blocks[q_id].size();
            number_of_points_scanned+=str_tree_obj.block_store_.NumOfPointsInBlocks(refined_blocks[q_id]);
            }


            log_json["refinement_latency"] = chrono::duration_cast<chrono::nanoseconds>(refinement_end - refinement_start).count()/countbased_queries.size();
            log_json["number_of_refined_blocks"] = number_of_refined_blocks/countbased_queries.size();
            log_json["number_blocks_in_blockstore"] = str_tree_obj.block_store_.block_list_.size();
            log_json["number_of_points_scanned"] = number_of_points_scanned/countbased_queries.size();
            log_json["block_size_quantiles"] = str_tree_obj.block_store_.QuantilesOfBlockSizes();


            {
                str_tree_obj.block_store_.use_memory_mapped_data=true;
                size_t disk_backed_result_size=0;
                auto eval_start = std::chrono::high_resolution_clock::now();
                for(auto &query: countbased_queries){
                    result_vec = str_tree_obj.RangeQuery(query);  
                    disk_backed_result_size+=result_vec.size();
                }
                auto eval_end = std::chrono::high_resolution_clock::now();

                log_json["disk_backed_result_size"]=disk_backed_result_size;
                log_json["disk_backed_query_latency"] = chrono::duration_cast<chrono::nanoseconds>(eval_end - eval_start).count()/countbased_queries.size();
            }


            list_of_results.push_back(log_json);
        }

    }
    cout<<dataset_folder_name<<" "<<data_sample_num<<" "<<data_ent_id<<" "<<BLOCK_SIZE<<" "<<query_ent_id<<" "<<selectivity<<" STR Finished"<<"\n";
    
    
    
    cout<<dataset_folder_name<<" "<<data_sample_num<<" "<<data_ent_id<<" "<<BLOCK_SIZE<<" "<<query_ent_id<<" "<<selectivity<<" RSTAR Started"<<"\n";

    
    shuffle_vector(datapoints);
    shuffle_vector(countbased_queries);
    {   //########## RSTAR #################
        // Training
        auto train_start = std::chrono::high_resolution_clock::now();
        RSTARTree rstar_tree_obj(datapoints);
        auto train_end = std::chrono::high_resolution_clock::now();
        double_t rstar_tree_build_time = chrono::duration_cast<chrono::nanoseconds>(train_end - train_start).count()/1000000000.0;

        log_json["model"]="RSTAR";
        log_json["build_time"] = rstar_tree_build_time;


        //################## Query Processing Count Based ##################
        log_json["area_or_count_based"]="count";
        {   
            result_size=0;
            auto eval_start = std::chrono::high_resolution_clock::now();
            for(auto &query: countbased_queries){
                result_vec = rstar_tree_obj.RangeQuery(query); 
                result_size+=result_vec.size();
            }
            auto eval_end = std::chrono::high_resolution_clock::now();

            log_json["result_size"]=result_size;
            log_json["query_latency"] = chrono::duration_cast<chrono::nanoseconds>(eval_end - eval_start).count()/countbased_queries.size();


            // Extra Query Processing metrics
            vector<vector<size_t>> refined_blocks(countbased_queries.size());


            auto refinement_start = std::chrono::high_resolution_clock::now();
            for(int q_id=0;q_id<countbased_queries.size();q_id++)
                rstar_tree_obj.Projection(refined_blocks[q_id],countbased_queries[q_id],rstar_tree_obj.root_);
            auto refinement_end = std::chrono::high_resolution_clock::now();
            
            double_t number_of_refined_blocks=0;
            double_t number_of_points_scanned=0;
            for(int q_id=0;q_id<countbased_queries.size();q_id++){
            number_of_refined_blocks+=refined_blocks[q_id].size();
            number_of_points_scanned+=rstar_tree_obj.block_store_.NumOfPointsInBlocks(refined_blocks[q_id]);
            }


            log_json["refinement_latency"] = chrono::duration_cast<chrono::nanoseconds>(refinement_end - refinement_start).count()/countbased_queries.size();
            log_json["number_of_refined_blocks"] = number_of_refined_blocks/countbased_queries.size();
            log_json["number_blocks_in_blockstore"] = rstar_tree_obj.block_store_.block_list_.size();
            log_json["number_of_points_scanned"] = number_of_points_scanned/countbased_queries.size();
            log_json["block_size_quantiles"] = rstar_tree_obj.block_store_.QuantilesOfBlockSizes();

        }

        {
            rstar_tree_obj.block_store_.use_memory_mapped_data=true;
            size_t disk_backed_result_size=0;
            auto eval_start = std::chrono::high_resolution_clock::now();
            for(auto &query: countbased_queries){
                result_vec = rstar_tree_obj.RangeQuery(query);  
                disk_backed_result_size+=result_vec.size();
            }
            auto eval_end = std::chrono::high_resolution_clock::now();

            log_json["disk_backed_result_size"]=disk_backed_result_size;
            log_json["disk_backed_query_latency"] = chrono::duration_cast<chrono::nanoseconds>(eval_end - eval_start).count()/countbased_queries.size();
        }

        list_of_results.push_back(log_json);
    }
    cout<<dataset_folder_name<<" "<<data_sample_num<<" "<<data_ent_id<<" "<<BLOCK_SIZE<<" "<<query_ent_id<<" "<<selectivity<<" RSTAR Finished"<<"\n";
    
    
    
    cout<<dataset_folder_name<<" "<<data_sample_num<<" "<<data_ent_id<<" "<<BLOCK_SIZE<<" "<<query_ent_id<<" "<<selectivity<<" CUR Started"<<"\n";

    
    shuffle_vector(datapoints);
    shuffle_vector(countbased_queries);
    {   //########## CUR #################
        // Training
        auto train_start = std::chrono::high_resolution_clock::now();
        CURTree cur_tree_obj(datapoints,countbased_queries);
        cout<<" CUR Finished Building"<<"\n";
        auto train_end = std::chrono::high_resolution_clock::now();
        double_t cur_tree_build_time = chrono::duration_cast<chrono::nanoseconds>(train_end - train_start).count()/1000000000.0;

        log_json["model"]="CUR";
        log_json["build_time"] = cur_tree_build_time;
        

        //################## Query Processing Count Based ##################
        log_json["area_or_count_based"]="count";
        {
            result_size=0;
            auto eval_start = std::chrono::high_resolution_clock::now();
            for(auto &query: countbased_queries){
                result_vec = cur_tree_obj.RangeQuery(query); 
                result_size+=result_vec.size();
            }
            auto eval_end = std::chrono::high_resolution_clock::now();

            log_json["result_size"]=result_size;
            log_json["query_latency"] = chrono::duration_cast<chrono::nanoseconds>(eval_end - eval_start).count()/countbased_queries.size();


            // Extra Query Processing metrics
            vector<vector<size_t>> refined_blocks(countbased_queries.size());


            auto refinement_start = std::chrono::high_resolution_clock::now();
            for(int q_id=0;q_id<countbased_queries.size();q_id++)
            cur_tree_obj.Projection(refined_blocks[q_id],countbased_queries[q_id],cur_tree_obj.root_);
            auto refinement_end = std::chrono::high_resolution_clock::now();
            
            double_t number_of_refined_blocks=0;
            double_t number_of_points_scanned=0;
            for(int q_id=0;q_id<countbased_queries.size();q_id++){
            number_of_refined_blocks+=refined_blocks[q_id].size();
            number_of_points_scanned+=cur_tree_obj.block_store_.NumOfPointsInBlocks(refined_blocks[q_id]);
            }


            log_json["refinement_latency"] = chrono::duration_cast<chrono::nanoseconds>(refinement_end - refinement_start).count()/countbased_queries.size();
            log_json["number_of_refined_blocks"] = number_of_refined_blocks/countbased_queries.size();
            log_json["number_blocks_in_blockstore"] = cur_tree_obj.block_store_.block_list_.size();
            log_json["number_of_points_scanned"] = number_of_points_scanned/countbased_queries.size();
            log_json["block_size_quantiles"] = cur_tree_obj.block_store_.QuantilesOfBlockSizes();

        }


        {
            cur_tree_obj.block_store_.use_memory_mapped_data=true;
            size_t disk_backed_result_size=0;
            auto eval_start = std::chrono::high_resolution_clock::now();
            for(auto &query: countbased_queries){
                result_vec = cur_tree_obj.RangeQuery(query);  
                disk_backed_result_size+=result_vec.size();
            }
            auto eval_end = std::chrono::high_resolution_clock::now();

            log_json["disk_backed_result_size"]=disk_backed_result_size;
            log_json["disk_backed_query_latency"] = chrono::duration_cast<chrono::nanoseconds>(eval_end - eval_start).count()/countbased_queries.size();
        }

        list_of_results.push_back(log_json);
    }
    cout<<dataset_folder_name<<" "<<data_sample_num<<" "<<data_ent_id<<" "<<BLOCK_SIZE<<" "<<query_ent_id<<" "<<selectivity<<" CUR Finished"<<"\n";
    
    
    
    cout<<dataset_folder_name<<" "<<data_sample_num<<" "<<data_ent_id<<" "<<BLOCK_SIZE<<" "<<query_ent_id<<" "<<selectivity<<" RW Started"<<"\n";

    
    shuffle_vector(datapoints);
    shuffle_vector(countbased_queries);
    {   //########## RW #################
        // Training
        auto train_start = std::chrono::high_resolution_clock::now();
        RWTree rw_tree_obj(datapoints,countbased_queries);
        auto train_end = std::chrono::high_resolution_clock::now();
        double_t rw_tree_build_time = chrono::duration_cast<chrono::nanoseconds>(train_end - train_start).count()/1000000000.0;

        log_json["model"]="RW";
        log_json["build_time"] = rw_tree_build_time;
        

        //################## Query Processing Count Based ##################
        log_json["area_or_count_based"]="count";
        {
            result_size=0;
            auto eval_start = std::chrono::high_resolution_clock::now();
            for(auto &query: countbased_queries){
                result_vec = rw_tree_obj.RangeQuery(query); 
                result_size+=result_vec.size();
            }
            auto eval_end = std::chrono::high_resolution_clock::now();

            log_json["result_size"]=result_size;
            log_json["query_latency"] = chrono::duration_cast<chrono::nanoseconds>(eval_end - eval_start).count()/countbased_queries.size();


            // Extra Query Processing metrics
            vector<vector<size_t>> refined_blocks(countbased_queries.size());

            auto refinement_start = std::chrono::high_resolution_clock::now();
            for(int q_id=0;q_id<countbased_queries.size();q_id++)
            rw_tree_obj.Projection(refined_blocks[q_id],countbased_queries[q_id],rw_tree_obj.root_);
            auto refinement_end = std::chrono::high_resolution_clock::now();
            
            double_t number_of_refined_blocks=0;
            double_t number_of_points_scanned=0;
            for(int q_id=0;q_id<countbased_queries.size();q_id++){
            number_of_refined_blocks+=refined_blocks[q_id].size();
            number_of_points_scanned+=rw_tree_obj.block_store_.NumOfPointsInBlocks(refined_blocks[q_id]);
            }


            log_json["refinement_latency"] = chrono::duration_cast<chrono::nanoseconds>(refinement_end - refinement_start).count()/countbased_queries.size();
            log_json["number_of_refined_blocks"] = number_of_refined_blocks/countbased_queries.size();
            log_json["number_blocks_in_blockstore"] = rw_tree_obj.block_store_.block_list_.size();
            log_json["number_of_points_scanned"] = number_of_points_scanned/countbased_queries.size();
            log_json["block_size_quantiles"] = rw_tree_obj.block_store_.QuantilesOfBlockSizes();

        }


        {
            rw_tree_obj.block_store_.use_memory_mapped_data=true;
            size_t disk_backed_result_size=0;
            auto eval_start = std::chrono::high_resolution_clock::now();
            for(auto &query: countbased_queries){
                result_vec = rw_tree_obj.RangeQuery(query);  
                disk_backed_result_size+=result_vec.size();
            }
            auto eval_end = std::chrono::high_resolution_clock::now();

            log_json["disk_backed_result_size"]=disk_backed_result_size;
            log_json["disk_backed_query_latency"] = chrono::duration_cast<chrono::nanoseconds>(eval_end - eval_start).count()/countbased_queries.size();
        }
        list_of_results.push_back(log_json);
    }
    cout<<dataset_folder_name<<" "<<data_sample_num<<" "<<data_ent_id<<" "<<BLOCK_SIZE<<" "<<query_ent_id<<" "<<selectivity<<" RW Finished"<<"\n";
    
    
    
    
    cout<<dataset_folder_name<<" "<<data_sample_num<<" "<<data_ent_id<<" "<<BLOCK_SIZE<<" "<<query_ent_id<<" "<<selectivity<<" RSMI Started"<<"\n";
   
    shuffle_vector(datapoints);
    shuffle_vector(countbased_queries);
    {    //########## RSMI-RTree-NoLocalModel #################
        // Training
        RTreeBASE rsmi_tree_obj(PROJECT_ROOT+"Experiments/"+dataset_folder_name+"/TrainedIndexes/RSMI/"+query_agnostic_tree_name+".tree");
        std::cout<<"Finished Loading RSMI-NoNN"<<"\n";

        ifstream build_time_file(PROJECT_ROOT+"Experiments/"+dataset_folder_name+"/TrainedIndexes/RSMI/"+query_agnostic_tree_name+".time",ios::in);
        double_t rsmi_tree_build_time;
        build_time_file>>rsmi_tree_build_time;
        build_time_file.close();


        log_json["model"]="RSMI";
        log_json["build_time"] = rsmi_tree_build_time;


        
        //################## Query Processing Count Based ##################
        log_json["area_or_count_based"]="count";
        {
            result_size=0;
            auto eval_start = std::chrono::high_resolution_clock::now();
            for(auto &query: countbased_queries){
                result_vec = rsmi_tree_obj.RangeQuery(query); 
                result_size+=result_vec.size();
            }
            auto eval_end = std::chrono::high_resolution_clock::now();

            log_json["result_size"]=result_size;
            log_json["query_latency"] = chrono::duration_cast<chrono::nanoseconds>(eval_end - eval_start).count()/countbased_queries.size();


            // Extra Query Processing metrics
            vector<vector<size_t>> refined_blocks(countbased_queries.size());

            auto refinement_start = std::chrono::high_resolution_clock::now();
            for(int q_id=0;q_id<countbased_queries.size();q_id++)
            rsmi_tree_obj.Projection(refined_blocks[q_id],countbased_queries[q_id],rsmi_tree_obj.root_);
            auto refinement_end = std::chrono::high_resolution_clock::now();
            
            double_t number_of_refined_blocks=0;
            double_t number_of_points_scanned=0;
            for(int q_id=0;q_id<countbased_queries.size();q_id++){
            number_of_refined_blocks+=refined_blocks[q_id].size();
            number_of_points_scanned+=rsmi_tree_obj.block_store_.NumOfPointsInBlocks(refined_blocks[q_id]);
            }


            log_json["refinement_latency"] = chrono::duration_cast<chrono::nanoseconds>(refinement_end - refinement_start).count()/countbased_queries.size();
            log_json["number_of_refined_blocks"] = number_of_refined_blocks/countbased_queries.size();
            log_json["number_blocks_in_blockstore"] = rsmi_tree_obj.block_store_.block_list_.size();
            log_json["number_of_points_scanned"] = number_of_points_scanned/countbased_queries.size();
            log_json["block_size_quantiles"] = rsmi_tree_obj.block_store_.QuantilesOfBlockSizes();

        }
        {
            rsmi_tree_obj.block_store_.use_memory_mapped_data=true;
            size_t disk_backed_result_size=0;
            auto eval_start = std::chrono::high_resolution_clock::now();
            for(auto &query: countbased_queries){
                result_vec = rsmi_tree_obj.RangeQuery(query);  
                disk_backed_result_size+=result_vec.size();
            }
            auto eval_end = std::chrono::high_resolution_clock::now();

            log_json["disk_backed_result_size"]=disk_backed_result_size;
            log_json["disk_backed_query_latency"] = chrono::duration_cast<chrono::nanoseconds>(eval_end - eval_start).count()/countbased_queries.size();
        }
        list_of_results.push_back(log_json);        
    }
    cout<<dataset_folder_name<<" "<<data_sample_num<<" "<<data_ent_id<<" "<<BLOCK_SIZE<<" "<<query_ent_id<<" "<<selectivity<<" RSMI Finished"<<"\n";
    
    
    
    
    cout<<dataset_folder_name<<" "<<data_sample_num<<" "<<data_ent_id<<" "<<BLOCK_SIZE<<" "<<query_ent_id<<" "<<selectivity<<" KDTREE Started"<<"\n";
    shuffle_vector(datapoints);
    shuffle_vector(countbased_queries);
    {   //############# KDTREE #################
        // Training
        auto train_start = std::chrono::high_resolution_clock::now();
        KDTree kd_tree_obj(datapoints);
        auto train_end = std::chrono::high_resolution_clock::now();


        double_t kd_tree_build_time = chrono::duration_cast<chrono::nanoseconds>(train_end - train_start).count()/1000000000.0;

        log_json["model"]="KD";
        log_json["build_time"] = kd_tree_build_time;


        //################## Query Processing Count Based ##################
        log_json["area_or_count_based"]="count";

        {
            result_size=0;
            auto eval_start = std::chrono::high_resolution_clock::now();
            for(auto &query: countbased_queries){
                result_vec = kd_tree_obj.RangeQuery(query); 
                result_size+=result_vec.size();
            }
            auto eval_end = std::chrono::high_resolution_clock::now();

            log_json["result_size"]=result_size;
            log_json["query_latency"] = chrono::duration_cast<chrono::nanoseconds>(eval_end - eval_start).count()/countbased_queries.size();


            // Extra Query Processing metrics
            vector<vector<size_t>> refined_blocks(countbased_queries.size());


            auto refinement_start = std::chrono::high_resolution_clock::now();
            for(int q_id=0;q_id<countbased_queries.size();q_id++)
                kd_tree_obj.Projection(refined_blocks[q_id],countbased_queries[q_id],kd_tree_obj.root_);
            auto refinement_end = std::chrono::high_resolution_clock::now();
            
            double_t number_of_refined_blocks=0;
            double_t number_of_points_scanned=0;
            for(int q_id=0;q_id<countbased_queries.size();q_id++){
                number_of_refined_blocks+=refined_blocks[q_id].size();
                number_of_points_scanned+=kd_tree_obj.block_store_.NumOfPointsInBlocks(refined_blocks[q_id]);
            }


            log_json["refinement_latency"] = chrono::duration_cast<chrono::nanoseconds>(refinement_end - refinement_start).count()/countbased_queries.size();
            log_json["number_of_refined_blocks"] = number_of_refined_blocks/countbased_queries.size();
            log_json["number_blocks_in_blockstore"] = kd_tree_obj.block_store_.block_list_.size();
            log_json["number_of_points_scanned"] = number_of_points_scanned/countbased_queries.size();
            log_json["block_size_quantiles"] = kd_tree_obj.block_store_.QuantilesOfBlockSizes();

        }


        {
            kd_tree_obj.block_store_.use_memory_mapped_data=true;
            size_t disk_backed_result_size=0;
            auto eval_start = std::chrono::high_resolution_clock::now();
            for(auto &query: countbased_queries){
                result_vec = kd_tree_obj.RangeQuery(query);  
                disk_backed_result_size+=result_vec.size();
            }
            auto eval_end = std::chrono::high_resolution_clock::now();

            log_json["disk_backed_result_size"]=disk_backed_result_size;
            log_json["disk_backed_query_latency"] = chrono::duration_cast<chrono::nanoseconds>(eval_end - eval_start).count()/countbased_queries.size();
        }

        list_of_results.push_back(log_json);
    }
    cout<<dataset_folder_name<<" "<<data_sample_num<<" "<<data_ent_id<<" "<<BLOCK_SIZE<<" "<<query_ent_id<<" "<<selectivity<<" KDTREE Finished"<<"\n";
    


    cout<<dataset_folder_name<<" "<<data_sample_num<<" "<<data_ent_id<<" "<<BLOCK_SIZE<<" "<<query_ent_id<<" "<<selectivity<<" QDTREE Started"<<"\n";
   
    shuffle_vector(datapoints);
    shuffle_vector(countbased_queries);
    {    //############# QDTREE RandomSearchVersion #################

        // Training
        auto train_start = std::chrono::high_resolution_clock::now();
        QDTreeTrainerRandomSearch(datapoints,countbased_queries,PROJECT_ROOT+"Experiments/"+dataset_folder_name+"/TrainedIndexes/QDTree/"+tree_name+"_areabased.txt");
        auto train_end = std::chrono::high_resolution_clock::now();
        
        double_t qd_tree_build_time = chrono::duration_cast<chrono::nanoseconds>(train_end - train_start).count()/1000000000.0;

        
        QDTree qd_tree_obj(datapoints,PROJECT_ROOT+"Experiments/"+dataset_folder_name+"/TrainedIndexes/QDTree/"+tree_name+"_areabased.txt");



        log_json["model"]="QD";
        log_json["build_time"] = qd_tree_build_time;

        //################## Query Processing Area Based ##################
        log_json["area_or_count_based"]="count";

        result_size=0;
        auto eval_start = std::chrono::high_resolution_clock::now();
        for(auto &query: countbased_queries){
            result_vec = qd_tree_obj.RangeQuery(query); 
            result_size+=result_vec.size();
        }
        auto eval_end = std::chrono::high_resolution_clock::now();

        log_json["result_size"]=result_size;
        log_json["query_latency"] = chrono::duration_cast<chrono::nanoseconds>(eval_end - eval_start).count()/countbased_queries.size();

        cout<<"QDTREE::Finished query processing"<<"\n";

        { // Extra Query Processing metrics
            vector<vector<size_t>> refined_blocks(countbased_queries.size());


            auto refinement_start = std::chrono::high_resolution_clock::now();
            for(int q_id=0;q_id<countbased_queries.size();q_id++)
                qd_tree_obj.Projection(refined_blocks[q_id],countbased_queries[q_id],qd_tree_obj.root_);
            auto refinement_end = std::chrono::high_resolution_clock::now();
            
            double_t number_of_refined_blocks=0;
            double_t number_of_points_scanned=0;
            for(int q_id=0;q_id<countbased_queries.size();q_id++){
                number_of_refined_blocks+=refined_blocks[q_id].size();
                number_of_points_scanned+=qd_tree_obj.block_store_.NumOfPointsInBlocks(refined_blocks[q_id]);
            }


            log_json["refinement_latency"] = chrono::duration_cast<chrono::nanoseconds>(refinement_end - refinement_start).count()/countbased_queries.size();
            log_json["number_of_refined_blocks"] = number_of_refined_blocks/countbased_queries.size();
            log_json["number_blocks_in_blockstore"] = qd_tree_obj.block_store_.block_list_.size();
            log_json["number_of_points_scanned"] = number_of_points_scanned/countbased_queries.size();
            log_json["block_size_quantiles"] = qd_tree_obj.block_store_.QuantilesOfBlockSizes();

        }

        {
            qd_tree_obj.block_store_.use_memory_mapped_data=true;
            size_t disk_backed_result_size=0;
            auto eval_start = std::chrono::high_resolution_clock::now();
            for(auto &query: countbased_queries){
                result_vec = qd_tree_obj.RangeQuery(query);  
                disk_backed_result_size+=result_vec.size();
            }
            auto eval_end = std::chrono::high_resolution_clock::now();

            log_json["disk_backed_result_size"]=disk_backed_result_size;
            log_json["disk_backed_query_latency"] = chrono::duration_cast<chrono::nanoseconds>(eval_end - eval_start).count()/countbased_queries.size();
        }
        list_of_results.push_back(log_json);
    }

    cout<<dataset_folder_name<<" "<<data_sample_num<<" "<<data_ent_id<<" "<<BLOCK_SIZE<<" "<<query_ent_id<<" "<<selectivity<<" QDTREE Random Search Finished"<<"\n";



    filesystem::create_directories(PROJECT_ROOT+"Experiments/"+dataset_folder_name+"/ResultsFolder_ExtendBlockSize");
    ofstream result_file(PROJECT_ROOT+"Experiments/"+dataset_folder_name+"/ResultsFolder_ExtendBlockSize/"+line_num,ios_base::app);
    // ofstream result_file(PROJECT_ROOT+"Experiments/"+dataset_folder_name+"/Results.json",ios_base::app);
    for(auto& result_json: list_of_results)
        result_file<<result_json<<"\n";
    result_file.close();
    return 0;
}
