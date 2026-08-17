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
#include<utility>

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

filesystem::path configured_output_dir(const string& dataset_folder_name) {
    const char* env_path = getenv("EXPERIMENT_OUTPUT_DIR");
    if(env_path != nullptr && string(env_path).size() > 0) {
        return filesystem::path(env_path);
    }
    return filesystem::path(PROJECT_ROOT) / "Experiments" / dataset_folder_name / "ResultsFolder";
}

// ===================== Storage-backend measurement passes =====================
//
// Every index section used to carry its own copy of the disk-backed timing loop.
// They now share RunStoragePasses() below; see phase4-evaluator-integration.md.

/** Frame budget and pass settings, read once from the environment. */
struct StoragePassConfig{
    vector<double> fractions{1.0, 0.25, 0.05, 0.01, 0.001};
    BufferPoolFloorMode floor_mode{BufferPoolFloorMode::kBlock};
    string policy{"LRU"};
    bool release_blocks{true};
    bool verify{false};
};

static bool EnvFlag(const char* name, bool fallback){
    const char* raw = getenv(name);
    if(raw == nullptr || string(raw).empty()) return fallback;
    return string(raw) == "1" || string(raw) == "true";
}

static StoragePassConfig LoadStoragePassConfig(){
    StoragePassConfig config;

    const char* fractions = getenv("BUFFER_POOL_FRACTIONS");
    if(fractions != nullptr && string(fractions).size() > 0){
        config.fractions.clear();
        stringstream stream(fractions);
        string field;
        while(getline(stream, field, ','))
            if(!field.empty()) config.fractions.push_back(stod(field));
        if(config.fractions.empty())
            throw runtime_error("BUFFER_POOL_FRACTIONS is set but parsed to nothing");
    }

    const char* floor_mode = getenv("BUFFER_POOL_FLOOR_MODE");
    if(floor_mode != nullptr && string(floor_mode) == "minimal")
        config.floor_mode = BufferPoolFloorMode::kMinimal;

    const char* policy = getenv("BUFFER_POOL_POLICY");
    if(policy != nullptr && string(policy).size() > 0) config.policy = string(policy);

    config.release_blocks = EnvFlag("BUFFER_POOL_RELEASE_BLOCKS", true);
    config.verify         = EnvFlag("VERIFY_BACKENDS", false);
    return config;
}

/** FNV-1a over the raw coordinate bytes: catches reordering, not just size drift. */
static uint64_t FingerprintPoints(uint64_t hash, const vector<Point>& points){
    for(const Point& point: points){
        const unsigned char* bytes = reinterpret_cast<const unsigned char*>(point.elements_);
        for(size_t i=0;i<sizeof(double_t)*Constants::DIM;i++){
            hash ^= bytes[i];
            hash *= 1099511628211ULL;
        }
    }
    return hash;
}

/** Copy the keys that let a bp_ row be joined back to its results row. */
static void CopyJoinKeys(const json& log_json, json& row){
    for(const char* key: {"model","block_size","data_sample_num","dataset_entropy_id",
                          "query_entropy_id","selectivity","area_or_count_based","result_size"})
        if(log_json.contains(key)) row[key] = log_json[key];
}

/**
 * Run every disk-backed measurement for one index.
 *
 * `run_one(i, out)` executes query i and leaves its results in `out`. It takes the
 * output vector by reference rather than returning it so the timed loop keeps the
 * copy-assign the original code had -- returning would elide into a move and shift
 * the latency baseline away from previously collected numbers.
 *
 * Writes the existing disk_backed_* keys into `log_json`, and one row per budget
 * fraction into `bp_rows`.
 */
template <typename RunOne>
void RunStoragePasses(BlockStore& store, size_t query_count, RunOne&& run_one,
                      json& log_json, vector<json>& bp_rows, const StoragePassConfig& config){
    vector<Point> result_vec;

    // ---- 1. mmap pass: the pre-existing disk_backed_* keys, unchanged ----
    store.SetStorageMode(StorageMode::kMmap);
    size_t disk_backed_result_size = 0;
    auto eval_start = chrono::high_resolution_clock::now();
    for(size_t i=0;i<query_count;i++){
        run_one(i, result_vec);
        disk_backed_result_size += result_vec.size();
    }
    auto eval_end = chrono::high_resolution_clock::now();

    log_json["disk_backed_result_size"] = disk_backed_result_size;
    log_json["disk_backed_query_latency"] =
        chrono::duration_cast<chrono::nanoseconds>(eval_end - eval_start).count()/query_count;

    PagedDiskBackend* paged = store.PagedBackendPtr();

    // ---- 2. cross-backend verification, untimed and opt-in ----
    // Must run before the release below, since it compares against the in-memory scan.
    bool results_match = true;
    if(config.verify){
        const uint64_t kFnvOffset = 1469598103934665603ULL;
        uint64_t fingerprints[3] = {kFnvOffset, kFnvOffset, kFnvOffset};
        const StorageMode modes[3] = {StorageMode::kInMemory, StorageMode::kMmap, StorageMode::kBufferPool};

        for(int m=0;m<(paged ? 3 : 2);m++){
            store.SetStorageMode(modes[m]);
            for(size_t i=0;i<query_count;i++){
                run_one(i, result_vec);
                fingerprints[m] = FingerprintPoints(fingerprints[m], result_vec);
            }
        }
        results_match = (fingerprints[0] == fingerprints[1]) &&
                        (!paged || fingerprints[0] == fingerprints[2]);
        if(!results_match)
            cerr<<"BACKEND MISMATCH for "<<log_json.value("model","?")
                <<": in-memory="<<fingerprints[0]<<" mmap="<<fingerprints[1]
                <<" paged="<<(paged ? to_string(fingerprints[2]) : string("n/a"))<<endl;
        log_json["results_match"] = results_match;
    }

    if(!paged)
        return;

    // ---- 3. free the in-memory copy so the budget means something ----
    store.SetStorageMode(StorageMode::kBufferPool);
    if(config.release_blocks)
        store.ReleaseInMemoryBlocks();

    // ---- 4. one cold + one warm pass per budget fraction ----
    for(double fraction: config.fractions){
        BufferPoolConfig pool_config;
        pool_config.fraction   = fraction;
        pool_config.floor_mode = config.floor_mode;
        pool_config.policy     = config.policy;
        paged->RebuildPool(pool_config);

        paged->ClearCache();
        paged->ResetStats();
        auto cold_start = chrono::high_resolution_clock::now();
        for(size_t i=0;i<query_count;i++) run_one(i, result_vec);
        auto cold_end = chrono::high_resolution_clock::now();
        const StorageStats cold = paged->Stats();

        paged->ResetStats();
        size_t bp_result_size = 0;
        auto warm_start = chrono::high_resolution_clock::now();
        for(size_t i=0;i<query_count;i++){
            run_one(i, result_vec);
            bp_result_size += result_vec.size();
        }
        auto warm_end = chrono::high_resolution_clock::now();
        const StorageStats warm = paged->Stats();

        json row;
        CopyJoinKeys(log_json, row);

        row["storage_page_bytes"]        = paged->Geometry().page_bytes_;
        row["storage_record_bytes"]      = paged->Geometry().record_bytes_;
        row["storage_records_per_page"]  = paged->Geometry().records_per_page_;

        row["bufferpool_fraction"]           = fraction;
        row["bufferpool_frames"]             = paged->PoolFrames();
        row["bufferpool_bytes"]              = paged->PoolBytes();
        row["bufferpool_effective_fraction"] = paged->EffectiveFraction();
        row["bufferpool_frames_floored"]     = paged->FramesFloored();
        row["bufferpool_floor_mode"]         = (config.floor_mode == BufferPoolFloorMode::kBlock) ? "block" : "minimal";
        row["bp_replacement_policy"]         = paged->PolicyName();
        row["largest_block_pages"]           = paged->LargestBlockPages();

        row["index_file_bytes"]            = paged->FileBytes();
        row["index_total_pages"]           = paged->TotalDataPages()+1;
        row["index_directory_bytes"]       = paged->DirectoryBytes();
        row["index_metadata_bytes"]        = store.MetadataBytes();
        row["number_blocks_in_blockstore"] = store.NumOfBlocks();
        row["blocks_released"]             = store.BlocksReleased();

        row["bp_result_size"] = bp_result_size;
        row["results_match"]  = results_match;

        row["bp_cold_query_latency"]   = chrono::duration_cast<chrono::nanoseconds>(cold_end-cold_start).count()/query_count;
        row["bp_cold_pages_requested"] = cold.pages_requested;
        row["bp_cold_page_misses"]     = cold.page_misses;
        row["bp_cold_hit_rate"]        = cold.HitRate();
        row["bp_cold_bytes_read"]      = cold.bytes_read;

        row["bp_warm_query_latency"]   = chrono::duration_cast<chrono::nanoseconds>(warm_end-warm_start).count()/query_count;
        row["bp_warm_pages_requested"] = warm.pages_requested;
        row["bp_warm_page_misses"]     = warm.page_misses;
        row["bp_warm_hit_rate"]        = warm.HitRate();
        row["bp_warm_bytes_read"]      = warm.bytes_read;
        row["bp_warm_evictions"]       = warm.evictions;
        row["bp_warm_blocks_scanned"]  = warm.blocks_scanned;
        row["bp_warm_points_decoded"]  = warm.points_decoded;

        row["bp_pages_requested_per_query"] = double(warm.pages_requested)/double(query_count);
        row["bp_page_misses_per_query"]     = double(warm.page_misses)/double(query_count);
        row["bp_blocks_scanned_per_query"]  = double(warm.blocks_scanned)/double(query_count);

        bp_rows.push_back(row);
    }
}
// ==============================================================================


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
void shuffle_vector(vector<T>& values, mt19937& rng) {
    shuffle(values.begin(), values.end(), rng);
}

int main(int argc, char* argv[]){
    cout << unitbuf;
    cerr << unitbuf;

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
    const char* env_result_file = getenv("EXPERIMENT_RESULT_FILE");
    if(env_result_file != nullptr && string(env_result_file).size() > 0) {
        line_num = string(env_result_file);
    }


    cout<<dataset_folder_name<<" "<<data_sample_num<<" "<<data_ent_id<<" "<<BLOCK_SIZE<<" "<<query_ent_id<<" "<<selectivity<<" "<<PROJECT_ROOT<<endl;


    /*Reading the dataset and the entropy values*/
    vector<Point> datapoints;
    double_t a, b, c, d;
    filesystem::path dataset_root =
        filesystem::path(PROJECT_ROOT) / "Datasets" / dataset_folder_name / to_string(data_sample_num);
    filesystem::path points_path =
        dataset_root / "datapoints" / to_string(data_ent_id);
    ifstream pointsfile(points_path,ios::in);
    if(!pointsfile.is_open()){
        cerr<<"Unable to open datapoints file: "<<points_path<<endl;
        return 1;
    }
    while (pointsfile >> a >> b)
        datapoints.push_back(Point(a,b));
    pointsfile.close();
    cout<<"Finished loading points |D|:"<<datapoints.size()<<endl;
    if(datapoints.empty()){
        cerr<<"Datapoints file is empty: "<<points_path<<endl;
        return 1;
    }

    vector<double_t> data_entropy;
    filesystem::path data_entropy_path = dataset_root / "datapoints" / "entropy_values";
    ifstream data_entropy_file(data_entropy_path,ios::in);
    if(!data_entropy_file.is_open()){
        cerr<<"Unable to open data entropy file: "<<data_entropy_path<<endl;
        return 1;
    }
    while (data_entropy_file >> a >> b)
        data_entropy.push_back(b);
    data_entropy_file.close();
    if(data_entropy.empty()){
        cerr<<"Data entropy file is empty: "<<data_entropy_path<<endl;
        return 1;
    }

    vector<Query> countbased_queries;
    filesystem::path countbased_queries_path =
        dataset_root / "queries" / "otherDist" /
        (to_string(data_ent_id)+"_"+selectivity+"_countbased_"+to_string(query_ent_id));
    ifstream countbased_queriesfile(countbased_queries_path,ios::in);
    if(!countbased_queriesfile.is_open()){
        cerr<<"Unable to open count-based query file: "<<countbased_queries_path<<endl;
        return 1;
    }
    while (countbased_queriesfile >> a >> b >> c >> d)
        countbased_queries.push_back(Query(Point(a,b),Point(c,d)));
    countbased_queriesfile.close();
    cout<<"Finished loading countbased_queries |Q|:"<<countbased_queries.size()<<endl;
    if(countbased_queries.empty()){
        cerr<<"Count-based query file is empty: "<<countbased_queries_path<<endl;
        return 1;
    }

    mt19937 datapoint_shuffle_rng(1337);
    mt19937 query_shuffle_rng(7331);
    shuffle_vector(datapoints, datapoint_shuffle_rng);
    shuffle_vector(countbased_queries, query_shuffle_rng);

    struct QueryEntropyRow {
        int data_entropy_id;
        int query_entropy_id;
        double_t entropy;
    };
    vector<QueryEntropyRow> query_entropy_rows;
    filesystem::path query_entropy_path = dataset_root / "queries" / "entropy_values";
    ifstream query_entropy_file(query_entropy_path,ios::in);
    if(!query_entropy_file.is_open()){
        cerr<<"Unable to open query entropy file: "<<query_entropy_path<<endl;
        return 1;
    }
    while (query_entropy_file >> a >> b >>c)
        query_entropy_rows.push_back(
            QueryEntropyRow{
                static_cast<int>(llround(a)),
                static_cast<int>(llround(b)),
                c
            }
        );
    query_entropy_file.close();
    if(query_entropy_rows.empty()){
        cerr<<"Query entropy file is empty: "<<query_entropy_path<<endl;
        return 1;
    }


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
    bool query_entropy_found = false;
    double_t query_entropy_value = 0.0;
    for(const auto& row : query_entropy_rows){
        if(row.data_entropy_id == data_ent_id && row.query_entropy_id == query_ent_id){
            query_entropy_value = row.entropy;
            query_entropy_found = true;
            break;
        }
    }
    if(!query_entropy_found){
        cerr<<"Missing query entropy row for data_ent_id "<<data_ent_id
            <<" and query_ent_id "<<query_ent_id
            <<" in "<<query_entropy_path<<endl;
        return 1;
    }
    log_json["query_entropy"] = query_entropy_value;
    log_json["selectivity"] = selectivity;
    
    std::vector<json> list_of_results;

    // Buffer-pool rows go to their own file. They vary along bufferpool_fraction,
    // which is not part of the grouping key prepare_results() uses in
    // Results/plot_results.py -- putting them in the main results file would make
    // it silently average five fractions into one row and shift every figure.
    std::vector<json> bp_list_of_results;
    const StoragePassConfig storage_pass_config = LoadStoragePassConfig();

    cout<<dataset_folder_name<<" "<<data_sample_num<<" "<<data_ent_id<<" "<<BLOCK_SIZE<<" "<<query_ent_id<<" "<<selectivity<<" WAZI Started"<<endl;

    {   //############# WAZI #################
        // Training
        vector<Point> model_datapoints = datapoints;
        vector<Query> model_queries = countbased_queries;
        auto train_start = std::chrono::high_resolution_clock::now();
        SamplZTree wazi_obj(model_datapoints,model_queries);
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
            log_json["number_blocks_in_blockstore"] = wazi_obj.block_store_.NumOfBlocks();
            log_json["number_of_points_scanned"] = number_of_points_scanned/countbased_queries.size();
            log_json["block_size_quantiles"] = wazi_obj.block_store_.QuantilesOfBlockSizes();

        }

        RunStoragePasses(wazi_obj.block_store_, countbased_queries.size(),
            [&](size_t q_idx, vector<Point>& out){ out = wazi_obj.RangeQuery(countbased_queries[q_idx]); },
            log_json, bp_list_of_results, storage_pass_config);
        list_of_results.push_back(log_json);
    }

    cout<<dataset_folder_name<<" "<<data_sample_num<<" "<<data_ent_id<<" "<<BLOCK_SIZE<<" "<<query_ent_id<<" "<<selectivity<<" WAZI Finished"<<endl;




    cout<<dataset_folder_name<<" "<<data_sample_num<<" "<<data_ent_id<<" "<<BLOCK_SIZE<<" "<<query_ent_id<<" "<<selectivity<<" ZIndexStarted"<<endl;
    {   //############# ZIndex #################
        // Training
        vector<Point> model_datapoints = datapoints;
        auto train_start = std::chrono::high_resolution_clock::now();
        BaseZTree zindex_obj(model_datapoints);
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
            log_json["number_blocks_in_blockstore"] = zindex_obj.block_store_.NumOfBlocks();
            log_json["number_of_points_scanned"] = number_of_points_scanned/countbased_queries.size();
            log_json["block_size_quantiles"] = zindex_obj.block_store_.QuantilesOfBlockSizes();

        }

        RunStoragePasses(zindex_obj.block_store_, countbased_queries.size(),
            [&](size_t q_idx, vector<Point>& out){ out = zindex_obj.RangeQuery(countbased_queries[q_idx]); },
            log_json, bp_list_of_results, storage_pass_config);

        list_of_results.push_back(log_json);
    }
    cout<<dataset_folder_name<<" "<<data_sample_num<<" "<<data_ent_id<<" "<<BLOCK_SIZE<<" "<<query_ent_id<<" "<<selectivity<<" ZIndex Finished"<<endl;



    cout<<dataset_folder_name<<" "<<data_sample_num<<" "<<data_ent_id<<" "<<BLOCK_SIZE<<" "<<query_ent_id<<" "<<selectivity<<" ZM-Index Started"<<endl;
    {   //############# ZM-Index #################
        // Training
        vector<Point> model_datapoints = datapoints;
        auto train_start = std::chrono::high_resolution_clock::now();
        ZMIndex zmindex_obj(std::move(model_datapoints));
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
            log_json["number_blocks_in_blockstore"] = zmindex_obj.block_store_.NumOfBlocks();
            log_json["number_of_points_scanned"] = number_of_points_scanned/countbased_queries.size();
            log_json["block_size_quantiles"] = zmindex_obj.block_store_.QuantilesOfBlockSizes();

            std::cout<<"Starting Disk based queries"<<std::endl;

            RunStoragePasses(zmindex_obj.block_store_, countbased_queries.size(),
                [&](size_t q_idx, vector<Point>& out){ out = zmindex_obj.RangeQuery(countbased_queries[q_idx],query_lows[q_idx],query_highs[q_idx]); },
                log_json, bp_list_of_results, storage_pass_config);
        }

        list_of_results.push_back(log_json);
    }
    cout<<dataset_folder_name<<" "<<data_sample_num<<" "<<data_ent_id<<" "<<BLOCK_SIZE<<" "<<query_ent_id<<" "<<selectivity<<" ZM-Index Finished"<<endl;
    
    
    cout<<dataset_folder_name<<" "<<data_sample_num<<" "<<data_ent_id<<" "<<BLOCK_SIZE<<" "<<query_ent_id<<" "<<selectivity<<" GRID Started"<<endl;
    {   //############# GRID #################
        // Training
        vector<Point> model_datapoints = datapoints;
        auto train_start = std::chrono::high_resolution_clock::now();
        int num_splits_uniform_grid = int(sqrt(datapoints.size()/BLOCK_SIZE));
        std::array<int, 2> split_per_dim{num_splits_uniform_grid, num_splits_uniform_grid};
        std::array<int, 2> dim_order{0, 1};
        FloodIndex unigrid_obj( dim_order,split_per_dim);
        unigrid_obj.LoadElements(model_datapoints);
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
            log_json["number_blocks_in_blockstore"] = unigrid_obj.block_store_.NumOfBlocks();
            log_json["number_of_points_scanned"] = number_of_points_scanned/countbased_queries.size();
            log_json["block_size_quantiles"] = unigrid_obj.block_store_.QuantilesOfBlockSizes();

        }

        RunStoragePasses(unigrid_obj.block_store_, countbased_queries.size(),
            [&](size_t q_idx, vector<Point>& out){ out = unigrid_obj.RangeQuery(countbased_queries[q_idx]); },
            log_json, bp_list_of_results, storage_pass_config);
        list_of_results.push_back(log_json);
    }
    cout<<dataset_folder_name<<" "<<data_sample_num<<" "<<data_ent_id<<" "<<BLOCK_SIZE<<" "<<query_ent_id<<" "<<selectivity<<" GRID Finished"<<endl;
    
    
    
    
    cout<<dataset_folder_name<<" "<<data_sample_num<<" "<<data_ent_id<<" "<<BLOCK_SIZE<<" "<<query_ent_id<<" "<<selectivity<<" FLOOD Started"<<endl;
    {   //############# FLOOD #################
        // Training
        vector<Point> trainer_datapoints = datapoints;
        vector<Point> model_datapoints = datapoints;
        vector<Query> model_queries = countbased_queries;
        auto train_start = std::chrono::high_resolution_clock::now();
        auto flood_config = FloodTrainerRandomSearch(trainer_datapoints,model_queries);
        FloodIndex flood_obj(flood_config.first,flood_config.second);
        flood_obj.LoadElements(model_datapoints);
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
            log_json["number_blocks_in_blockstore"] = flood_obj.block_store_.NumOfBlocks();
            log_json["number_of_points_scanned"] = number_of_points_scanned/countbased_queries.size();
            log_json["block_size_quantiles"] = flood_obj.block_store_.QuantilesOfBlockSizes();

        }

        RunStoragePasses(flood_obj.block_store_, countbased_queries.size(),
            [&](size_t q_idx, vector<Point>& out){ out = flood_obj.RangeQuery(countbased_queries[q_idx]); },
            log_json, bp_list_of_results, storage_pass_config);

        list_of_results.push_back(log_json);
    }
    cout<<dataset_folder_name<<" "<<data_sample_num<<" "<<data_ent_id<<" "<<BLOCK_SIZE<<" "<<query_ent_id<<" "<<selectivity<<" FLOOD Finished"<<endl;
    
    
    
    cout<<dataset_folder_name<<" "<<data_sample_num<<" "<<data_ent_id<<" "<<BLOCK_SIZE<<" "<<query_ent_id<<" "<<selectivity<<" STR Started"<<endl;
    {   /* ##########################    STR   ######################################*/
        // Training
        vector<Point> model_datapoints = datapoints;
        auto train_start = std::chrono::high_resolution_clock::now();
        STRTree str_tree_obj(std::move(model_datapoints));
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
            log_json["number_blocks_in_blockstore"] = str_tree_obj.block_store_.NumOfBlocks();
            log_json["number_of_points_scanned"] = number_of_points_scanned/countbased_queries.size();
            log_json["block_size_quantiles"] = str_tree_obj.block_store_.QuantilesOfBlockSizes();


            RunStoragePasses(str_tree_obj.block_store_, countbased_queries.size(),
                [&](size_t q_idx, vector<Point>& out){ out = str_tree_obj.RangeQuery(countbased_queries[q_idx]); },
                log_json, bp_list_of_results, storage_pass_config);


            list_of_results.push_back(log_json);
        }

    }
    cout<<dataset_folder_name<<" "<<data_sample_num<<" "<<data_ent_id<<" "<<BLOCK_SIZE<<" "<<query_ent_id<<" "<<selectivity<<" STR Finished"<<endl;
    
    
    
    cout<<dataset_folder_name<<" "<<data_sample_num<<" "<<data_ent_id<<" "<<BLOCK_SIZE<<" "<<query_ent_id<<" "<<selectivity<<" RSTAR Started"<<endl;

    {   //########## RSTAR #################
        // Training
        vector<Point> model_datapoints = datapoints;
        auto train_start = std::chrono::high_resolution_clock::now();
        RSTARTree rstar_tree_obj(std::move(model_datapoints));
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
            log_json["number_blocks_in_blockstore"] = rstar_tree_obj.block_store_.NumOfBlocks();
            log_json["number_of_points_scanned"] = number_of_points_scanned/countbased_queries.size();
            log_json["block_size_quantiles"] = rstar_tree_obj.block_store_.QuantilesOfBlockSizes();

        }

        RunStoragePasses(rstar_tree_obj.block_store_, countbased_queries.size(),
            [&](size_t q_idx, vector<Point>& out){ out = rstar_tree_obj.RangeQuery(countbased_queries[q_idx]); },
            log_json, bp_list_of_results, storage_pass_config);

        list_of_results.push_back(log_json);
    }
    cout<<dataset_folder_name<<" "<<data_sample_num<<" "<<data_ent_id<<" "<<BLOCK_SIZE<<" "<<query_ent_id<<" "<<selectivity<<" RSTAR Finished"<<endl;
    
    
    
    cout<<dataset_folder_name<<" "<<data_sample_num<<" "<<data_ent_id<<" "<<BLOCK_SIZE<<" "<<query_ent_id<<" "<<selectivity<<" CUR Started"<<endl;

    {   //########## CUR #################
        // Training
        vector<Point> model_datapoints = datapoints;
        vector<Query> model_queries = countbased_queries;
        auto train_start = std::chrono::high_resolution_clock::now();
        CURTree cur_tree_obj(std::move(model_datapoints),std::move(model_queries));
        cout<<" CUR Finished Building"<<endl;
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
            log_json["number_blocks_in_blockstore"] = cur_tree_obj.block_store_.NumOfBlocks();
            log_json["number_of_points_scanned"] = number_of_points_scanned/countbased_queries.size();
            log_json["block_size_quantiles"] = cur_tree_obj.block_store_.QuantilesOfBlockSizes();

        }


        RunStoragePasses(cur_tree_obj.block_store_, countbased_queries.size(),
            [&](size_t q_idx, vector<Point>& out){ out = cur_tree_obj.RangeQuery(countbased_queries[q_idx]); },
            log_json, bp_list_of_results, storage_pass_config);

        list_of_results.push_back(log_json);
    }
    cout<<dataset_folder_name<<" "<<data_sample_num<<" "<<data_ent_id<<" "<<BLOCK_SIZE<<" "<<query_ent_id<<" "<<selectivity<<" CUR Finished"<<endl;
    
    
    
    cout<<dataset_folder_name<<" "<<data_sample_num<<" "<<data_ent_id<<" "<<BLOCK_SIZE<<" "<<query_ent_id<<" "<<selectivity<<" RW Started"<<endl;

    {   //########## RW #################
        // Training
        vector<Point> model_datapoints = datapoints;
        vector<Query> model_queries = countbased_queries;
        auto train_start = std::chrono::high_resolution_clock::now();
        cout<<" RW Build Started"<<endl;
        RWTree rw_tree_obj(std::move(model_datapoints),std::move(model_queries));
        cout<<" RW Build Finished"<<endl;
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
            log_json["number_blocks_in_blockstore"] = rw_tree_obj.block_store_.NumOfBlocks();
            log_json["number_of_points_scanned"] = number_of_points_scanned/countbased_queries.size();
            log_json["block_size_quantiles"] = rw_tree_obj.block_store_.QuantilesOfBlockSizes();

        }


        RunStoragePasses(rw_tree_obj.block_store_, countbased_queries.size(),
            [&](size_t q_idx, vector<Point>& out){ out = rw_tree_obj.RangeQuery(countbased_queries[q_idx]); },
            log_json, bp_list_of_results, storage_pass_config);
        list_of_results.push_back(log_json);
    }
    cout<<dataset_folder_name<<" "<<data_sample_num<<" "<<data_ent_id<<" "<<BLOCK_SIZE<<" "<<query_ent_id<<" "<<selectivity<<" RW Finished"<<endl;
    
    
    
    
    cout<<dataset_folder_name<<" "<<data_sample_num<<" "<<data_ent_id<<" "<<BLOCK_SIZE<<" "<<query_ent_id<<" "<<selectivity<<" RSMI Started"<<endl;
    {    //########## RSMI-RTree-NoLocalModel #################
        // Training
        RTreeBASE rsmi_tree_obj(PROJECT_ROOT+"Experiments/"+dataset_folder_name+"/TrainedIndexes/RSMI/"+query_agnostic_tree_name+".tree");
        std::cout<<"Finished Loading RSMI-NoNN"<<std::endl;

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
            log_json["number_blocks_in_blockstore"] = rsmi_tree_obj.block_store_.NumOfBlocks();
            log_json["number_of_points_scanned"] = number_of_points_scanned/countbased_queries.size();
            log_json["block_size_quantiles"] = rsmi_tree_obj.block_store_.QuantilesOfBlockSizes();

        }
        RunStoragePasses(rsmi_tree_obj.block_store_, countbased_queries.size(),
            [&](size_t q_idx, vector<Point>& out){ out = rsmi_tree_obj.RangeQuery(countbased_queries[q_idx]); },
            log_json, bp_list_of_results, storage_pass_config);
        list_of_results.push_back(log_json);        
    }
    cout<<dataset_folder_name<<" "<<data_sample_num<<" "<<data_ent_id<<" "<<BLOCK_SIZE<<" "<<query_ent_id<<" "<<selectivity<<" RSMI Finished"<<endl;
    
    
    
    
    cout<<dataset_folder_name<<" "<<data_sample_num<<" "<<data_ent_id<<" "<<BLOCK_SIZE<<" "<<query_ent_id<<" "<<selectivity<<" KDTREE Started"<<endl;
    {   //############# KDTREE #################
        // Training
        vector<Point> model_datapoints = datapoints;
        auto train_start = std::chrono::high_resolution_clock::now();
        KDTree kd_tree_obj(std::move(model_datapoints));
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
            log_json["number_blocks_in_blockstore"] = kd_tree_obj.block_store_.NumOfBlocks();
            log_json["number_of_points_scanned"] = number_of_points_scanned/countbased_queries.size();
            log_json["block_size_quantiles"] = kd_tree_obj.block_store_.QuantilesOfBlockSizes();

        }


        RunStoragePasses(kd_tree_obj.block_store_, countbased_queries.size(),
            [&](size_t q_idx, vector<Point>& out){ out = kd_tree_obj.RangeQuery(countbased_queries[q_idx]); },
            log_json, bp_list_of_results, storage_pass_config);

        list_of_results.push_back(log_json);
    }
    cout<<dataset_folder_name<<" "<<data_sample_num<<" "<<data_ent_id<<" "<<BLOCK_SIZE<<" "<<query_ent_id<<" "<<selectivity<<" KDTREE Finished"<<endl;
    


    cout<<dataset_folder_name<<" "<<data_sample_num<<" "<<data_ent_id<<" "<<BLOCK_SIZE<<" "<<query_ent_id<<" "<<selectivity<<" QDTREE Started"<<endl;
    {    //############# QDTREE RandomSearchVersion #################

        // Training
        vector<Point> trainer_datapoints = datapoints;
        vector<Point> model_datapoints = datapoints;
        vector<Query> model_queries = countbased_queries;
        auto train_start = std::chrono::high_resolution_clock::now();
        QDTreeTrainerRandomSearch(std::move(trainer_datapoints),model_queries,PROJECT_ROOT+"Experiments/"+dataset_folder_name+"/TrainedIndexes/QDTree/"+tree_name+"_areabased.txt");
        auto train_end = std::chrono::high_resolution_clock::now();
        
        double_t qd_tree_build_time = chrono::duration_cast<chrono::nanoseconds>(train_end - train_start).count()/1000000000.0;

        
        QDTree qd_tree_obj(std::move(model_datapoints),PROJECT_ROOT+"Experiments/"+dataset_folder_name+"/TrainedIndexes/QDTree/"+tree_name+"_areabased.txt");



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

        cout<<"QDTREE::Finished query processing"<<endl;

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
            log_json["number_blocks_in_blockstore"] = qd_tree_obj.block_store_.NumOfBlocks();
            log_json["number_of_points_scanned"] = number_of_points_scanned/countbased_queries.size();
            log_json["block_size_quantiles"] = qd_tree_obj.block_store_.QuantilesOfBlockSizes();

        }

        RunStoragePasses(qd_tree_obj.block_store_, countbased_queries.size(),
            [&](size_t q_idx, vector<Point>& out){ out = qd_tree_obj.RangeQuery(countbased_queries[q_idx]); },
            log_json, bp_list_of_results, storage_pass_config);
        list_of_results.push_back(log_json);
    }

    cout<<dataset_folder_name<<" "<<data_sample_num<<" "<<data_ent_id<<" "<<BLOCK_SIZE<<" "<<query_ent_id<<" "<<selectivity<<" QDTREE Random Search Finished"<<endl;



    filesystem::path result_dir = configured_output_dir(dataset_folder_name);
    filesystem::create_directories(result_dir);
    ofstream result_file((result_dir / line_num).string(),ios_base::app);
    // ofstream result_file(PROJECT_ROOT+"Experiments/"+dataset_folder_name+"/Results.json",ios_base::app);
    for(auto& result_json: list_of_results)
        result_file<<result_json<<"\n";
    result_file.close();

    // Separate file, separate schema: <line>.jsonl keeps exactly the keys it always
    // had, so the existing plotting path and archived comparisons are untouched.
    if(!bp_list_of_results.empty()){
        ofstream bp_result_file((result_dir / ("bp_" + line_num)).string(), ios_base::app);
        for(auto& bp_json: bp_list_of_results)
            bp_result_file<<bp_json<<"\n";
        bp_result_file.close();
        cout<<"Wrote "<<bp_list_of_results.size()<<" buffer-pool rows to bp_"<<line_num<<endl;
    }
    return 0;
}
