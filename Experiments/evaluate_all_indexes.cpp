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
#include"../Indexes/utils/device_probe.h"
#include"../Indexes/utils/build_profile.h"



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

static bool EnvFlag(const char* name, bool fallback);   // defined with the storage-pass config below

/**
 * The query workload an index is TRAINED on -- normally the real one.
 *
 * With NULL_WORKLOAD=1 each query keeps its exact width and height but is moved
 * to a uniformly random position inside the data extent. Box count, box sizes and
 * therefore the selectivity distribution are all preserved; the only thing
 * destroyed is the spatial correlation between queries and data, and between one
 * query and the next.
 *
 * That is the ablation R4's question needs. The phase timers say how many seconds
 * an index spends in workload-aware code; they cannot say how many of those
 * seconds are *attributable* to the workload carrying signal, because the same
 * code runs either way. Differencing a real-workload build against a
 * null-workload one answers that, and it does so without depending on where the
 * phase boundaries were drawn.
 *
 * The MEASURED workload is untouched -- only training input changes -- so query
 * latency in an ablation run is the cost of having trained on noise, which is a
 * second, separately interesting number.
 */
static vector<Query> TrainingWorkload(const vector<Query>& real_queries,
                                      const vector<Point>& datapoints){
    if(!EnvFlag("NULL_WORKLOAD", false) || real_queries.empty() || datapoints.empty())
        return real_queries;

    double lo[Constants::DIM], hi[Constants::DIM];
    for(size_t d=0;d<Constants::DIM;d++){ lo[d]=datapoints[0].elements_[d]; hi[d]=lo[d]; }
    for(const Point& p: datapoints)
        for(size_t d=0;d<Constants::DIM;d++){
            lo[d]=min(lo[d],p.elements_[d]);
            hi[d]=max(hi[d],p.elements_[d]);
        }

    mt19937 rng(ExperimentSeed());
    vector<Query> null_queries;
    null_queries.reserve(real_queries.size());
    for(const Query& q: real_queries){
        Query n = q;
        for(size_t d=0;d<Constants::DIM;d++){
            const double extent = q.high_.elements_[d] - q.low_.elements_[d];
            const double span   = max(0.0, (hi[d]-lo[d]) - extent);
            const double start  = lo[d] + uniform_real_distribution<double>(0.0, 1.0)(rng)*span;
            n.low_.elements_[d]  = start;
            n.high_.elements_[d] = start + extent;
        }
        null_queries.push_back(n);
    }
    cerr<<"NULL_WORKLOAD: trained on "<<null_queries.size()<<" position-randomized queries"<<endl;
    return null_queries;
}


/**
 * Write the build-cost decomposition alongside the wall-clock build time.
 *
 * `build_time` keeps its existing meaning and position so nothing downstream
 * moves. The phases are additional columns; `build_unaccounted_s` is the part
 * of the wall clock no phase claimed, logged rather than hidden so a decomposition
 * that has drifted out of date is visible in the data instead of being asserted.
 */
static void LogBuildProfile(json& log_json, double wall_seconds){
    const BuildProfile& p = CurrentBuildProfile();

    log_json["build_workload_model_s"]  = p.workload_model_s;
    log_json["build_workload_oracle_s"] = p.workload_oracle_s;
    log_json["build_learn_s"]           = p.learn_s;
    log_json["build_learn_oracle_s"]    = p.learn_oracle_s;
    log_json["build_eval_s"]            = p.eval_s;
    // Everything the index did not claim as learning, workload-modelling or
    // serialization. Derived so the five phases always sum to build_time.
    log_json["build_construct_s"]      = wall_seconds - p.ClaimedSeconds();
    log_json["build_serialize_s"]      = p.serialize_s;

    // The two headline components. Disjoint, so together with construct and
    // serialize they partition build_total_s and can be stacked in a figure.
    log_json["build_learning_s"]           = p.LearningSeconds();
    log_json["build_workload_awareness_s"] = p.WorkloadAwarenessSeconds();

    log_json["build_search_trials"]       = p.search_trials;
    log_json["build_search_improvements"] = p.search_improvements;
    log_json["build_oracle_calls"]        = p.oracle_calls;
    log_json["build_training_queries"]    = p.training_queries;
    log_json["build_learned_nodes"]       = p.learned_nodes;
    log_json["build_fallback_nodes"]      = p.fallback_nodes;

    // build_time keeps its historical meaning, which for RSMI is the Python
    // training span only. build_total_s is the figure the phases always sum to,
    // so downstream never has to know which indexes those two differ for.
    log_json["build_total_s"]        = wall_seconds;
    log_json["build_accounting"]     = BuildAccountingName(p.accounting);
    log_json["null_workload"]        = EnvFlag("NULL_WORKLOAD", false);
}


// ===================== Storage-backend measurement passes =====================
//
// Every index section used to carry its own copy of the disk-backed timing loop.
// They now share RunStoragePasses() below; see phase4-evaluator-integration.md.

/** Frame budget and pass settings, read once from the environment. */
struct StoragePassConfig{
    vector<double> fractions{0.25, 0.05, 0.01, 0.002};
    BufferPoolFloorMode floor_mode{BufferPoolFloorMode::kBlock};
    string policy{"LRU"};
    bool release_blocks{true};
    bool verify{false};
    bool direct_io{false};

    // Below this block size the paged store is measuring padding, not the index.
    // A block never shares a page, so with 4096-byte pages and 16-byte records a
    // block of 256 records fills a page exactly and anything smaller leaves at
    // least half of every page empty -- 8x write amplification at BLOCK_SIZE=32,
    // where miss counts then say more about the page layout than about the index.
    size_t min_block_size{256};

    // What one page miss costs on this node's storage, measured once per task.
    // Logged with every row so a latency taken on a contended node is
    // identifiable afterwards rather than merely suspected.
    DeviceProbeResult device_probe;
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
    config.direct_io      = EnvFlag("BUFFER_POOL_DIRECT_IO", false);
    config.min_block_size = EnvSizeT("BUFFER_POOL_MIN_BLOCK_SIZE", 256);

    // Defaults on exactly when direct I/O is on: with the page cache in the way
    // a per-miss cost is not a device property and there is nothing to
    // calibrate, but once misses reach the device the number is what makes them
    // interpretable. Costs a few seconds, once per task.
    if(EnvFlag("DEVICE_LATENCY_PROBE", config.direct_io)){
        const size_t page_bytes = EnvSizeT("PAGE_BYTES", kDefaultPageBytes);
        config.device_probe = ProbeBlockstoreDevice(page_bytes, config.direct_io);
        if(!config.device_probe.ran)
            cerr<<"device probe did not run: "<<config.device_probe.error<<endl;
        else
            cerr<<"device probe ("<<(config.direct_io ? "O_DIRECT" : "buffered")<<"): mean "
                <<config.device_probe.mean_ns<<" ns/page, p50 "<<config.device_probe.p50_ns
                <<", p99 "<<config.device_probe.p99_ns<<endl;
    }
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

/**
 * Run every disk-backed measurement for one index.
 *
 * `run_one(i, out)` executes query i and leaves its results in `out`. It takes the
 * output vector by reference rather than returning it so the timed loop keeps the
 * copy-assign the original code had -- returning would elide into a move and shift
 * the latency baseline away from previously collected numbers.
 *
 * Results are nested inside `log_json` rather than written to a parallel bp_ file.
 * The split existed because prepare_results() in Results/plot_results.py groups on
 * a key that does not mention bufferpool_fraction and would have averaged the
 * budgets together; nesting sidesteps that without a second file, because the row
 * count is still one per index.
 *
 *     log_json["storage"]              geometry and device facts, once
 *     log_json["disk_backed_results"]  one object per budget fraction
 *
 * This function is also the only place that materializes the disk backends. That
 * is deliberate: it runs after the caller's build timer has stopped, so the cost
 * of writing the index file cannot land in build_time. Doing it here rather than
 * in twelve index sections also means no index can be missed -- a missed one would
 * have left PagedBackendPtr() null and silently dropped that index from the disk
 * results with no error anywhere.
 */
template <typename RunOne>
void RunStoragePasses(BlockStore& store, size_t query_count, RunOne&& run_one,
                      json& log_json, const StoragePassConfig& config){
    // log_json is reused across all twelve index sections, so these have to be
    // assigned unconditionally: an early return below would otherwise leave the
    // previous index's storage results attached to this one.
    log_json["disk_backed_results"] = json::array();
    for(const char* stale: {"storage", "storage_materialize_s", "disk_backed_skipped"})
        log_json.erase(stale);

    if(BLOCK_SIZE < config.min_block_size){
        log_json["disk_backed_skipped"] = "block_size_below_" + to_string(config.min_block_size);
        return;
    }

    // ---- 0. materialize, outside anybody's build timer ----
    auto materialize_start = chrono::high_resolution_clock::now();
    store.MaterializeDiskBackends();
    auto materialize_end = chrono::high_resolution_clock::now();
    log_json["storage_materialize_s"] =
        chrono::duration_cast<chrono::nanoseconds>(materialize_end - materialize_start).count()/1e9;

    PagedDiskBackend* paged = store.PagedBackendPtr();

    vector<Point> result_vec;

    // ---- 1. cross-backend verification, untimed and opt-in ----
    // Must run before the release below, since it compares against the in-memory scan.
    bool results_match = true;
    if(config.verify){
        const uint64_t kFnvOffset = 1469598103934665603ULL;
        uint64_t fingerprints[3] = {kFnvOffset, kFnvOffset, kFnvOffset};
        const StorageMode modes[3] = {StorageMode::kInMemory, StorageMode::kMmap, StorageMode::kBufferPool};
        const bool have_mmap = store.MmapBackendPtr() != nullptr;

        for(int m=0;m<3;m++){
            if(m == 1 && !have_mmap) continue;   // ENABLE_MMAP_BACKEND was off
            if(m == 2 && !paged) continue;
            store.SetStorageMode(modes[m]);
            for(size_t i=0;i<query_count;i++){
                run_one(i, result_vec);
                fingerprints[m] = FingerprintPoints(fingerprints[m], result_vec);
            }
        }
        results_match = (!have_mmap || fingerprints[0] == fingerprints[1]) &&
                        (!paged     || fingerprints[0] == fingerprints[2]);
        if(!results_match)
            cerr<<"BACKEND MISMATCH for "<<log_json.value("model","?")
                <<": in-memory="<<fingerprints[0]
                <<" mmap="<<(have_mmap ? to_string(fingerprints[1]) : string("n/a"))
                <<" paged="<<(paged ? to_string(fingerprints[2]) : string("n/a"))<<endl;
        store.SetStorageMode(StorageMode::kInMemory);
    }

    if(!paged)
        return;

    // ---- 2. free the in-memory copy so the budget means something ----
    store.SetStorageMode(StorageMode::kBufferPool);
    if(config.release_blocks)
        store.ReleaseInMemoryBlocks();

    // ---- 3. facts that do not vary with the budget, recorded once ----
    json storage;
    storage["page_bytes"]        = paged->Geometry().page_bytes_;
    storage["record_bytes"]      = paged->Geometry().record_bytes_;
    storage["records_per_page"]  = paged->Geometry().records_per_page_;
    storage["index_file_bytes"]  = paged->FileBytes();
    storage["index_total_pages"] = paged->TotalDataPages()+1;
    storage["index_directory_bytes"] = paged->DirectoryBytes();
    storage["index_metadata_bytes"]  = store.MetadataBytes();
    storage["largest_block_pages"]   = paged->LargestBlockPages();
    storage["blocks_released"]       = store.BlocksReleased();
    storage["direct_io"]             = paged->DirectIo();
    storage["replacement_policy"]    = paged->PolicyName();
    storage["floor_mode"]            = (config.floor_mode == BufferPoolFloorMode::kBlock) ? "block" : "minimal";

    // Provenance, not just the verdict. `results_match` is initialised true and
    // only ever assigned inside the `if(config.verify)` block above, so writing it
    // unconditionally reported a passed check on a check that never ran whenever
    // VERIFY_BACKENDS was unset -- which is the default. Same shape as
    // device_probe_ran below: the flag is always present, the value only when it
    // means something.
    storage["backends_verified"] = config.verify;
    if(config.verify)
        storage["results_match"] = results_match;

    storage["device_probe_ran"]    = config.device_probe.ran;
    storage["device_probe_direct"] = config.device_probe.direct;
    if(config.device_probe.ran){
        storage["device_ns_per_page_mean"] = config.device_probe.mean_ns;
        storage["device_ns_per_page_p50"]  = config.device_probe.p50_ns;
        storage["device_ns_per_page_p90"]  = config.device_probe.p90_ns;
        storage["device_ns_per_page_p99"]  = config.device_probe.p99_ns;
    }
    log_json["storage"] = storage;

    // ---- 4. one cold pass per budget fraction ----
    //
    // Cold only. There used to be a second, warm pass whose numbers were reported
    // instead, on the theory that it measured a settled cache. It did not: LRU
    // retains roughly the last (frames / pages-per-query) queries, so at every
    // budget the sweep now uses except 0.25 the warm hit rate was within 0.011 of
    // the cold one -- the hits were being generated inside the pass itself, not
    // inherited. Two passes to move one number at one budget is not worth double
    // the query time, and the cold pass is the one with a defined starting state.
    for(double fraction: config.fractions){
        BufferPoolConfig pool_config;
        pool_config.fraction   = fraction;
        pool_config.floor_mode = config.floor_mode;
        pool_config.policy     = config.policy;
        paged->RebuildPool(pool_config);

        paged->ClearCache();
        paged->ResetStats();
        size_t bp_result_size = 0;
        auto pass_start = chrono::high_resolution_clock::now();
        for(size_t i=0;i<query_count;i++){
            run_one(i, result_vec);
            bp_result_size += result_vec.size();
        }
        auto pass_end = chrono::high_resolution_clock::now();
        const StorageStats stats = paged->Stats();

        json row;
        row["bufferpool_fraction"]           = fraction;
        row["bufferpool_frames"]             = paged->PoolFrames();
        row["bufferpool_bytes"]              = paged->PoolBytes();
        row["bufferpool_effective_fraction"] = paged->EffectiveFraction();
        row["bufferpool_frames_floored"]     = paged->FramesFloored();

        row["bp_result_size"]      = bp_result_size;
        row["bp_query_latency"]    = chrono::duration_cast<chrono::nanoseconds>(pass_end-pass_start).count()/query_count;
        row["bp_pages_requested"]  = stats.pages_requested;
        row["bp_page_misses"]      = stats.page_misses;
        row["bp_hit_rate"]         = stats.HitRate();
        row["bp_bytes_read"]       = stats.bytes_read;
        row["bp_evictions"]        = stats.evictions;
        row["bp_blocks_scanned"]   = stats.blocks_scanned;
        row["bp_points_decoded"]   = stats.points_decoded;

        row["bp_pages_requested_per_query"] = double(stats.pages_requested)/double(query_count);
        row["bp_page_misses_per_query"]     = double(stats.page_misses)/double(query_count);
        row["bp_blocks_scanned_per_query"]  = double(stats.blocks_scanned)/double(query_count);

        // Attribute the pass's time over its misses, taking the in-memory pass as
        // the zero-I/O baseline for everything that is not a page fetch. Under
        // direct I/O this should land near device_ns_per_page_p50; if it does not,
        // either the miss accounting or the read path is wrong, so it is worth
        // logging even though it is derivable.
        const double pass_latency = double(chrono::duration_cast<chrono::nanoseconds>(pass_end-pass_start).count())/double(query_count);
        const double misses_per_query = double(stats.page_misses)/double(query_count);
        const double compute_baseline = double(log_json.value("query_latency", 0));
        row["bp_ns_per_miss"] = (misses_per_query > 0.0)
                                    ? (pass_latency - compute_baseline)/misses_per_query
                                    : 0.0;

        log_json["disk_backed_results"].push_back(row);
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

    const StoragePassConfig storage_pass_config = LoadStoragePassConfig();

    cout<<dataset_folder_name<<" "<<data_sample_num<<" "<<data_ent_id<<" "<<BLOCK_SIZE<<" "<<query_ent_id<<" "<<selectivity<<" WAZI Started"<<endl;

    {   //############# WAZI #################
        // Training
        vector<Point> model_datapoints = datapoints;
        vector<Query> model_queries = TrainingWorkload(countbased_queries, datapoints);
        ResetBuildProfile();
        auto train_start = std::chrono::high_resolution_clock::now();
        SamplZTree wazi_obj(model_datapoints,model_queries);
        auto train_end = std::chrono::high_resolution_clock::now();
        double_t wazi_tree_build_time = chrono::duration_cast<chrono::nanoseconds>(train_end - train_start).count()/1000000000.0;

        log_json["model"]="WAZI";
        log_json["build_time"] = wazi_tree_build_time;
        LogBuildProfile(log_json, wazi_tree_build_time);

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
            log_json, storage_pass_config);
        list_of_results.push_back(log_json);
    }

    cout<<dataset_folder_name<<" "<<data_sample_num<<" "<<data_ent_id<<" "<<BLOCK_SIZE<<" "<<query_ent_id<<" "<<selectivity<<" WAZI Finished"<<endl;




    cout<<dataset_folder_name<<" "<<data_sample_num<<" "<<data_ent_id<<" "<<BLOCK_SIZE<<" "<<query_ent_id<<" "<<selectivity<<" ZIndexStarted"<<endl;
    {   //############# ZIndex #################
        // Training
        vector<Point> model_datapoints = datapoints;
        ResetBuildProfile();
        auto train_start = std::chrono::high_resolution_clock::now();
        BaseZTree zindex_obj(model_datapoints);
        auto train_end = std::chrono::high_resolution_clock::now();
        double_t zindex_tree_build_time = chrono::duration_cast<chrono::nanoseconds>(train_end - train_start).count()/1000000000.0;

        log_json["model"]="ZIndex";
        log_json["build_time"] = zindex_tree_build_time;
        LogBuildProfile(log_json, zindex_tree_build_time);

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
            log_json, storage_pass_config);

        list_of_results.push_back(log_json);
    }
    cout<<dataset_folder_name<<" "<<data_sample_num<<" "<<data_ent_id<<" "<<BLOCK_SIZE<<" "<<query_ent_id<<" "<<selectivity<<" ZIndex Finished"<<endl;



    cout<<dataset_folder_name<<" "<<data_sample_num<<" "<<data_ent_id<<" "<<BLOCK_SIZE<<" "<<query_ent_id<<" "<<selectivity<<" ZM-Index Started"<<endl;
    {   //############# ZM-Index #################
        // Training
        vector<Point> model_datapoints = datapoints;
        ResetBuildProfile();
        auto train_start = std::chrono::high_resolution_clock::now();
        ZMIndex zmindex_obj(std::move(model_datapoints));
        auto train_end = std::chrono::high_resolution_clock::now();
        double_t zmindex_tree_build_time = chrono::duration_cast<chrono::nanoseconds>(train_end - train_start).count()/1000000000.0;
        std::cout<<"Finished building ZM"<<std::endl;
        log_json["model"]="ZM";
        log_json["build_time"] = zmindex_tree_build_time;
        LogBuildProfile(log_json, zmindex_tree_build_time);



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
                log_json, storage_pass_config);
        }

        list_of_results.push_back(log_json);
    }
    cout<<dataset_folder_name<<" "<<data_sample_num<<" "<<data_ent_id<<" "<<BLOCK_SIZE<<" "<<query_ent_id<<" "<<selectivity<<" ZM-Index Finished"<<endl;
    
    
    cout<<dataset_folder_name<<" "<<data_sample_num<<" "<<data_ent_id<<" "<<BLOCK_SIZE<<" "<<query_ent_id<<" "<<selectivity<<" GRID Started"<<endl;
    {   //############# GRID #################
        // Training
        vector<Point> model_datapoints = datapoints;
        ResetBuildProfile();
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
        LogBuildProfile(log_json, unigrid_build_time);


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
            log_json, storage_pass_config);
        list_of_results.push_back(log_json);
    }
    cout<<dataset_folder_name<<" "<<data_sample_num<<" "<<data_ent_id<<" "<<BLOCK_SIZE<<" "<<query_ent_id<<" "<<selectivity<<" GRID Finished"<<endl;
    
    
    
    
    cout<<dataset_folder_name<<" "<<data_sample_num<<" "<<data_ent_id<<" "<<BLOCK_SIZE<<" "<<query_ent_id<<" "<<selectivity<<" FLOOD Started"<<endl;
    {   //############# FLOOD #################
        // Training
        vector<Point> trainer_datapoints = datapoints;
        vector<Point> model_datapoints = datapoints;
        vector<Query> model_queries = TrainingWorkload(countbased_queries, datapoints);
        ResetBuildProfile();
        auto train_start = std::chrono::high_resolution_clock::now();
        auto flood_config = FloodTrainerRandomSearch(trainer_datapoints,model_queries);
        FloodIndex flood_obj(flood_config.first,flood_config.second);
        flood_obj.LoadElements(model_datapoints);
        auto train_end = std::chrono::high_resolution_clock::now();


        double_t flood_build_time = chrono::duration_cast<chrono::nanoseconds>(train_end - train_start).count()/1000000000.0;

        log_json["model"]="FLOOD";
        log_json["build_time"] = flood_build_time;
        LogBuildProfile(log_json, flood_build_time);

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
            log_json, storage_pass_config);

        list_of_results.push_back(log_json);
    }
    cout<<dataset_folder_name<<" "<<data_sample_num<<" "<<data_ent_id<<" "<<BLOCK_SIZE<<" "<<query_ent_id<<" "<<selectivity<<" FLOOD Finished"<<endl;
    
    
    
    cout<<dataset_folder_name<<" "<<data_sample_num<<" "<<data_ent_id<<" "<<BLOCK_SIZE<<" "<<query_ent_id<<" "<<selectivity<<" STR Started"<<endl;
    {   /* ##########################    STR   ######################################*/
        // Training
        vector<Point> model_datapoints = datapoints;
        ResetBuildProfile();
        auto train_start = std::chrono::high_resolution_clock::now();
        STRTree str_tree_obj(std::move(model_datapoints));
        auto train_end = std::chrono::high_resolution_clock::now();
        double_t str_tree_build_time = chrono::duration_cast<chrono::nanoseconds>(train_end - train_start).count()/1000000000.0;

        log_json["model"]="STR";
        log_json["build_time"] = str_tree_build_time;
        LogBuildProfile(log_json, str_tree_build_time);

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
                log_json, storage_pass_config);


            list_of_results.push_back(log_json);
        }

    }
    cout<<dataset_folder_name<<" "<<data_sample_num<<" "<<data_ent_id<<" "<<BLOCK_SIZE<<" "<<query_ent_id<<" "<<selectivity<<" STR Finished"<<endl;
    
    
    
    cout<<dataset_folder_name<<" "<<data_sample_num<<" "<<data_ent_id<<" "<<BLOCK_SIZE<<" "<<query_ent_id<<" "<<selectivity<<" RSTAR Started"<<endl;

    {   //########## RSTAR #################
        // Training
        vector<Point> model_datapoints = datapoints;
        ResetBuildProfile();
        auto train_start = std::chrono::high_resolution_clock::now();
        RSTARTree rstar_tree_obj(std::move(model_datapoints));
        auto train_end = std::chrono::high_resolution_clock::now();
        double_t rstar_tree_build_time = chrono::duration_cast<chrono::nanoseconds>(train_end - train_start).count()/1000000000.0;

        log_json["model"]="RSTAR";
        log_json["build_time"] = rstar_tree_build_time;
        LogBuildProfile(log_json, rstar_tree_build_time);


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
            log_json, storage_pass_config);

        list_of_results.push_back(log_json);
    }
    cout<<dataset_folder_name<<" "<<data_sample_num<<" "<<data_ent_id<<" "<<BLOCK_SIZE<<" "<<query_ent_id<<" "<<selectivity<<" RSTAR Finished"<<endl;
    
    
    
    cout<<dataset_folder_name<<" "<<data_sample_num<<" "<<data_ent_id<<" "<<BLOCK_SIZE<<" "<<query_ent_id<<" "<<selectivity<<" CUR Started"<<endl;

    {   //########## CUR #################
        // Training
        vector<Point> model_datapoints = datapoints;
        vector<Query> model_queries = TrainingWorkload(countbased_queries, datapoints);
        ResetBuildProfile();
        auto train_start = std::chrono::high_resolution_clock::now();
        CURTree cur_tree_obj(std::move(model_datapoints),std::move(model_queries));
        cout<<" CUR Finished Building"<<endl;
        auto train_end = std::chrono::high_resolution_clock::now();
        double_t cur_tree_build_time = chrono::duration_cast<chrono::nanoseconds>(train_end - train_start).count()/1000000000.0;

        log_json["model"]="CUR";
        log_json["build_time"] = cur_tree_build_time;
        LogBuildProfile(log_json, cur_tree_build_time);
        

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
            log_json, storage_pass_config);

        list_of_results.push_back(log_json);
    }
    cout<<dataset_folder_name<<" "<<data_sample_num<<" "<<data_ent_id<<" "<<BLOCK_SIZE<<" "<<query_ent_id<<" "<<selectivity<<" CUR Finished"<<endl;
    
    
    
    cout<<dataset_folder_name<<" "<<data_sample_num<<" "<<data_ent_id<<" "<<BLOCK_SIZE<<" "<<query_ent_id<<" "<<selectivity<<" RW Started"<<endl;

    {   //########## RW #################
        // Training
        vector<Point> model_datapoints = datapoints;
        vector<Query> model_queries = TrainingWorkload(countbased_queries, datapoints);
        ResetBuildProfile();
        auto train_start = std::chrono::high_resolution_clock::now();
        cout<<" RW Build Started"<<endl;
        RWTree rw_tree_obj(std::move(model_datapoints),std::move(model_queries));
        cout<<" RW Build Finished"<<endl;
        auto train_end = std::chrono::high_resolution_clock::now();
        double_t rw_tree_build_time = chrono::duration_cast<chrono::nanoseconds>(train_end - train_start).count()/1000000000.0;

        log_json["model"]="RW";
        log_json["build_time"] = rw_tree_build_time;
        LogBuildProfile(log_json, rw_tree_build_time);
        

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
            log_json, storage_pass_config);
        list_of_results.push_back(log_json);
    }
    cout<<dataset_folder_name<<" "<<data_sample_num<<" "<<data_ent_id<<" "<<BLOCK_SIZE<<" "<<query_ent_id<<" "<<selectivity<<" RW Finished"<<endl;
    
    
    
    
    cout<<dataset_folder_name<<" "<<data_sample_num<<" "<<data_ent_id<<" "<<BLOCK_SIZE<<" "<<query_ent_id<<" "<<selectivity<<" RSMI Started"<<endl;
    {    //########## RSMI-RTree-NoLocalModel #################
        // Training
        ResetBuildProfile();
        auto rsmi_load_start = std::chrono::high_resolution_clock::now();
        RTreeBASE rsmi_tree_obj(PROJECT_ROOT+"Experiments/"+dataset_folder_name+"/TrainedIndexes/RSMI/"+query_agnostic_tree_name+".tree");
        auto rsmi_load_end = std::chrono::high_resolution_clock::now();
        const double rsmi_tree_load_seconds =
            chrono::duration_cast<chrono::nanoseconds>(rsmi_load_end-rsmi_load_start).count()/1e9;
        std::cout<<"Finished Loading RSMI-NoNN"<<std::endl;

        ifstream build_time_file(PROJECT_ROOT+"Experiments/"+dataset_folder_name+"/TrainedIndexes/RSMI/"+query_agnostic_tree_name+".time",ios::in);
        double_t rsmi_tree_build_time;
        build_time_file>>rsmi_tree_build_time;
        build_time_file.close();


        log_json["model"]="RSMI";
        log_json["build_time"] = rsmi_tree_build_time;

        // Read from a .time file written by RSMI.py, covering TrainRSMINode --
        // NN fitting plus tree serialization, in a separate Python process. It
        // excludes the dataset load there and the .tree load here, so it is not
        // the same measurement as every other row's build_time. The accounting
        // flag is what stops the two being averaged together downstream.
        CurrentBuildProfile().accounting = BuildAccounting::kOfflinePython;
        CurrentBuildProfile().learn_s    = rsmi_tree_build_time;
        CurrentBuildProfile().serialize_s = rsmi_tree_load_seconds;

        // Optional companion written by RSMI.py. Absent for trees trained before
        // the trainer was instrumented, so a miss is silent rather than fatal --
        // the counters simply stay zero, which is what they meant before anyway.
        {
            ifstream node_stats_file(PROJECT_ROOT+"Experiments/"+dataset_folder_name+
                                     "/TrainedIndexes/RSMI/"+query_agnostic_tree_name+".nodes");
            if(node_stats_file.good()){
                try{
                    json node_stats; node_stats_file>>node_stats;
                    CurrentBuildProfile().learned_nodes  = node_stats.value("learned_nodes", 0);
                    CurrentBuildProfile().fallback_nodes = node_stats.value("fallback_nodes", 0);
                    log_json["build_rsmi_leaf_nodes"]    = node_stats.value("leaf_nodes", 0);
                    log_json["build_rsmi_nn_fit_s"]      = node_stats.value("nn_fit_seconds", 0.0);
                }catch(const std::exception& e){
                    cerr<<"could not parse RSMI .nodes: "<<e.what()<<endl;
                }
            }
        }
        log_json["build_tree_load_s"] = rsmi_tree_load_seconds;
        LogBuildProfile(log_json, rsmi_tree_build_time + rsmi_tree_load_seconds);


        
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
            log_json, storage_pass_config);
        list_of_results.push_back(log_json);        
    }
    cout<<dataset_folder_name<<" "<<data_sample_num<<" "<<data_ent_id<<" "<<BLOCK_SIZE<<" "<<query_ent_id<<" "<<selectivity<<" RSMI Finished"<<endl;
    
    
    
    
    cout<<dataset_folder_name<<" "<<data_sample_num<<" "<<data_ent_id<<" "<<BLOCK_SIZE<<" "<<query_ent_id<<" "<<selectivity<<" KDTREE Started"<<endl;
    {   //############# KDTREE #################
        // Training
        vector<Point> model_datapoints = datapoints;
        ResetBuildProfile();
        auto train_start = std::chrono::high_resolution_clock::now();
        KDTree kd_tree_obj(std::move(model_datapoints));
        auto train_end = std::chrono::high_resolution_clock::now();


        double_t kd_tree_build_time = chrono::duration_cast<chrono::nanoseconds>(train_end - train_start).count()/1000000000.0;

        log_json["model"]="KD";
        log_json["build_time"] = kd_tree_build_time;
        LogBuildProfile(log_json, kd_tree_build_time);


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
            log_json, storage_pass_config);

        list_of_results.push_back(log_json);
    }
    cout<<dataset_folder_name<<" "<<data_sample_num<<" "<<data_ent_id<<" "<<BLOCK_SIZE<<" "<<query_ent_id<<" "<<selectivity<<" KDTREE Finished"<<endl;
    


    cout<<dataset_folder_name<<" "<<data_sample_num<<" "<<data_ent_id<<" "<<BLOCK_SIZE<<" "<<query_ent_id<<" "<<selectivity<<" QDTREE Started"<<endl;
    {    //############# QDTREE RandomSearchVersion #################

        // Training
        vector<Point> trainer_datapoints = datapoints;
        vector<Point> model_datapoints = datapoints;
        vector<Query> model_queries = TrainingWorkload(countbased_queries, datapoints);
        // The QDTree construction below used to sit OUTSIDE this window, so QD's
        // build_time reported the search only while every other index reported
        // search plus construction. Materializing the winning tree is part of
        // building the index, so it is timed here like everywhere else.
        ResetBuildProfile();
        auto train_start = std::chrono::high_resolution_clock::now();
        QDTreeTrainerRandomSearch(std::move(trainer_datapoints),model_queries,PROJECT_ROOT+"Experiments/"+dataset_folder_name+"/TrainedIndexes/QDTree/"+tree_name+"_areabased.txt");

        QDTree qd_tree_obj(std::move(model_datapoints),PROJECT_ROOT+"Experiments/"+dataset_folder_name+"/TrainedIndexes/QDTree/"+tree_name+"_areabased.txt");
        auto train_end = std::chrono::high_resolution_clock::now();

        double_t qd_tree_build_time = chrono::duration_cast<chrono::nanoseconds>(train_end - train_start).count()/1000000000.0;



        log_json["model"]="QD";
        log_json["build_time"] = qd_tree_build_time;
        LogBuildProfile(log_json, qd_tree_build_time);

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
            log_json, storage_pass_config);
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

    return 0;
}
