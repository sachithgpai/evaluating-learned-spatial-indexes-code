/**
 * Direct I/O must change what a miss costs, and nothing else.
 *
 * The pool, its frames, its LRU and its accounting are all supposed to be
 * indifferent to how a page arrives. This test builds the same index twice --
 * once reading through the OS page cache, once with O_DIRECT -- replays one
 * fixed workload against both, and requires that the results are byte-identical
 * and the page-hit/miss counts are *exactly* equal.
 *
 * The miss-count equality is the sharp end of it. A reference string is a
 * property of the workload and the layout, so any divergence between the two
 * modes means the accounting has picked up a dependency on the read path, which
 * would quietly invalidate every count already published.
 *
 * Build:
 *   g++ -std=c++17 -O2 -I.. direct_io_test.cpp -o direct_io_test
 *   ENABLE_PAGED_BACKEND=1 TEMP_BLOCKSTORE_DIR=/tmp/$USER/bs/ ./direct_io_test
 *
 * Skips rather than fails when the filesystem refuses direct I/O -- Lustre and
 * NFS both do, and that is a property of where TEMP_BLOCKSTORE_DIR points, not
 * a defect in the code under test.
 */

#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include"../../KDTree/kdtree.h"
#include"../device_probe.h"

static int g_failures = 0;

#define CHECK(cond, what)                                                       \
    do{ if(!(cond)){ std::cout<<"  FAIL: "<<(what)<<"  ["<<__LINE__<<"]\n"; g_failures++; } }while(0)

#define CHECK_EQ(got, want, what)                                               \
    do{ auto g_=(got); auto w_=(want);                                          \
        if(!(g_==w_)){ std::cout<<"  FAIL: "<<(what)<<" got "<<g_<<" want "<<w_ \
                                <<"  ["<<__LINE__<<"]\n"; g_failures++; } }while(0)


/** FNV-1a over the raw coordinates: catches reordering, not merely size drift. */
static uint64_t Fingerprint(uint64_t hash, const std::vector<Point>& points){
    for(const Point& point: points){
        const unsigned char* bytes = reinterpret_cast<const unsigned char*>(point.elements_);
        for(size_t i=0;i<sizeof(double_t)*Constants::DIM;i++){
            hash ^= bytes[i];
            hash *= 1099511628211ULL;
        }
    }
    return hash;
}


/** One (fraction, pass) measurement, reduced to what must not vary. */
struct PassRecord{
    uint64_t fingerprint = 1469598103934665603ULL;
    uint64_t cold_requested = 0, cold_misses = 0;
    uint64_t warm_requested = 0, warm_hits = 0, warm_misses = 0, warm_evictions = 0;
    double   warm_ns_per_query = 0.0;
};


static std::vector<Point> MakeData(){
    std::mt19937 rng(31337);
    std::normal_distribution<double_t> spread(0.0, 60.0);
    std::uniform_real_distribution<double_t> centre(0.0, 1000.0);
    std::vector<Point> data;
    for(int c=0;c<20;c++){
        const double_t cx = centre(rng), cy = centre(rng);
        for(int i=0;i<10000;i++) data.emplace_back(cx+spread(rng), cy+spread(rng));
    }
    return data;
}

static std::vector<Query> MakeQueries(){
    std::mt19937 rng(9001);
    std::uniform_real_distribution<double_t> qx(0.0, 1000.0);
    std::vector<Query> queries;
    for(int q=0;q<200;q++){
        const double_t lx=qx(rng), ly=qx(rng), w=25.0;
        queries.emplace_back(Point(lx,ly), Point(lx+w, ly+w));
    }
    return queries;
}


/**
 * Build the index under the current BUFFER_POOL_DIRECT_IO setting and sweep it.
 *
 * The index is rebuilt from scratch each time rather than reusing one store:
 * the read descriptor's flags are fixed when Build() opens it, so switching
 * modes genuinely requires a new store.
 */
static bool RunSweep(const std::vector<Point>& data, std::vector<Query>& queries,
                     const double* fractions, size_t fraction_count,
                     bool expect_direct, std::vector<PassRecord>& out){
    KDTree index(data);
    index.block_store_.MaterializeDiskBackends();
    PagedDiskBackend* paged = index.block_store_.PagedBackendPtr();
    if(!paged){
        std::cout<<"FAILED: paged backend not built -- set ENABLE_PAGED_BACKEND=1\n";
        return false;
    }
    CHECK_EQ(paged->DirectIo(), expect_direct, "backend reports the requested read mode");

    for(size_t f=0;f<fraction_count;f++){
        BufferPoolConfig config;
        config.fraction = fractions[f];
        paged->RebuildPool(config);
        index.block_store_.SetStorageMode(StorageMode::kBufferPool);

        PassRecord record;

        paged->ClearCache();
        paged->ResetStats();
        for(Query& query: queries) index.RangeQuery(query);
        const StorageStats cold = paged->Stats();
        record.cold_requested = cold.pages_requested;
        record.cold_misses    = cold.page_misses;

        paged->ResetStats();
        const auto start = std::chrono::high_resolution_clock::now();
        for(Query& query: queries){
            std::vector<Point> got = index.RangeQuery(query);
            record.fingerprint = Fingerprint(record.fingerprint, got);
        }
        const auto stop = std::chrono::high_resolution_clock::now();
        const StorageStats warm = paged->Stats();

        record.warm_requested    = warm.pages_requested;
        record.warm_hits         = warm.page_hits;
        record.warm_misses       = warm.page_misses;
        record.warm_evictions    = warm.evictions;
        record.warm_ns_per_query =
            double(std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start).count())/double(queries.size());

        out.push_back(record);
    }
    return true;
}


int main(){
    BLOCK_SIZE = 256;

    // Fail fast and legibly if this filesystem will not serve direct reads at
    // all, rather than reporting it as a defect in the backend.
    const DeviceProbeResult probe = ProbeDeviceLatency(ResolveBlockstoreDir(), 4096, 64ull<<20, 200, true);
    if(!probe.ran){
        std::cout<<"SKIPPED: direct I/O unavailable under "<<ResolveBlockstoreDir()
                 <<"\n  "<<probe.error<<"\n";
        return 0;
    }

    const std::vector<Point> data = MakeData();
    std::vector<Query> queries = MakeQueries();
    const double fractions[] = {1.0, 0.25, 0.05, 0.01};
    const size_t fraction_count = sizeof(fractions)/sizeof(fractions[0]);

    std::cout<<"buffered sweep\n";
    setenv("BUFFER_POOL_DIRECT_IO", "0", 1);
    std::vector<PassRecord> buffered;
    if(!RunSweep(data, queries, fractions, fraction_count, false, buffered)) return 1;

    std::cout<<"direct sweep\n";
    setenv("BUFFER_POOL_DIRECT_IO", "1", 1);
    std::vector<PassRecord> direct;
    if(!RunSweep(data, queries, fractions, fraction_count, true, direct)) return 1;

    CHECK_EQ(buffered.size(), direct.size(), "both sweeps produced the same number of points");

    std::cout<<"\n  fraction   misses/q   buffered us/q   direct us/q    ratio\n";
    for(size_t f=0;f<buffered.size() && f<direct.size();f++){
        const PassRecord& b = buffered[f];
        const PassRecord& d = direct[f];

        // The whole point of the exercise: same answers, same I/O, different cost.
        CHECK_EQ(d.fingerprint,    b.fingerprint,    "results are identical to the buffered run");
        CHECK_EQ(d.cold_requested, b.cold_requested, "cold pages requested is read-path independent");
        CHECK_EQ(d.cold_misses,    b.cold_misses,    "cold misses are read-path independent");
        CHECK_EQ(d.warm_requested, b.warm_requested, "warm pages requested is read-path independent");
        CHECK_EQ(d.warm_hits,      b.warm_hits,      "warm hits are read-path independent");
        CHECK_EQ(d.warm_misses,    b.warm_misses,    "warm misses are read-path independent");
        CHECK_EQ(d.warm_evictions, b.warm_evictions, "evictions are read-path independent");

        std::cout<<"  "<<std::setw(8)<<std::fixed<<std::setprecision(3)<<fractions[f]
                 <<std::setw(11)<<std::setprecision(1)<<(double(d.warm_misses)/double(queries.size()))
                 <<std::setw(16)<<(b.warm_ns_per_query/1000.0)
                 <<std::setw(14)<<(d.warm_ns_per_query/1000.0)
                 <<std::setw(9)<<(b.warm_ns_per_query > 0.0 ? d.warm_ns_per_query/b.warm_ns_per_query : 0.0)
                 <<"\n";
    }

    // With misses actually reaching the device, the direct run has to be slower
    // wherever there are misses to pay for. If it is not, O_DIRECT was silently
    // ignored -- which is exactly the failure this whole change exists to avoid.
    for(size_t f=0;f<buffered.size() && f<direct.size();f++)
        if(direct[f].warm_misses > 0)
            CHECK(direct[f].warm_ns_per_query > buffered[f].warm_ns_per_query,
                  "direct I/O is slower than cached reads where misses occur");

    std::cout<<"\n"<<(g_failures ? "FAILED" : "PASSED")<<" ("<<g_failures<<" failures)\n";
    return g_failures ? 1 : 0;
}
