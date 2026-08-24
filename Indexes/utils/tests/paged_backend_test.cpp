/**
 * End-to-end test: a real index over seeded data, queried through all three
 * backends, plus the buffer-pool budget sweep and its invariants.
 *
 * Build:
 *   g++ -std=c++17 -I.. paged_backend_test.cpp -o paged_backend_test
 *   ENABLE_PAGED_BACKEND=1 TEMP_BLOCKSTORE_DIR=/tmp/bs ./paged_backend_test
 */

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include"../../KDTree/kdtree.h"

static int g_failures = 0;

#define CHECK(cond, what)                                                       \
    do{ if(!(cond)){ std::cout<<"  FAIL: "<<(what)<<"  ["<<__LINE__<<"]\n"; g_failures++; } }while(0)

#define CHECK_EQ(got, want, what)                                               \
    do{ auto g_=(got); auto w_=(want);                                          \
        if(!(g_==w_)){ std::cout<<"  FAIL: "<<(what)<<" got "<<g_<<" want "<<w_ \
                                <<"  ["<<__LINE__<<"]\n"; g_failures++; } }while(0)


static bool SamePoints(const std::vector<Point>& a, const std::vector<Point>& b){
    if(a.size() != b.size()) return false;
    for(size_t i=0;i<a.size();i++)
        if(std::memcmp(a[i].elements_, b[i].elements_, sizeof(double_t)*Constants::DIM) != 0)
            return false;
    return true;
}


int main(){
    // This test asserts that all three backends return the same points, so it
    // turns the mmap backend on itself rather than inheriting it. The evaluator
    // leaves it off (no mmap pass is timed any more), which would otherwise make
    // the outcome depend on the caller's environment.
    setenv("ENABLE_MMAP_BACKEND", "1", 1);

    BLOCK_SIZE = 256;

    // Clustered rather than uniform, so leaf occupancy actually varies -- KDTree
    // splits on the midpoint of the value range, not the median rank.
    std::mt19937 rng(31337);
    std::normal_distribution<double_t> spread(0.0, 60.0);
    std::uniform_real_distribution<double_t> centre(0.0, 1000.0);
    std::vector<Point> data;
    for(int c=0;c<20;c++){
        const double_t cx = centre(rng), cy = centre(rng);
        for(int i=0;i<10000;i++) data.emplace_back(cx+spread(rng), cy+spread(rng));
    }

    KDTree index(data);
    index.block_store_.MaterializeDiskBackends();
    PagedDiskBackend* paged = index.block_store_.PagedBackendPtr();
    if(!paged){
        std::cout<<"FAILED: paged backend not built -- set ENABLE_PAGED_BACKEND=1\n";
        return 1;
    }

    std::cout<<"points="<<data.size()
             <<" blocks="<<index.block_store_.NumOfBlocks()
             <<" data_pages="<<paged->TotalDataPages()
             <<" file="<<(paged->FileBytes()/1024)<<"KB"
             <<" largest_block_pages="<<paged->LargestBlockPages()
             <<" policy="<<paged->PolicyName()<<"\n";

    // Build a fixed query workload once; every configuration below replays it.
    std::uniform_real_distribution<double_t> qx(0.0, 1000.0);
    std::vector<Query> queries;
    for(int q=0;q<200;q++){
        const double_t lx=qx(rng), ly=qx(rng), w=25.0;
        queries.emplace_back(Point(lx,ly), Point(lx+w, ly+w));
    }

    // ---- 1. all three backends return byte-identical results ----
    std::cout<<"three backends agree, point for point\n";
    std::vector<std::vector<Point>> truth;
    size_t total_results = 0;
    for(Query& query: queries){
        index.block_store_.SetStorageMode(StorageMode::kInMemory);
        std::vector<Point> mem = index.RangeQuery(query);
        index.block_store_.SetStorageMode(StorageMode::kMmap);
        std::vector<Point> mmp = index.RangeQuery(query);
        index.block_store_.SetStorageMode(StorageMode::kBufferPool);
        std::vector<Point> pgd = index.RangeQuery(query);

        CHECK(SamePoints(mem, mmp), "mmap matches in-memory");
        CHECK(SamePoints(mem, pgd), "paged matches in-memory");
        truth.push_back(mem);
        total_results += mem.size();
    }
    CHECK(total_results > 0, "the workload actually returned something");
    std::cout<<"  "<<total_results<<" result points across "<<queries.size()<<" queries\n";

    // ---- 2. budget sweep, with the invariants that must hold ----
    std::cout<<"budget sweep\n";
    const double fractions[] = {1.0, 0.25, 0.05, 0.01, 0.001};

    uint64_t requested_at_full = 0;
    uint64_t previous_misses = 0;
    bool first = true;

    std::cout<<"    fraction  frames   eff.frac  floored   requested     cold    warm\n";
    for(double fraction: fractions){
        BufferPoolConfig config;
        config.fraction = fraction;
        paged->RebuildPool(config);
        index.block_store_.SetStorageMode(StorageMode::kBufferPool);

        // cold: empty pool
        paged->ClearCache();
        paged->ResetStats();
        for(size_t q=0;q<queries.size();q++){
            std::vector<Point> got = index.RangeQuery(queries[q]);
            if(!SamePoints(got, truth[q])) CHECK(false, "cold pass result mismatch");
        }
        const StorageStats cold = paged->Stats();

        // warm: same workload again, pool keeps whatever survived
        paged->ResetStats();
        for(size_t q=0;q<queries.size();q++){
            std::vector<Point> got = index.RangeQuery(queries[q]);
            if(!SamePoints(got, truth[q])) CHECK(false, "warm pass result mismatch");
        }
        const StorageStats warm = paged->Stats();

        std::cout<<"    "<<std::setw(8)<<fraction
                 <<std::setw(8)<<paged->PoolFrames()
                 <<std::setw(11)<<std::fixed<<std::setprecision(4)<<paged->EffectiveFraction()
                 <<std::setw(9)<<(paged->FramesFloored()?"yes":"no")
                 <<std::setw(12)<<warm.pages_requested
                 <<std::setw(9)<<cold.page_misses
                 <<std::setw(8)<<warm.page_misses<<"\n";

        CHECK_EQ(warm.pages_requested, warm.page_hits+warm.page_misses, "hits + misses == requests");
        CHECK(cold.page_misses <= paged->TotalDataPages()+1, "cold misses cannot exceed the file");

        if(first){
            requested_at_full = warm.pages_requested;
            CHECK_EQ(warm.page_misses, uint64_t(0), "fraction=1.0 must warm up to zero misses");
            first = false;
        }else{
            // Invariant 2: the logical access sequence never changes with the budget.
            CHECK_EQ(warm.pages_requested, requested_at_full,
                     "pages_requested is identical across fractions");
            // Invariant 3: LRU is a stack algorithm -- shrinking the pool can only
            // ever cost more misses, never fewer.
            CHECK(warm.page_misses >= previous_misses,
                  "misses are monotone as the pool shrinks (fraction "+std::to_string(fraction)+")");
        }
        previous_misses = warm.page_misses;
    }

    // ---- 3. the floor is reported honestly when it bites ----
    std::cout<<"floor modes\n";
    {
        BufferPoolConfig blocked;
        blocked.fraction = 0.0;                                   // ask for nothing at all
        blocked.floor_mode = BufferPoolFloorMode::kBlock;
        paged->RebuildPool(blocked);
        CHECK(paged->FramesFloored(), "asking for zero is reported as floored");
        CHECK_EQ(paged->PoolFrames(), paged->LargestBlockPages()+blocked.slack_frames,
                 "kBlock floor holds the largest block plus slack");

        BufferPoolConfig minimal = blocked;
        minimal.floor_mode = BufferPoolFloorMode::kMinimal;
        paged->RebuildPool(minimal);
        CHECK_EQ(paged->PoolFrames(), size_t(2), "kMinimal floor is 2 frames");

        // Even at the minimum budget the answers must still be right.
        index.block_store_.SetStorageMode(StorageMode::kBufferPool);
        bool all_match = true;
        for(size_t q=0;q<queries.size();q++)
            if(!SamePoints(index.RangeQuery(queries[q]), truth[q])) all_match = false;
        CHECK(all_match, "results are correct even with a 2-frame pool");
    }

    std::cout<<(g_failures ? "\nFAILED" : "\nPASS")<<"  ("<<g_failures<<" failures)\n";
    return g_failures != 0;
}
