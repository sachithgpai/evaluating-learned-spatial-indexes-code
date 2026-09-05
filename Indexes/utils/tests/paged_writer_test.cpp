/**
 * Writer test for the paged store: build a real BlockStore, then read the file
 * back and check it against the in-memory truth.
 *
 * This is the "dumper" check -- it goes through ReadPageRaw(), so it exercises
 * the on-disk layout with no caching anywhere in the path.
 *
 * Build:
 *   g++ -std=c++17 -I.. paged_writer_test.cpp -o paged_writer_test
 *   ENABLE_PAGED_BACKEND=1 TEMP_BLOCKSTORE_DIR=/tmp/bs ./paged_writer_test
 */

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

#include"local_model.h"

static int g_failures = 0;

#define CHECK(cond, what)                                                       \
    do{ if(!(cond)){ std::cout<<"  FAIL: "<<(what)<<"  ["<<__LINE__<<"]\n"; g_failures++; } }while(0)

#define CHECK_EQ(got, want, what)                                               \
    do{ auto g_=(got); auto w_=(want);                                          \
        if(!(g_==w_)){ std::cout<<"  FAIL: "<<(what)<<" got "<<g_<<" want "<<w_ \
                                <<"  ["<<__LINE__<<"]\n"; g_failures++; } }while(0)


/** Build a store whose blocks are deliberately ragged, including empty ones. */
static void FillStore(BlockStore& store, std::vector<std::vector<Point>>& truth,
                      std::vector<Point>& data, const std::vector<size_t>& sizes){
    size_t cursor = 0;
    for(size_t want: sizes){
        const size_t end = std::min(data.size(), cursor+want);
        store.InsertNewBlock(data.begin()+cursor, data.begin()+end);
        truth.emplace_back(data.begin()+cursor, data.begin()+end);
        cursor = end;
    }
}


int main(){
    // This test asserts that all three backends return the same points, so it
    // turns the mmap backend on itself rather than inheriting it. The evaluator
    // leaves it off (no mmap pass is timed any more), which would otherwise make
    // the outcome depend on the caller's environment.
    setenv("ENABLE_MMAP_BACKEND", "1", 1);

    // Every assertion here is about bytes and about page hit/miss counts, both of
    // which are computed in software and cannot be moved by what the OS happens
    // to be caching. So this test is entitled to run where the page cache cannot
    // be purged; direct_io_test, which asserts on elapsed time, is not and does
    // not set this.
    setenv("ALLOW_WARM_PAGE_CACHE", "1", 1);

    std::mt19937 rng(90210);
    std::uniform_real_distribution<double_t> d(0.0, 1000.0);

    std::vector<Point> data;
    for(int i=0;i<40000;i++) data.emplace_back(d(rng), d(rng));

    // 0 exercises the empty-block path; 1 and 257 straddle the records-per-page
    // boundary at the default geometry; 256 lands exactly on it.
    std::vector<size_t> sizes;
    for(int rep=0;rep<40;rep++)
        for(size_t s: {size_t(256), size_t(1), size_t(0), size_t(257), size_t(255), size_t(700)})
            sizes.push_back(s);

    BlockStore store;
    std::vector<std::vector<Point>> truth;
    FillStore(store, truth, data, sizes);
    store.FinishedConstruction();
    store.MaterializeDiskBackends();

    PagedDiskBackend* paged = store.PagedBackendPtr();
    if(!paged){
        std::cout<<"FAILED: paged backend not built -- set ENABLE_PAGED_BACKEND=1\n";
        return 1;
    }
    const PageGeometry& geom = paged->Geometry();

    std::cout<<"blocks="<<paged->BlockCount()
             <<" points="<<paged->TotalPoints()
             <<" data_pages="<<paged->TotalDataPages()
             <<" file_bytes="<<paged->FileBytes()
             <<" rec/page="<<geom.records_per_page_<<"\n";

    // ---- 1. the header round-trips ----
    std::cout<<"header\n";
    PagedStoreHeader header = paged->ReadHeader();
    CHECK(std::memcmp(header.magic, "LSIPAGE", 8) == 0, "magic");
    CHECK_EQ(header.format_version, kPagedFormatVersion, "format version");
    CHECK_EQ(size_t(header.page_bytes), geom.page_bytes_, "page bytes");
    CHECK_EQ(size_t(header.record_bytes), geom.record_bytes_, "record bytes");
    CHECK_EQ(size_t(header.dim), Constants::DIM, "dim");
    CHECK_EQ(header.block_count, paged->BlockCount(), "block count");
    CHECK_EQ(header.total_data_pages, paged->TotalDataPages(), "data pages");
    CHECK_EQ(header.total_points, paged->TotalPoints(), "total points");

    // ---- 2. the directory is consistent and gapless ----
    std::cout<<"block directory\n";
    uint64_t expected_next = 1;                       // page 0 is the header
    uint64_t summed_pages = 0;
    for(size_t b=0;b<paged->BlockCount();b++){
        const size_t want_pages = geom.PagesForBlock(truth[b].size());
        CHECK_EQ(size_t(paged->PageCount(b)), want_pages, "pages for block "+std::to_string(b));
        CHECK_EQ(paged->FirstPage(b), expected_next, "block "+std::to_string(b)+" starts where the last ended");
        expected_next += want_pages;
        summed_pages += want_pages;
    }
    CHECK_EQ(summed_pages, paged->TotalDataPages(), "directory covers every data page");
    CHECK_EQ(paged->FileBytes(), (paged->TotalDataPages()+1)*geom.page_bytes_, "file is a whole number of pages");

    // ---- 3. every block reads back byte-identical, straight off the disk ----
    std::cout<<"raw page reads reproduce every block\n";
    std::vector<char> page(geom.page_bytes_);
    size_t mismatches = 0, points_checked = 0;
    for(size_t b=0;b<paged->BlockCount();b++){
        size_t remaining = truth[b].size(), taken = 0;
        uint64_t page_id = paged->FirstPage(b);
        while(remaining){
            const size_t here = std::min(remaining, geom.records_per_page_);
            paged->ReadPageRaw(page_id, page.data());
            for(size_t r=0;r<here;r++){
                Point got = DecodeRecord(page.data() + r*geom.record_bytes_);
                const Point& want = truth[b][taken+r];
                if(std::memcmp(got.elements_, want.elements_, sizeof(double_t)*Constants::DIM) != 0)
                    mismatches++;
                points_checked++;
            }
            taken += here; remaining -= here; page_id++;
        }
    }
    CHECK_EQ(mismatches, size_t(0), "decoded points match the in-memory blocks");
    CHECK_EQ(points_checked, size_t(paged->TotalPoints()), "every point was checked");

    // ---- 4. a block's tail page is zero-padded past its last record ----
    std::cout<<"tail padding\n";
    for(size_t b=0;b<paged->BlockCount() && b<12;b++){
        if(truth[b].empty()) continue;
        const size_t on_last = truth[b].size() % geom.records_per_page_;
        if(on_last == 0) continue;                    // block ends flush with a page
        paged->ReadPageRaw(paged->FirstPage(b)+paged->PageCount(b)-1, page.data());
        bool zeroed = true;
        for(size_t off=on_last*geom.record_bytes_; off<geom.page_bytes_; off++)
            if(page[off] != 0) zeroed = false;
        CHECK(zeroed, "unused tail of block "+std::to_string(b)+"'s last page is zeroed");
    }

    // ---- 5. scanning through the backend agrees with the in-memory backend ----
    std::cout<<"paged Scan() agrees with the in-memory scan\n";
    std::vector<size_t> all;
    for(size_t b=0;b<paged->BlockCount();b++) all.push_back(b);

    size_t query_mismatches = 0, total_results = 0;
    for(int q=0;q<200;q++){
        double_t lx=d(rng), ly=d(rng), w=10.0+d(rng)*0.05;
        Query query(Point(lx,ly), Point(lx+w, ly+w));

        std::vector<Point> mem, mmp, pgd;
        store.SetStorageMode(StorageMode::kInMemory);   store.FilterPointsFromBlocksForQuery(query, all, mem);
        store.SetStorageMode(StorageMode::kMmap);       store.FilterPointsFromBlocksForQuery(query, all, mmp);
        store.SetStorageMode(StorageMode::kBufferPool); store.FilterPointsFromBlocksForQuery(query, all, pgd);

        if(mem.size()!=pgd.size() || mem.size()!=mmp.size()){ query_mismatches++; continue; }
        for(size_t i=0;i<mem.size();i++)
            if(std::memcmp(mem[i].elements_, pgd[i].elements_, sizeof(double_t)*Constants::DIM) != 0 ||
               std::memcmp(mem[i].elements_, mmp[i].elements_, sizeof(double_t)*Constants::DIM) != 0){
                query_mismatches++; break;
            }
        total_results += mem.size();
    }
    CHECK_EQ(query_mismatches, size_t(0), "all three backends return identical results");
    CHECK(total_results > 0, "the queries actually matched something");
    std::cout<<"  checked "<<points_checked<<" stored points, "<<total_results<<" query results\n";

    std::cout<<(g_failures ? "\nFAILED" : "\nPASS")<<"  ("<<g_failures<<" failures)\n";
    return g_failures != 0;
}
