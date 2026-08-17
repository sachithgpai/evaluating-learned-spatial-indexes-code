/**
 * Direct tests for BufferPool, against a synthetic file where page i is filled
 * with byte pattern i. No BlockStore, no index, no dataset.
 *
 * Build:
 *   g++ -std=c++17 -I.. buffer_pool_test.cpp -o buffer_pool_test && ./buffer_pool_test
 */

#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <list>
#include <random>
#include <string>
#include <unistd.h>
#include <unordered_map>
#include <vector>

#include"buffer_pool.h"

static int g_failures = 0;

#define CHECK(cond, what)                                                       \
    do{ if(!(cond)){ std::cout<<"  FAIL: "<<(what)<<"  ["<<__LINE__<<"]\n"; g_failures++; } }while(0)

#define CHECK_EQ(got, want, what)                                               \
    do{ auto g_=(got); auto w_=(want);                                          \
        if(!(g_==w_)){ std::cout<<"  FAIL: "<<(what)<<" got "<<g_<<" want "<<w_ \
                                <<"  ["<<__LINE__<<"]\n"; g_failures++; } }while(0)

static const size_t kPageBytes = 4096;


/** A temp file of `pages` pages; page i is filled with the byte (i % 251). */
class ScratchFile{
    public:
        ScratchFile(size_t pages): pages_(pages){
            const char* dir = std::getenv("TEMP_BLOCKSTORE_DIR");
            name_ = std::string(dir ? dir : "/tmp") + "/bp_test_" + std::to_string(getpid());
            fd_ = open(name_.c_str(), O_RDWR|O_CREAT|O_TRUNC, 0600);
            if(fd_ < 0) throw std::runtime_error("cannot create "+name_);

            std::vector<char> page(kPageBytes);
            for(size_t p=0;p<pages;p++){
                std::memset(page.data(), Pattern(p), kPageBytes);
                if(write(fd_, page.data(), kPageBytes) != ssize_t(kPageBytes))
                    throw std::runtime_error("short write");
            }
            fsync(fd_);
            std::remove(name_.c_str());          // same unlink-while-open trick as the backend
        }
        ~ScratchFile(){ if(fd_ >= 0) close(fd_); }

        static char Pattern(size_t page_id){ return char(page_id % 251); }
        int fd() const { return fd_; }
        size_t pages() const { return pages_; }

    private:
        std::string name_;
        int fd_{-1};
        size_t pages_;
};

static BufferPool MakePool(const ScratchFile& file, size_t frames){
    return BufferPool(file.fd(), kPageBytes, frames, MakeReplacementPolicy("LRU"));
}

/** Pin, verify the bytes are page_id's pattern, unpin. */
static void Touch(BufferPool& pool, uint64_t page_id, int& bad){
    const char* data = pool.Pin(page_id);
    if(data[0] != ScratchFile::Pattern(page_id) || data[kPageBytes-1] != ScratchFile::Pattern(page_id))
        bad++;
    pool.Unpin(page_id);
}


static void TestReadsAreCorrect(){
    std::cout<<"every page reads back as its own pattern\n";
    ScratchFile file(64);
    BufferPool pool = MakePool(file, 8);
    int bad = 0;
    for(uint64_t p=0;p<file.pages();p++) Touch(pool, p, bad);
    for(uint64_t p=file.pages();p-- > 0;)  Touch(pool, p, bad);
    CHECK_EQ(bad, 0, "pool never handed back the wrong frame");
}


static void TestFitsEntirely(){
    std::cout<<"pool big enough to hold everything\n";
    ScratchFile file(32);
    BufferPool pool = MakePool(file, 32);
    int bad = 0;

    for(uint64_t p=0;p<32;p++) Touch(pool, p, bad);
    CHECK_EQ(pool.Stats().page_misses, uint64_t(32), "first pass misses every page");
    CHECK_EQ(pool.Stats().evictions, uint64_t(0), "nothing had to be evicted");

    pool.ResetStats();
    for(uint64_t p=0;p<32;p++) Touch(pool, p, bad);
    CHECK_EQ(pool.Stats().page_misses, uint64_t(0), "second pass is all hits");
    CHECK_EQ(pool.Stats().page_hits, uint64_t(32), "second pass hit count");
    CHECK(pool.Stats().HitRate() == 1.0, "hit rate is 1.0");
    CHECK_EQ(bad, 0, "contents still correct");

    // ResetStats must not disturb residency.
    CHECK_EQ(pool.ResidentPages(), size_t(32), "ResetStats left the pages resident");
}


static void TestSingleFrameThrash(){
    std::cout<<"one frame, alternating pages -- pure thrash\n";
    ScratchFile file(4);
    BufferPool pool = MakePool(file, 1);
    int bad = 0;
    for(int i=0;i<20;i++) Touch(pool, uint64_t(i%2), bad);

    CHECK_EQ(pool.Stats().pages_requested, uint64_t(20), "every access counted");
    CHECK_EQ(pool.Stats().page_misses, uint64_t(20), "every access missed");
    CHECK_EQ(pool.Stats().page_hits, uint64_t(0), "no hits are possible with one frame");
    CHECK(pool.Stats().HitRate() == 0.0, "hit rate is 0.0");
    CHECK_EQ(bad, 0, "contents still correct while thrashing");
}


static void TestLruNotFifo(){
    std::cout<<"smallest case that separates LRU from FIFO\n";
    // frames=2, sequence 1,2,3,2 -> miss, miss, miss(evicting 1, the LRU), hit.
    // FIFO would evict 1 as well here, so also check 1,2,1,3,2:
    //   LRU  evicts 2 on the 3 (1 was touched more recently) -> final 2 misses
    //   FIFO evicts 1 on the 3                               -> final 2 hits
    ScratchFile file(8);
    int bad = 0;
    {
        BufferPool pool = MakePool(file, 2);
        for(uint64_t p: {1,2,3,2}) Touch(pool, p, bad);
        CHECK_EQ(pool.Stats().page_misses, uint64_t(3), "1,2,3,2 misses 3 times");
        CHECK_EQ(pool.Stats().page_hits, uint64_t(1), "1,2,3,2 hits once");
    }
    {
        BufferPool pool = MakePool(file, 2);
        for(uint64_t p: {1,2,1,3,2}) Touch(pool, p, bad);
        // 1 miss, 2 miss, 1 hit, 3 miss (evicts 2), 2 miss  => 4 misses, 1 hit
        CHECK_EQ(pool.Stats().page_misses, uint64_t(4), "1,2,1,3,2 misses 4 times under LRU");
        CHECK_EQ(pool.Stats().page_hits, uint64_t(1), "1,2,1,3,2 hits once under LRU");
    }
    CHECK_EQ(bad, 0, "contents correct");
}


/** An independently written LRU, used to cross-check the pool's whole miss sequence. */
class ReferenceLru{
    public:
        explicit ReferenceLru(size_t capacity): capacity_(capacity) {}

        /** True on a hit. */
        bool Access(uint64_t page){
            auto it = where_.find(page);
            if(it != where_.end()){
                order_.erase(it->second);
                order_.push_back(page);
                where_[page] = std::prev(order_.end());
                return true;
            }
            if(order_.size() == capacity_){
                where_.erase(order_.front());
                order_.pop_front();
            }
            order_.push_back(page);
            where_[page] = std::prev(order_.end());
            return false;
        }

    private:
        size_t capacity_;
        std::list<uint64_t> order_;                                   // front = LRU
        std::unordered_map<uint64_t, std::list<uint64_t>::iterator> where_;
};


static void TestAgainstReferenceLru(){
    std::cout<<"whole hit/miss sequence vs an independent LRU (10000 accesses, 50 pages)\n";
    ScratchFile file(50);

    for(size_t frames: {size_t(1), size_t(2), size_t(8), size_t(17), size_t(50)}){
        BufferPool pool = MakePool(file, frames);
        ReferenceLru reference(frames);

        std::mt19937 rng(424242);
        std::uniform_int_distribution<uint64_t> pick(0, 49);

        size_t divergences = 0;
        uint64_t expect_hits = 0;
        for(int i=0;i<10000;i++){
            const uint64_t page = pick(rng);

            const uint64_t hits_before = pool.Stats().page_hits;
            int bad = 0;
            Touch(pool, page, bad);
            const bool pool_hit = (pool.Stats().page_hits != hits_before);

            if(bad) divergences++;
            const bool ref_hit = reference.Access(page);
            if(pool_hit != ref_hit) divergences++;
            if(ref_hit) expect_hits++;
        }

        CHECK_EQ(divergences, size_t(0), "per-access agreement at frames="+std::to_string(frames));
        CHECK_EQ(pool.Stats().page_hits, expect_hits, "total hits at frames="+std::to_string(frames));
        CHECK_EQ(pool.Stats().pages_requested, uint64_t(10000), "request count");
        CHECK_EQ(pool.Stats().page_hits + pool.Stats().page_misses, uint64_t(10000),
                 "hits + misses == requests");
    }
}


static void TestMonotoneInPoolSize(){
    std::cout<<"LRU is a stack algorithm: misses never increase with more frames\n";
    ScratchFile file(50);
    uint64_t previous = UINT64_MAX;
    for(size_t frames: {size_t(1), size_t(2), size_t(4), size_t(8), size_t(16), size_t(32), size_t(50)}){
        BufferPool pool = MakePool(file, frames);
        std::mt19937 rng(99);
        std::uniform_int_distribution<uint64_t> pick(0, 49);
        int bad = 0;
        for(int i=0;i<5000;i++) Touch(pool, pick(rng), bad);

        const uint64_t misses = pool.Stats().page_misses;
        CHECK(misses <= previous, "misses non-increasing at frames="+std::to_string(frames)+
                                  " ("+std::to_string(misses)+" vs "+std::to_string(previous)+")");
        previous = misses;
    }
    CHECK_EQ(previous, uint64_t(50), "with a frame per page, misses == compulsory misses only");
}


static void TestPinnedPagesSurvive(){
    std::cout<<"pinned pages are never evicted\n";
    ScratchFile file(64);
    BufferPool pool = MakePool(file, 4);

    const char* held = pool.Pin(7);                 // keep this one pinned throughout
    CHECK(held[0] == ScratchFile::Pattern(7), "pinned page contents");

    int bad = 0;
    for(uint64_t p=10;p<40;p++) Touch(pool, p, bad);  // churn the other three frames

    CHECK(pool.IsResident(7), "page 7 stayed resident while pinned");
    CHECK(held[0] == ScratchFile::Pattern(7) && held[kPageBytes-1] == ScratchFile::Pattern(7),
          "pinned frame was not reused underneath us");
    pool.Unpin(7);
    CHECK_EQ(bad, 0, "other pages still correct");
}


static void TestOverPinningThrows(){
    std::cout<<"pinning more distinct pages than frames throws\n";
    ScratchFile file(16);
    BufferPool pool = MakePool(file, 3);

    pool.Pin(0); pool.Pin(1); pool.Pin(2);
    bool threw = false;
    try { pool.Pin(3); } catch(const std::exception&){ threw = true; }
    CHECK(threw, "fourth simultaneous pin is refused, not silently corrupting");

    pool.Unpin(0);
    bool ok = true;
    try { pool.Pin(3); pool.Unpin(3); } catch(const std::exception&){ ok = false; }
    CHECK(ok, "after an unpin there is room again");
    pool.Unpin(1); pool.Unpin(2);

    threw = false;
    try { pool.Unpin(9); } catch(const std::exception&){ threw = true; }
    CHECK(threw, "unpinning a non-resident page is an error");
}


static void TestClearIsCold(){
    std::cout<<"Clear() returns the pool to cold\n";
    ScratchFile file(16);
    BufferPool pool = MakePool(file, 16);
    int bad = 0;

    for(uint64_t p=0;p<16;p++) Touch(pool, p, bad);
    CHECK_EQ(pool.ResidentPages(), size_t(16), "everything resident");

    pool.Clear();
    CHECK_EQ(pool.ResidentPages(), size_t(0), "Clear() evicted everything");

    pool.ResetStats();
    for(uint64_t p=0;p<16;p++) Touch(pool, p, bad);
    CHECK_EQ(pool.Stats().page_misses, uint64_t(16), "post-Clear pass misses everything again");
    CHECK_EQ(bad, 0, "contents correct after Clear");
}


int main(){
    try{
        TestReadsAreCorrect();
        TestFitsEntirely();
        TestSingleFrameThrash();
        TestLruNotFifo();
        TestAgainstReferenceLru();
        TestMonotoneInPoolSize();
        TestPinnedPagesSurvive();
        TestOverPinningThrows();
        TestClearIsCold();
    }catch(const std::exception& e){
        std::cout<<"EXCEPTION: "<<e.what()<<"\n";
        return 2;
    }

    std::cout<<(g_failures ? "\nFAILED" : "\nPASS")<<"  ("<<g_failures<<" failures)\n";
    return g_failures != 0;
}
