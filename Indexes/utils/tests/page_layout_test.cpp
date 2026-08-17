/**
 * Geometry and record-encoding tests for the paged store.
 *
 * Build (each test is its own single translation unit -- constants.h defines
 * non-inline globals, so they cannot be linked together):
 *   g++ -std=c++17 -I.. page_layout_test.cpp -o page_layout_test && ./page_layout_test
 */

#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>

#include"page_layout.h"

static int g_failures = 0;

#define CHECK(cond, what)                                                       \
    do{ if(!(cond)){ std::cout<<"  FAIL: "<<(what)<<"  ["<<__LINE__<<"]\n"; g_failures++; } }while(0)

#define CHECK_EQ(got, want, what)                                               \
    do{ auto g_=(got); auto w_=(want);                                          \
        if(!(g_==w_)){ std::cout<<"  FAIL: "<<(what)<<" got "<<g_<<" want "<<w_ \
                                <<"  ["<<__LINE__<<"]\n"; g_failures++; } }while(0)


static void TestDefaultGeometry(){
    std::cout<<"default geometry (4096 / 16)\n";
    PageGeometry g;
    CHECK_EQ(g.page_bytes_, size_t(4096), "default page bytes");
    CHECK_EQ(g.record_bytes_, sizeof(Point), "default record bytes == sizeof(Point)");
    CHECK_EQ(g.records_per_page_, size_t(256), "records per page");
    CHECK_EQ(g.DeadTailBytes(), size_t(0), "4096/16 divides exactly, so no dead tail");

    // Block-to-page arithmetic, including the boundaries.
    CHECK_EQ(g.PagesForBlock(0),   size_t(0), "an empty block occupies no pages");
    CHECK_EQ(g.PagesForBlock(1),   size_t(1), "a 1-point block still costs a whole page");
    CHECK_EQ(g.PagesForBlock(255), size_t(1), "255 points fit in one page");
    CHECK_EQ(g.PagesForBlock(256), size_t(1), "256 points exactly fill one page");
    CHECK_EQ(g.PagesForBlock(257), size_t(2), "257 points spill into a second page");
    CHECK_EQ(g.PagesForBlock(512), size_t(2), "512 points fill two pages exactly");
}


static void TestGeometrySweep(){
    std::cout<<"geometry sweep (exact-divide and ragged-tail cases)\n";
    const size_t page_sizes[]   = {512, 4096, 65536};
    const size_t record_sizes[] = {16, 24, 144, 512};

    for(size_t page_bytes: page_sizes){
        for(size_t record_bytes: record_sizes){
            if(record_bytes > page_bytes) continue;

            PageGeometry g(page_bytes, record_bytes);
            CHECK_EQ(g.records_per_page_, page_bytes/record_bytes, "records per page");

            // A record must never straddle a page boundary: the last slot has to end
            // at or before the end of the page.
            CHECK(g.records_per_page_*record_bytes <= page_bytes, "records fit inside the page");
            CHECK_EQ(g.DeadTailBytes(), page_bytes - g.records_per_page_*record_bytes, "dead tail");

            for(size_t n: {size_t(1), g.records_per_page_-1, g.records_per_page_,
                           g.records_per_page_+1, 5*g.records_per_page_}){
                const size_t want = (n + g.records_per_page_ - 1)/g.records_per_page_;
                CHECK_EQ(g.PagesForBlock(n), want, "ceil division");
            }
        }
    }
}


static void TestGeometryRejectsNonsense(){
    std::cout<<"geometry validation\n";
    bool threw = false;
    try { PageGeometry(4096, 8); }            catch(const std::exception&){ threw = true; }
    CHECK(threw, "record smaller than a Point is rejected");

    threw = false;
    try { PageGeometry(4096, 8192); }         catch(const std::exception&){ threw = true; }
    CHECK(threw, "record larger than a page is rejected");

    threw = false;
    try { PageGeometry(5000, 16); }           catch(const std::exception&){ threw = true; }
    CHECK(threw, "page size that is not a multiple of 512 is rejected");
}


static void TestRecordRoundTrip(){
    std::cout<<"record encode/decode is bit-exact\n";
    const double_t nasty[] = {
        0.0, -0.0, 1.0, -1.0,
        std::numeric_limits<double_t>::min(),
        std::numeric_limits<double_t>::max(),
        std::numeric_limits<double_t>::lowest(),
        std::numeric_limits<double_t>::denorm_min(),
        std::numeric_limits<double_t>::epsilon(),
        std::numeric_limits<double_t>::infinity(),
        -std::numeric_limits<double_t>::infinity(),
        std::numeric_limits<double_t>::quiet_NaN(),
        3.14159265358979311599796346854,
    };
    const size_t count = sizeof(nasty)/sizeof(nasty[0]);

    // Sweep record widths, including one that is not a multiple of 8, to prove the
    // memcpy decode does not depend on record alignment.
    for(size_t record_bytes: {sizeof(Point), size_t(20), size_t(24), size_t(144)}){
        std::vector<char> buffer(record_bytes*count + 64, char(0xAB));

        for(size_t i=0;i<count;i++){
            Point original(nasty[i], nasty[(i+1)%count]);
            EncodeRecord(buffer.data() + i*record_bytes, original, record_bytes);
        }
        for(size_t i=0;i<count;i++){
            Point original(nasty[i], nasty[(i+1)%count]);
            Point decoded = DecodeRecord(buffer.data() + i*record_bytes);

            // Compare the raw bits: NaN != NaN under ==, but the bytes must survive.
            CHECK(std::memcmp(original.elements_, decoded.elements_,
                              sizeof(double_t)*Constants::DIM) == 0,
                  "round trip is bit-exact at record width "+std::to_string(record_bytes));
        }
        // Padding past the coordinates must be zeroed, not left as the 0xAB filler.
        if(record_bytes > sizeof(double_t)*Constants::DIM){
            bool all_zero = true;
            for(size_t b=sizeof(double_t)*Constants::DIM;b<record_bytes;b++)
                if(buffer[b] != 0) all_zero = false;
            CHECK(all_zero, "payload padding is zeroed");
        }
    }
}


static void TestHeaderFits(){
    std::cout<<"header\n";
    CHECK(sizeof(PagedStoreHeader) <= 512, "header fits in the smallest legal page");
}


int main(){
    TestDefaultGeometry();
    TestGeometrySweep();
    TestGeometryRejectsNonsense();
    TestRecordRoundTrip();
    TestHeaderFits();

    std::cout<<(g_failures ? "\nFAILED" : "\nPASS")<<"  ("<<g_failures<<" failures)\n";
    return g_failures != 0;
}
