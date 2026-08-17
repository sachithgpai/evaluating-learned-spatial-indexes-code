#ifndef PAGE_LAYOUT_H
#define PAGE_LAYOUT_H

/**
 * @file page_layout.h
 * @brief On-disk page geometry for the paged block store.
 *
 * The paged store groups point records into fixed-size pages. A page is the
 * unit of I/O, of caching and (from Phase 3) of pinning; a block is the unit
 * the index reasons about. Keeping the two separate is the whole point: it is
 * what lets the buffer pool hold a budget measured in pages while the indexes
 * keep choosing blocks.
 *
 * One rule governs the layout: **a block never shares a page with another
 * block**. A block of n points therefore occupies ceil(n / records_per_page)
 * pages, and the unused tail of its final page is simply wasted. That waste is
 * not an oversight -- it is the cost of underfull blocks, made visible and
 * measurable rather than hidden.
 */

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <string>

#include"point.h"


/**
 * Default page size, in bytes.
 *
 * 4096 is chosen because it is the size essentially every layer of a real
 * system already agrees on:
 *   - the x86-64 virtual-memory page, so it is the granularity the kernel
 *     faults in and the granularity `mmap` would have used anyway;
 *   - the default block size of ext4/XFS, so one page is one filesystem block
 *     and a read never straddles two of them;
 *   - squarely in the range real database engines pick for a leaf page
 *     (SQLite 4K, PostgreSQL 8K, InnoDB 16K).
 *
 *
 * Be aware of one consequence at the default record width. 4096 / 16 = 256
 * records per page, so for every BLOCK_SIZE <= 256 a full block fits in
 * exactly one page and pages-per-fetch is pinned at 1. That is not a
 * measurement artifact -- a block that fits in a page really does cost one
 * I/O -- but it does mean the interesting variation at those block sizes comes
 * from cache reuse (pages_requested vs page_misses), not from I/O
 * amplification. Sweep PAGE_BYTES over {512, 1024, 4096, 16384} to show the
 * index ranking is stable under I/O granularity.
 */
static constexpr size_t kDefaultPageBytes = 4096;

/** Default record width: exactly a Point, with no payload padding. */
static constexpr size_t kDefaultRecordBytes = sizeof(Point);

static constexpr uint32_t kPagedFormatVersion = 1;


/**
 * Page 0 of the file: enough metadata to interpret the rest without knowing
 * which environment produced it.
 *
 * Everything here is also known at runtime from the configuration, so this
 * costs one page purely to buy two things: a standalone dumper that can open a
 * file cold, and detection of a file written under one geometry being read
 * under another (which would otherwise decode to silent garbage).
 */
struct PagedStoreHeader{
    char     magic[8];             // "LSIPAGE\0"
    uint32_t format_version;
    uint32_t page_bytes;
    uint32_t record_bytes;
    uint32_t dim;
    uint32_t reserved;
    uint64_t block_count;
    uint64_t total_data_pages;     // excludes this header page
    uint64_t total_points;
};

static_assert(sizeof(PagedStoreHeader) <= 512,
              "header must fit inside the smallest legal page");


/**
 * Page/record arithmetic for one paged store.
 *
 * Any `page_bytes % record_bytes` remainder is left unused rather than split,
 * which is what guarantees a record never straddles a page boundary. At the
 * defaults (4096 / 16) the division is exact and the remainder is zero.
 */
struct PageGeometry{
    size_t page_bytes_;
    size_t record_bytes_;
    size_t records_per_page_;

    PageGeometry(size_t page_bytes = kDefaultPageBytes, size_t record_bytes = kDefaultRecordBytes)
        : page_bytes_(page_bytes), record_bytes_(record_bytes){

        if(record_bytes_ < sizeof(Point))
            throw std::runtime_error("PageGeometry: record_bytes ("+std::to_string(record_bytes_)+
                                     ") is smaller than a Point ("+std::to_string(sizeof(Point))+")");
        if(page_bytes_ < record_bytes_)
            throw std::runtime_error("PageGeometry: page_bytes ("+std::to_string(page_bytes_)+
                                     ") cannot hold a single record ("+std::to_string(record_bytes_)+")");
        if(page_bytes_ % 512 != 0)
            throw std::runtime_error("PageGeometry: page_bytes ("+std::to_string(page_bytes_)+
                                     ") must be a multiple of 512");

        records_per_page_ = page_bytes_ / record_bytes_;
    }

    /** Pages occupied by a block of `points` points. Zero points means zero pages. */
    size_t PagesForBlock(size_t points) const {
        return (points + records_per_page_ - 1) / records_per_page_;
    }

    /** Bytes of a page that hold no record, and never will. */
    size_t DeadTailBytes() const {
        return page_bytes_ - records_per_page_*record_bytes_;
    }
};


/**
 * Write one point into a record slot, zeroing any payload padding.
 *
 * The coordinates always sit at the front of the record, so raising
 * `record_bytes` to model a wider tuple never moves them.
 */
inline void EncodeRecord(char* dst, const Point& point, size_t record_bytes){
    std::memset(dst, 0, record_bytes);
    std::memcpy(dst, point.elements_, sizeof(double_t)*Constants::DIM);
}


/**
 * Read one point back out of a record slot.
 *
 * This is a `memcpy` rather than `*reinterpret_cast<const Point*>(src)` for
 * two reasons, both of which bite silently:
 *
 *   1. Strict aliasing. No Point object was ever constructed at `src` -- those
 *      bytes arrived through a char buffer and a pread. Reading them through a
 *      Point* permits the optimizer to assume the two pointers cannot refer to
 *      the same storage, and at -O2 GCC acts on that assumption: it may hoist
 *      or reorder the load with respect to the read that filled the page. The
 *      resulting wrong answers appear only once optimization is enabled.
 *
 *   2. Alignment. Point holds doubles, which want 8-byte alignment, but a
 *      record begins at `page + r*record_bytes`. That is only 8-byte aligned
 *      when record_bytes is a multiple of 8 -- set it to 20 to model a 20-byte
 *      tuple and record 1 starts misaligned. On x86 that is merely slow; on
 *      other architectures it faults. memcpy has no alignment precondition,
 *      which is precisely why record_bytes is free to be any width here.
 *
 * The safety is free: compilers treat a small fixed-size memcpy as a builtin,
 * so 16 bytes becomes two 8-byte moves -- the same instructions the unsafe
 * cast would have emitted. (At -O0 it is a real call, which is one more reason
 * the page-miss count rather than wall-clock is the metric to lead with.)
 */
inline Point DecodeRecord(const char* src){
    Point point;
    std::memcpy(point.elements_, src, sizeof(double_t)*Constants::DIM);
    return point;
}


#endif
