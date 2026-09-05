#ifndef DEVICE_PROBE_H
#define DEVICE_PROBE_H

/**
 * @file device_probe.h
 * @brief Measures what one page miss costs on the storage this run will use.
 *
 * A page-miss count is machine-independent; the time a miss takes is not. This
 * turns the second one into a logged number instead of an assumption, which is
 * what makes a measured latency interpretable after the fact.
 *
 * The specific thing it protects against: $TMPDIR is a per-job *directory* on a
 * filesystem shared by every job on the node, not a private device or a quota'd
 * volume. The site documents its capacity as possibly "shared with other jobs
 * or users on the same node", and there is no per-job bandwidth reservation --
 * so on a 384-core node, concurrent tasks contend for one NVMe queue. A task
 * that landed beside a dozen noisy neighbours then produces latencies that look
 * like a property of the index and are a property of the queue. With the probe
 * recorded alongside every measurement, a contended task is identifiable rather
 * than merely suspected.
 *
 * Method: write a scratch file large enough to defeat the device's own DRAM
 * cache, fsync it, drop it from the page cache, then time single-page random
 * O_DIRECT reads at queue depth one -- exactly the shape of a buffer-pool miss.
 */

#include <fcntl.h>
#include <unistd.h>
#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include"storage_backend.h"
#include"sort_tools.h"


struct DeviceProbeResult{
    bool   ran{false};
    bool   direct{false};
    double mean_ns{0.0};
    double p50_ns{0.0};
    double p90_ns{0.0};
    double p99_ns{0.0};
    size_t samples{0};
    size_t page_bytes{0};
    uint64_t file_bytes{0};
    bool   page_cache_dropped{false};  // direct probes only; see the drop below
    std::string error;          // empty when the probe succeeded
};


/**
 * Time single-page random reads against `dir`.
 *
 * Never throws: a probe that cannot run is a missing diagnostic, not a reason
 * to lose an evaluation task. Failures come back in `error` with `ran = false`.
 *
 * @param file_bytes scratch file size. The default is deliberately larger than
 *        an SSD's on-board DRAM cache -- a small file would report the cache's
 *        latency rather than the medium's.
 */
inline DeviceProbeResult ProbeDeviceLatency(const std::string& dir,
                                            size_t page_bytes = 4096,
                                            uint64_t file_bytes = 256ull<<20,
                                            size_t samples = 4000,
                                            bool direct = true){
    DeviceProbeResult result;
    result.direct     = direct;
    result.page_bytes = page_bytes;
    result.file_bytes = file_bytes;
    result.samples    = samples;

    if(page_bytes == 0 || (direct && page_bytes % 4096 != 0)){
        result.error = "page_bytes must be a nonzero multiple of 4096 for a direct probe";
        return result;
    }

    const uint64_t page_count = file_bytes/page_bytes;
    if(page_count < 2){
        result.error = "file_bytes is too small to sample";
        return result;
    }

    std::error_code dir_error;
    std::filesystem::create_directories(dir, dir_error);
    const std::string path = dir+"devprobe_"+generate_random_alphanumeric_string(16);

    int fd = open(path.c_str(), O_RDWR|O_CREAT|O_EXCL, 0600);
    if(fd < 0){
        result.error = "cannot create "+path+": "+std::string(strerror(errno));
        return result;
    }

    // Anything unlinked here is gone for good; the descriptor keeps it alive
    // until we close it, exactly as the paged store does.
    std::remove(path.c_str());

    void* buffer = nullptr;
    if(posix_memalign(&buffer, 4096, page_bytes) != 0 || buffer == nullptr){
        close(fd);
        result.error = "cannot allocate an aligned probe buffer";
        return result;
    }
    std::memset(buffer, 0xA5, page_bytes);

    for(uint64_t page=0;page<page_count;page++){
        if(pwrite(fd, buffer, page_bytes, off_t(page*page_bytes)) != ssize_t(page_bytes)){
            std::free(buffer); close(fd);
            result.error = "short write while laying out the probe file: "+std::string(strerror(errno));
            return result;
        }
    }
    if(fsync(fd) != 0){
        std::free(buffer); close(fd);
        result.error = "fsync failed: "+std::string(strerror(errno));
        return result;
    }
    // Only the direct probe drops the writer's pages. The buffered probe is
    // meant to reproduce what the buffered backend actually experiences, and
    // there the store was written moments earlier and is still cached -- that
    // residency is the effect being measured, not a contaminant.
    //
    // Not fatal when it does not happen: the reads below are O_DIRECT and bypass
    // the cache regardless, so an undropped cache costs the probe RAM rather than
    // accuracy. Recorded so that a probe taken where the cache cannot be purged
    // is identifiable afterwards instead of indistinguishable.
    if(direct){
        const PageCacheDropStatus drop = DropFileFromPageCache(fd);
        result.page_cache_dropped = (drop == PageCacheDropStatus::kDropped);
    }

    int read_fd = fd;
    if(direct){
#ifdef O_DIRECT
        // Reopening through /proc/self/fd keeps this working after the unlink
        // above: the path no longer exists, but the descriptor still names it.
        read_fd = open(("/proc/self/fd/"+std::to_string(fd)).c_str(), O_RDONLY|O_DIRECT);
        if(read_fd < 0){
            std::free(buffer); close(fd);
            result.error = "cannot open the probe file O_DIRECT: "+std::string(strerror(errno));
            return result;
        }
#else
        std::free(buffer); close(fd);
        result.error = "O_DIRECT is not available on this platform";
        return result;
#endif
    }

    std::mt19937_64 generator(20260819ull);
    std::uniform_int_distribution<uint64_t> pick(0, page_count-1);

    std::vector<double> timings;
    timings.reserve(samples);
    bool failed = false;

    for(size_t s=0;s<samples;s++){
        const off_t offset = off_t(pick(generator)*page_bytes);
        timespec start{}, stop{};
        clock_gettime(CLOCK_MONOTONIC, &start);
        const ssize_t n = pread(read_fd, buffer, page_bytes, offset);
        clock_gettime(CLOCK_MONOTONIC, &stop);
        if(n != ssize_t(page_bytes)){
            result.error = "probe read failed: "+std::string(strerror(errno));
            failed = true;
            break;
        }
        timings.push_back(double(stop.tv_sec-start.tv_sec)*1e9 +
                          double(stop.tv_nsec-start.tv_nsec));
    }

    std::free(buffer);
    if(read_fd != fd) close(read_fd);
    close(fd);

    if(failed || timings.empty())
        return result;

    std::sort(timings.begin(), timings.end());
    double sum = 0.0;
    for(double value: timings) sum += value;

    result.ran     = true;
    result.samples = timings.size();
    result.mean_ns = sum/double(timings.size());
    result.p50_ns  = timings[timings.size()/2];
    result.p90_ns  = timings[(timings.size()*9)/10];
    result.p99_ns  = timings[(timings.size()*99)/100];
    return result;
}


/** Probe the directory the block stores actually live in. */
inline DeviceProbeResult ProbeBlockstoreDevice(size_t page_bytes, bool direct){
    return ProbeDeviceLatency(ResolveBlockstoreDir(), page_bytes,
                              256ull<<20, 4000, direct);
}


#endif
