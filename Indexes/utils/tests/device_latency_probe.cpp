/**
 * Standalone page-miss cost probe.
 *
 * Reports what a single-page random read costs on whatever storage
 * TEMP_BLOCKSTORE_DIR points at, buffered and direct, side by side. The gap
 * between the two columns is the whole reason the paged backend can now bypass
 * the page cache: if they are close, the "disk" being measured is RAM.
 *
 *   g++ -std=c++17 -O2 device_latency_probe.cpp -o device_latency_probe.out
 *   TEMP_BLOCKSTORE_DIR=/tmp/$USER/probe/ ./device_latency_probe.out
 *
 * Exits nonzero if the direct probe could not run, so a batch script can use it
 * as a precondition check before committing a task to direct-I/O measurement.
 */

#include <cstdio>
#include <string>

#include "../device_probe.h"


static void Report(const char* label, const DeviceProbeResult& probe){
    if(!probe.ran){
        std::printf("  %-10s FAILED: %s\n", label, probe.error.c_str());
        return;
    }
    std::printf("  %-10s mean %8.0f   p50 %8.0f   p90 %8.0f   p99 %8.0f ns   (n=%zu)\n",
                label, probe.mean_ns, probe.p50_ns, probe.p90_ns, probe.p99_ns, probe.samples);
}


int main(int argc, char** argv){
    const size_t page_bytes = (argc > 1) ? std::stoul(argv[1]) : 4096;
    const std::string dir = ResolveBlockstoreDir();

    std::printf("blockstore dir : %s\n", dir.c_str());
    std::printf("page bytes     : %zu\n\n", page_bytes);

    const DeviceProbeResult buffered = ProbeDeviceLatency(dir, page_bytes, 256ull<<20, 4000, false);
    const DeviceProbeResult direct   = ProbeDeviceLatency(dir, page_bytes, 256ull<<20, 4000, true);

    Report("buffered", buffered);
    Report("O_DIRECT", direct);

    if(buffered.ran && direct.ran)
        std::printf("\n  direct/buffered ratio: %.1fx\n", direct.mean_ns/buffered.mean_ns);

    if(!direct.ran){
        std::printf("\nDirect I/O is unavailable here; BUFFER_POOL_DIRECT_IO=1 would abort at Build().\n");
        return 1;
    }

    std::printf("\n{\"probe_dir\":\"%s\",\"page_bytes\":%zu,"
                "\"direct_mean_ns\":%.1f,\"direct_p50_ns\":%.1f,\"direct_p99_ns\":%.1f,"
                "\"buffered_mean_ns\":%.1f}\n",
                dir.c_str(), page_bytes, direct.mean_ns, direct.p50_ns, direct.p99_ns,
                buffered.ran ? buffered.mean_ns : -1.0);
    return 0;
}
