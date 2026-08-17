# Storage-backend unit tests

Small, self-contained checks for the paged block store. They need no dataset and
no trained models, so they run in seconds and are the fast inner loop while
working on `page_layout.h`, `paged_disk_backend.h` and (later) `buffer_pool.h`.

## Building

Each test is its own **single translation unit** and must be compiled
separately. `constants.h` defines non-`inline` globals at namespace scope, which
only works because the project has exactly one TU — linking two of these
together would produce duplicate-symbol errors.

```bash
cd Indexes/utils/tests
for t in page_layout_test buffer_pool_test paged_writer_test paged_backend_test; do
    g++ -std=c++17 -I.. $t.cpp -o $t
done
```

## Running

```bash
# no disk access, no environment needed
./page_layout_test

# the rest need a scratch dir; the two that build a BlockStore also need the gate
mkdir -p /tmp/bs
TEMP_BLOCKSTORE_DIR=/tmp/bs ./buffer_pool_test
ENABLE_PAGED_BACKEND=1 TEMP_BLOCKSTORE_DIR=/tmp/bs ./paged_writer_test
ENABLE_PAGED_BACKEND=1 TEMP_BLOCKSTORE_DIR=/tmp/bs ./paged_backend_test
```

Each prints `PASS` / `FAILED` and exits non-zero on failure.

## What each covers

| Test | Checks |
|---|---|
| `page_layout_test` | records-per-page arithmetic, `ceil` block→page division at the page boundaries, geometry validation, dead-tail accounting, and bit-exact record round-trips over `{16, 20, 24, 144}`-byte records — including ±0.0, denormals, ±inf, NaN and `DBL_MAX`. The 20-byte width is there on purpose: it leaves records unaligned, which is what the `memcpy` decode exists to tolerate. |
| `buffer_pool_test` | The pool on its own, against a synthetic file where page *i* holds byte pattern *i*: correct frame contents, all-hits when the pool fits everything, pure thrash at one frame, the 1,2,1,3,2 case that separates LRU from FIFO, pinned pages surviving churn, over-pinning throwing rather than corrupting, and `Clear()` returning to cold. The important one is the **whole hit/miss sequence cross-checked against an independently written LRU** over 10,000 random accesses at five different frame counts. |
| `paged_writer_test` | Builds a real `BlockStore` with ragged blocks (0, 1, 255, 256, 257, 700 points — straddling the records-per-page boundary in both directions), then verifies the header round-trips, the block directory is gapless and covers every data page, every stored point reads back byte-identical through raw `pread` (bypassing the pool), tail pages are zero-padded, and all three backends return identical query results. |
| `paged_backend_test` | End to end on a real `KDTree` over 200k clustered points: all three backends byte-identical across 200 queries, then a budget sweep (fraction 1.0 → 0.001) asserting the three invariants — `fraction=1.0` warms up to **zero** misses, `pages_requested` is identical at every fraction, and misses are monotone as the pool shrinks. Also checks both floor modes and that results stay correct with a 2-frame pool. |

## Environment variables

| Variable | Effect |
|---|---|
| `ENABLE_PAGED_BACKEND=1` | Build the paged file at all. Off by default, so a normal experiment run is untouched. |
| `TEMP_BLOCKSTORE_DIR` | Where scratch files are written. Defaults to `$PROJECT_ROOT/temp_blockstore/`. |
| `KEEP_BLOCKSTORE_FILES=1` | Skip the immediate `unlink`, so the paged file keeps its name and can be inspected while the process runs. |

## A note on the paged file

`PagedDiskBackend::Build()` unlinks the file as soon as it is written, keeping
only the open descriptor. That means `ls` shows nothing during a run even though
`df` still counts the space — this is expected, not a leak, and it is what makes
the space reclaim automatically when a task is killed. Use
`KEEP_BLOCKSTORE_FILES=1` when you need to look at the bytes.
