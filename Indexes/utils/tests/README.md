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
g++ -std=c++17 -I.. page_layout_test.cpp  -o page_layout_test
g++ -std=c++17 -I.. paged_writer_test.cpp -o paged_writer_test
```

## Running

```bash
# geometry and record encoding — no disk access, no environment needed
./page_layout_test

# writer + on-disk layout — needs the paged backend switched on and a scratch dir
mkdir -p /tmp/bs
ENABLE_PAGED_BACKEND=1 TEMP_BLOCKSTORE_DIR=/tmp/bs ./paged_writer_test
```

Both print `PASS` / `FAILED` and exit non-zero on failure.

## What each covers

| Test | Checks |
|---|---|
| `page_layout_test` | records-per-page arithmetic, `ceil` block→page division at the page boundaries, geometry validation, dead-tail accounting, and bit-exact record round-trips over `{16, 20, 24, 144}`-byte records — including ±0.0, denormals, ±inf, NaN and `DBL_MAX`. The 20-byte width is there on purpose: it leaves records unaligned, which is what the `memcpy` decode exists to tolerate. |
| `paged_writer_test` | Builds a real `BlockStore` with ragged blocks (0, 1, 255, 256, 257, 700 points — straddling the records-per-page boundary in both directions), then verifies the header round-trips, the block directory is gapless and covers every data page, every stored point reads back byte-identical through raw `pread`, tail pages are zero-padded past their last record, and all three backends return identical query results. |

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
