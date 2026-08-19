#!/usr/bin/env python3
"""Join the results and buffer-pool JSONL files and compare the two disk backends.

The evaluator writes two files per task:
  <line>.jsonl     one row per index, including the mmap pass (disk_backed_*)
  bp_<line>.jsonl  one row per (index, buffer-pool fraction)

They are deliberately separate -- bp rows vary along bufferpool_fraction, which is
not part of the grouping key plot_results.py uses, so merging them into one file
would silently average the fractions together. This script does the join instead.

Usage:
    python3 compare_mmap_vs_bufferpool.py <ResultsFolder> [--model KD] [--csv out.csv]
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

JOIN_KEYS = ("model", "block_size", "data_sample_num",
             "dataset_entropy_id", "query_entropy_id", "selectivity")


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def collect(results_dir: Path) -> tuple[list[dict], list[dict]]:
    results, bp_rows = [], []
    for path in sorted(results_dir.glob("*.jsonl")):
        (bp_rows if path.name.startswith("bp_") else results).extend(load_jsonl(path))
    return results, bp_rows


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--model", help="restrict to one index")
    parser.add_argument("--csv", type=Path, help="also write the joined rows here")
    args = parser.parse_args()

    results, bp_rows = collect(args.results_dir)
    if not results:
        raise SystemExit(f"No results *.jsonl found in {args.results_dir}")
    if not bp_rows:
        raise SystemExit(f"No bp_*.jsonl found in {args.results_dir}. "
                         "Was ENABLE_PAGED_BACKEND=1 set when the tasks ran?")

    # mmap side, keyed for the join
    mmap_by_key = {tuple(row[k] for k in JOIN_KEYS): row for row in results}

    joined, unmatched = [], 0
    for bp in bp_rows:
        key = tuple(bp[k] for k in JOIN_KEYS)
        base = mmap_by_key.get(key)
        if base is None:
            unmatched += 1
            continue
        queries = max(bp["bp_warm_pages_requested"], 1)
        joined.append({
            **{k: bp[k] for k in JOIN_KEYS},
            "fraction": bp["bufferpool_fraction"],
            "frames": bp["bufferpool_frames"],
            "floored": bp["bufferpool_frames_floored"],
            "mmap_latency_ns": base["disk_backed_query_latency"],
            "bp_warm_latency_ns": bp["bp_warm_query_latency"],
            "bp_cold_latency_ns": bp["bp_cold_query_latency"],
            "misses_per_query": bp["bp_page_misses_per_query"],
            "requests_per_query": bp["bp_pages_requested_per_query"],
            "hit_rate": bp["bp_warm_hit_rate"],
            "bytes_read": bp["bp_warm_bytes_read"],
            "results_match": bp.get("results_match", True),
            "index_file_bytes": bp["index_file_bytes"],
        })

    if args.model:
        joined = [row for row in joined if row["model"] == args.model]
    if not joined:
        raise SystemExit("Nothing to report after filtering.")

    mismatches = [row for row in joined if not row["results_match"]]
    print(f"joined {len(joined)} rows"
          + (f"  ({unmatched} bp rows had no matching results row)" if unmatched else "")
          + (f"  ** {len(mismatches)} ROWS WITH results_match=false **" if mismatches else ""))

    # Aggregate over everything except model and fraction.
    buckets = defaultdict(list)
    for row in joined:
        buckets[(row["model"], row["fraction"])].append(row)

    print()
    print(f"{'model':8}{'frac':>8}{'frames':>8}{'flr':>5}"
          f"{'mmap ns':>10}{'bp warm ns':>12}{'ratio':>8}"
          f"{'miss/q':>9}{'req/q':>8}{'hit%':>7}")
    print("-" * 85)
    for model in sorted({row["model"] for row in joined}):
        fractions = sorted({f for m, f in buckets if m == model}, reverse=True)
        for fraction in fractions:
            rows = buckets[(model, fraction)]
            mmap_ns = mean([r["mmap_latency_ns"] for r in rows])
            bp_ns = mean([r["bp_warm_latency_ns"] for r in rows])
            print(f"{model:8}{fraction:>8}{int(mean([r['frames'] for r in rows])):>8}"
                  f"{('y' if any(r['floored'] for r in rows) else ''):>5}"
                  f"{mmap_ns:>10.0f}{bp_ns:>12.0f}{(bp_ns/mmap_ns if mmap_ns else float('nan')):>8.2f}"
                  f"{mean([r['misses_per_query'] for r in rows]):>9.1f}"
                  f"{mean([r['requests_per_query'] for r in rows]):>8.1f}"
                  f"{100*mean([r['hit_rate'] for r in rows]):>7.1f}")
        print()

    # The headline ranking: misses at the tightest budget, which is deterministic.
    tightest = min(row["fraction"] for row in joined)
    ranking = sorted(
        ((model, mean([r["misses_per_query"] for r in buckets[(model, tightest)]]))
         for model in sorted({row["model"] for row in joined})),
        key=lambda pair: pair[1])
    print(f"Ranking by page misses per query at fraction={tightest} "
          f"(deterministic, machine-independent):")
    for position, (model, misses) in enumerate(ranking, 1):
        print(f"  {position:2}. {model:8} {misses:9.1f}")

    if args.csv:
        import csv
        with args.csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(joined[0]))
            writer.writeheader()
            writer.writerows(joined)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
