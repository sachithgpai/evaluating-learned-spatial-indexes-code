#!/usr/bin/env python3
"""Two figures comparing mmap against the managed buffer pool.

ARCHIVE-ONLY. These figures need an mmap latency per index, and the evaluator no
longer runs an mmap pass -- that was the point of the comparison, and having made
it, paying for a second full copy of every index on disk to keep re-making it was
not worth it. The script therefore reads the previous two-file layout
(`<n>.jsonl` + `bp_<n>.jsonl`, with `disk_backed_query_latency` and separate
cold/warm columns), which now lives in `OLD_ResultsFolder/`.

It is kept rather than deleted because it is the evidence for a design decision
the paper states: that the page cache cannot be given a memory budget, and what
replacing it costs. Do not point it at a current ResultsFolder -- the numbers are
not the same measurement (that pass was warm, the current one is cold).

Usage:
    python3 Results/plot_backend_comparison.py Experiments/<dataset>/OLD_ResultsFolder [outdir]
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# --- design tokens -----------------------------------------------------------
# Categorical slots 1/2/3/7 of the reference palette. Validated as a set under
# the all-pairs pairlist in light mode: worst CVD dE 9.2 (deutan, target >= 8),
# worst normal-vision dE 16.3 (floor 15). Aqua sits at 2.74:1 on this surface,
# below the 3:1 bar, so the relief rule applies -- every series is direct-labelled
# and every series is direct-labelled in the figures themselves.
SURFACE = "#fcfcfb"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#d8d7d2"
TYPE_COLOR = {"GRID": "#2a78d6", "SPACE": "#eb6834", "DATA": "#1baf7a", "ORDER": "#4a3aa7"}
TYPE_ORDER = ["GRID", "SPACE", "DATA", "ORDER"]
INDEX_TYPE = {"CUR": "DATA", "FLOOD": "GRID", "RSTAR": "DATA", "STR": "DATA",
              "GRID": "GRID", "ZIndex": "ORDER", "ZM": "ORDER", "WAZI": "ORDER",
              "KD": "SPACE", "QD": "SPACE", "RSMI": "DATA", "RW": "DATA"}

COLUMN_FIT = 241.14749 / 72.26999
PAGE_FIT = 506.295 / 72.26999


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": GRID,
        "xtick.color": INK_2, "ytick.color": INK_2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.alpha": 0.9,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 8, "axes.titlesize": 9, "axes.titleweight": "bold",
        "legend.frameon": False, "figure.dpi": 200, "lines.linewidth": 2.0,
        "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
    })


def load(results_dir: Path):
    results, bp = [], []
    for path in sorted(results_dir.glob("*.jsonl")):
        rows = [json.loads(line) for line in path.open() if line.strip()]
        (bp if path.name.startswith("bp_") else results).extend(rows)
    if not bp:
        raise SystemExit(
            f"no bp_*.jsonl in {results_dir}.\n"
            "This script reads the archived two-file layout only -- point it at "
            "OLD_ResultsFolder/. Current results nest their buffer-pool rows inside "
            "each index row and carry no mmap pass; see Results/results_io.py.")
    if not results:
        raise SystemExit(f"no index rows (*.jsonl) in {results_dir}")
    mmap_lat = {r["model"]: r["disk_backed_query_latency"] for r in results}
    by_model = defaultdict(list)
    for row in bp:
        by_model[row["model"]].append(row)
    for rows in by_model.values():
        rows.sort(key=lambda r: -r["bufferpool_fraction"])
    return mmap_lat, by_model


def type_legend(ax, loc="best"):
    ax.legend(handles=[Line2D([], [], color=TYPE_COLOR[t], lw=2, label=t) for t in TYPE_ORDER],
              loc=loc, fontsize=7, labelspacing=0.3, handlelength=1.4)


def type_legend_below(fig):
    """Horizontal legend under the panels -- no data region to collide with."""
    fig.legend(handles=[Line2D([], [], color=TYPE_COLOR[t], lw=2, label=t) for t in TYPE_ORDER],
               loc="lower center", ncol=4, fontsize=7.5, handlelength=1.4,
               columnspacing=1.8, bbox_to_anchor=(0.5, -0.02))


def figure_one(by_model, out: Path) -> None:
    """What separates the indexes is pages touched -- not cache efficiency."""
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(PAGE_FIT, 2.9))

    # Left: pages requested per query. Identical at every budget, so take the first.
    pairs = sorted(((m, rows[0]["bp_pages_requested_per_query"]) for m, rows in by_model.items()),
                   key=lambda kv: kv[1])
    names = [m for m, _ in pairs]
    values = [v for _, v in pairs]
    colors = [TYPE_COLOR[INDEX_TYPE[m]] for m in names]
    bars = ax_l.barh(names, values, color=colors, height=0.68)
    ax_l.bar_label(bars, fmt="%.0f", padding=3, fontsize=7, color=INK_2)
    ax_l.set_xlim(0, max(values) * 1.18)
    ax_l.set_xlabel("pages requested per query")
    ax_l.set_title("Pages a query must touch", loc="left")
    ax_l.grid(axis="y", visible=False)
    ax_l.tick_params(axis="y", length=0)

    # Right: hit rate against budget -- every index lands on the same curve.
    for model, rows in by_model.items():
        ax_r.plot([r["bufferpool_fraction"] for r in rows],
                  [100 * r["bp_warm_hit_rate"] for r in rows],
                  color=TYPE_COLOR[INDEX_TYPE[model]], lw=1.6, alpha=0.85,
                  marker="o", markersize=4, markeredgecolor=SURFACE, markeredgewidth=1.2)
    fractions = sorted({r["bufferpool_fraction"] for rows in by_model.values() for r in rows})
    for fraction in fractions:
        band = [100 * r["bp_warm_hit_rate"] for rows in by_model.values()
                for r in rows if r["bufferpool_fraction"] == fraction]
        ax_r.annotate(f"{min(band):.0f}–{max(band):.0f}%", (fraction, max(band)),
                      textcoords="offset points", xytext=(0, 8), ha="center",
                      fontsize=7, color=INK_2)
    ax_r.set_xscale("log")
    ax_r.set_xticks(fractions)
    ax_r.set_xticklabels([f"{f:g}" for f in fractions])
    ax_r.set_ylim(0, 60)
    ax_r.set_xlabel("buffer pool size (fraction of index file)")
    ax_r.set_ylabel("cache hit rate (%)")
    ax_r.set_ylim(0, 100)
    ax_r.set_title("All indexes cache equally well", loc="left")

    fig.suptitle("Index quality shows up as pages touched, not as cache efficiency",
                 x=0.005, ha="left", fontsize=10, fontweight="bold")
    type_legend_below(fig)
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    fig.tight_layout(rect=[0, 0.10, 1, 0.90])
    for ext in ("png", "pdf"):
        fig.savefig(out.with_suffix(f".{ext}"), bbox_inches="tight")
    plt.close(fig)


def figure_two(mmap_lat, by_model, out: Path) -> None:
    """Miss counts are deterministic; wall-clock is not."""
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(PAGE_FIT, 3.0))
    fractions = sorted({r["bufferpool_fraction"] for rows in by_model.values() for r in rows})
    tight = min(fractions)

    # Only the separated series get a direct label; the rest converge into a band
    # and are carried by the legend instead of a pile of overlapping text.
    at_tight = {m: rows[-1]["bp_page_misses_per_query"] for m, rows in by_model.items()}
    ranked = sorted(at_tight.items(), key=lambda kv: -kv[1])
    labelled = {ranked[0][0], ranked[1][0], ranked[-1][0]}
    band = [v for m, v in ranked if m not in labelled]

    for model, rows in by_model.items():
        color = TYPE_COLOR[INDEX_TYPE[model]]
        xs = [r["bufferpool_fraction"] for r in rows]
        ax_l.plot(xs, [r["bp_page_misses_per_query"] for r in rows], color=color, lw=1.6,
                  alpha=0.9 if model in labelled else 0.5, marker="o", markersize=4,
                  markeredgecolor=SURFACE, markeredgewidth=1.2)
        ax_r.plot(xs, [r["bp_warm_query_latency"] / mmap_lat[model] for r in rows], color=color,
                  lw=1.6, alpha=0.75, marker="o", markersize=4,
                  markeredgecolor=SURFACE, markeredgewidth=1.2)
        if model in labelled:
            ax_l.annotate(model, (tight, at_tight[model]), textcoords="offset points",
                          xytext=(7, 0), ha="left", va="center", fontsize=7.5, color=INK)

    ax_l.annotate("%d indexes\nwithin %.0f-%.0f" % (len(band), min(band), max(band)),
                  (tight, sum(band) / len(band)), textcoords="offset points",
                  xytext=(7, -2), ha="left", va="center", fontsize=7, color=INK_2)

    for ax in (ax_l, ax_r):
        ax.set_xscale("log")
        ax.set_xticks(fractions)
        ax.set_xticklabels(["%g" % f for f in fractions])
        ax.set_xlim(tight * 0.78, max(fractions) * 1.6)
        ax.set_xlabel("buffer pool size (fraction of index file)")
    ax_l.set_ylabel("page misses per query")
    ax_l.set_title("Miss counts: monotone and ordered", loc="left")

    ax_r.axhline(1.0, color=INK_2, lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax_r.annotate("parity with mmap", (tight * 0.85, 1.0), textcoords="offset points",
                  xytext=(0, -12), ha="left", va="center", fontsize=7, color=INK_2)
    ax_r.set_ylabel("latency relative to mmap")
    ax_r.set_ylim(0.6, 4.5)
    ax_r.set_title("Wall-clock: crossing and non-monotone", loc="left")

    fig.suptitle("Why we report page misses rather than latency",
                 x=0.005, ha="left", fontsize=10, fontweight="bold")
    type_legend_below(fig)
    fig.tight_layout(rect=(0, 0.06, 1, 0.92))
    fig.tight_layout(rect=[0, 0.10, 1, 0.90])
    for ext in ("png", "pdf"):
        fig.savefig(out.with_suffix("." + ext), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    results_dir = Path(sys.argv[1])
    outdir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(".")
    outdir.mkdir(parents=True, exist_ok=True)

    style()
    mmap_lat, by_model = load(results_dir)
    results_rows = [json.loads(line) for path in sorted(results_dir.glob("*.jsonl"))
                    if not path.name.startswith("bp_") for line in path.open() if line.strip()]
    figure_three(results_rows, mmap_lat, by_model, outdir / "backend_mmap_vs_bufferpool")
    figure_one(by_model, outdir / "backend_pages_vs_hitrate")
    figure_two(mmap_lat, by_model, outdir / "backend_misses_vs_latency")
    print(f"wrote {outdir}/backend_pages_vs_hitrate.{{png,pdf}}")
    print(f"wrote {outdir}/backend_misses_vs_latency.{{png,pdf}}")
    print(f"wrote {outdir}/backend_mmap_vs_bufferpool.{{png,pdf}}")




# --- appended: the mmap-vs-buffer-pool comparison -----------------------------
MMAP_C, POOL_C = "#2a78d6", "#eb6834"   # validated all-pairs: CVD dE 24.7, normal 33.6


def figure_three(results_rows, mmap_lat, by_model, out: Path) -> None:
    """What replacing mmap with a managed buffer pool costs, and what it buys."""
    import statistics as st

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(PAGE_FIT, 3.2))
    fractions = sorted({r["bufferpool_fraction"] for rows in by_model.values() for r in rows})
    file_mb = st.median(rows[0]["index_file_bytes"] for rows in by_model.values()) / 1e6

    def at(fraction, key):
        return [r[key] for rows in by_model.values() for r in rows if r["bufferpool_fraction"] == fraction]

    pool_mb = [st.median(at(f, "bufferpool_bytes")) / 1e6 for f in fractions]
    pool_us = [st.median(at(f, "bp_warm_query_latency")) / 1e3 for f in fractions]
    pool_ms = [st.median(at(f, "bp_page_misses_per_query")) for f in fractions]
    mmap_us = st.median(mmap_lat.values()) / 1e3
    mem_us = st.median(r["query_latency"] for r in results_rows) / 1e3

    # ---- left: what it costs -------------------------------------------------
    ax_l.axhline(mem_us, color=INK_2, lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax_l.annotate(f"in-memory, no file: {mem_us:.0f} µs", (0.95, mem_us),
                  textcoords="offset points", xytext=(2, 5), ha="left", va="bottom",
                  fontsize=7, color=INK_2)

    for f, mb in zip(fractions, pool_mb):
        ys = [v / 1e3 for v in at(f, "bp_warm_query_latency")]
        ax_l.scatter([mb] * len(ys), ys, s=13, color=POOL_C, alpha=0.28,
                     edgecolors="none", zorder=2)
    ax_l.scatter([file_mb] * len(mmap_lat), [v / 1e3 for v in mmap_lat.values()],
                 s=13, color=MMAP_C, alpha=0.28, edgecolors="none", zorder=2)

    ax_l.plot(pool_mb, pool_us, color=POOL_C, lw=2, marker="o", markersize=7,
              markeredgecolor=SURFACE, markeredgewidth=1.6, zorder=3, label="buffer pool")
    ax_l.scatter([file_mb], [mmap_us], s=95, color=MMAP_C, marker="D", zorder=4,
                 edgecolors=SURFACE, linewidths=1.6, label="mmap")

    for f, mb, us in zip(fractions, pool_mb, pool_us):
        ax_l.annotate(f"{f:g} of file\n{mb:.0f} MB", (mb, us), textcoords="offset points",
                      xytext=(0, 11), ha="center", fontsize=6.8, color=INK_2)
    ax_l.annotate(f"whole file\n{file_mb:.0f} MB\n(OS decides)", (file_mb, mmap_us),
                  textcoords="offset points", xytext=(14, 0), ha="left", va="center",
                  fontsize=6.8, color=INK_2, linespacing=1.4)

    ax_l.set_xscale("log")
    ax_l.set_xlim(0.85, file_mb * 6.5)
    ax_l.set_ylim(0, max(pool_us) * 1.45)
    ax_l.set_xlabel("memory the data cache may use (MB, log)")
    ax_l.set_ylabel("query latency (µs, median)")
    ax_l.set_title("The cost: latency vs memory", loc="left")
    ax_l.legend(loc="upper right", fontsize=7.5, handlelength=1.4)

    # ---- right: what it buys -------------------------------------------------
    ax_r.plot(pool_mb, pool_ms, color=POOL_C, lw=2, marker="o", markersize=7,
              markeredgecolor=SURFACE, markeredgewidth=1.6, zorder=3)
    for f, mb in zip(fractions, pool_mb):
        ys = at(f, "bp_page_misses_per_query")
        ax_r.scatter([mb] * len(ys), ys, s=13, color=POOL_C, alpha=0.28,
                     edgecolors="none", zorder=2)
    for mb, ms in zip(pool_mb, pool_ms):
        ax_r.annotate(f"{ms:.0f}", (mb, ms), textcoords="offset points", xytext=(0, 10),
                      ha="center", fontsize=7.5, color=INK)

    ax_r.axvspan(file_mb / 3.0, file_mb * 6.5, color=MMAP_C, alpha=0.08, zorder=0)
    ax_r.annotate("mmap\nnot measurable —\nthe kernel reports\nno transfer count",
                  (file_mb * 1.25, max(pool_ms) * 0.50), ha="center", va="center",
                  fontsize=7.2, color=MMAP_C, linespacing=1.5)

    ax_r.set_xscale("log")
    ax_r.set_xlim(0.85, file_mb * 6.5)
    ax_r.set_ylim(0, max(pool_ms) * 1.3)
    ax_r.set_xlabel("memory the data cache may use (MB, log)")
    ax_r.set_ylabel("page misses per query")
    ax_r.set_title("The gain: I/O becomes countable", loc="left")

    fig.suptitle("Replacing mmap with a managed buffer pool", x=0.005, ha="left",
                 fontsize=10, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.tight_layout(rect=[0, 0.10, 1, 0.90])
    for ext in ("png", "pdf"):
        fig.savefig(out.with_suffix("." + ext), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
