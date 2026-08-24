#!/usr/bin/env python3
"""Per-index analysis figures from the <n>.jsonl / bp_<n>.jsonl pair.

Complements plot_backend_comparison.py, which aggregates across indexes. These
five figures keep the index identity:

  1. backend_latency_by_index   in-memory vs mmap vs buffer pool, per index
  2. index_ranking              page misses per query, per index, per budget
  3. why_latency_tracks_misses  the scatter + the I/O share behind it
  4. budget_sensitivity         what shrinking the memory budget costs
  5. build_cost                 learning vs workload-awareness vs construction

Also writes index_analysis.csv, the table view of every plotted number.

Usage:
    python3 Results/plot_index_analysis.py <ResultsFolder> [outdir]
"""

from __future__ import annotations

import csv
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# --- design tokens -----------------------------------------------------------
# Reused verbatim from plot_backend_comparison.py so the two scripts render as
# one system. That palette carries a recorded validation (all-pairs pairlist,
# light mode: worst CVD dE 9.2 deutan against a target of 8, worst normal-vision
# dE 16.3 against a floor of 15). No new hues are introduced here -- the backend
# series below are slots 1/2/3 of the same validated set, and a subset of an
# all-pairs-validated set is still all-pairs valid.
SURFACE = "#fcfcfb"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#d8d7d2"

TYPE_COLOR = {"GRID": "#2a78d6", "SPACE": "#eb6834", "DATA": "#1baf7a", "ORDER": "#4a3aa7"}
TYPE_ORDER = ["GRID", "SPACE", "DATA", "ORDER"]
INDEX_TYPE = {"CUR": "DATA", "FLOOD": "GRID", "RSTAR": "DATA", "STR": "DATA",
              "GRID": "GRID", "ZIndex": "ORDER", "ZM": "ORDER", "WAZI": "ORDER",
              "KD": "SPACE", "QD": "SPACE", "RSMI": "DATA", "RW": "DATA"}

# Backends are an identity encoding, so fixed order, never cycled.
BACKEND_COLOR = {"in-memory": "#2a78d6", "mmap": "#eb6834", "buffer pool": "#1baf7a"}

# Build-cost phases. The two components R4 asked about carry hue; plain structural
# work recedes to neutral, because it is the control rather than the finding. A
# separate legend from the family encoding, so reusing slots 1/2 does not collide.
PHASE_COLOR = {"learning": "#eb6834", "workload-awareness": "#2a78d6",
               "construction": "#c9c8c3", "serialization": "#8a8984"}
PHASE_KEY = [("learning", "build_learning_s"), ("workload-awareness", "build_workload_awareness_s"),
             ("construction", "build_construct_s"), ("serialization", "build_serialize_s")]

COLUMN_FIT = 241.14749 / 72.26999
PAGE_FIT = 506.295 / 72.26999


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "text.color": INK,
        "xtick.color": INK_2, "ytick.color": INK_2,
        "grid.color": GRID, "grid.linewidth": 0.6,
        "font.size": 8, "axes.titlesize": 9, "axes.titleweight": "bold",
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "figure.dpi": 160,
    })


def recessive(ax, axis="x") -> None:
    ax.grid(True, axis=axis, linewidth=0.6, color=GRID, zorder=0)
    ax.set_axisbelow(True)


def load(results_dir: Path):
    main, bp = [], []
    for path in sorted(results_dir.glob("*.jsonl")):
        rows = [json.loads(line) for line in path.open() if line.strip()]
        (bp if path.name.startswith("bp_") else main).extend(rows)
    if not main or not bp:
        raise SystemExit(f"need both *.jsonl and bp_*.jsonl in {results_dir}")
    return main, bp


def med(rows, key):
    return st.median([r[key] for r in rows])


def save(fig, out: Path) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(out.with_suffix("." + ext), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}.{{png,pdf}}")


def layout(fig, bottom=0.10, top=0.90) -> None:
    """Reserve strips for the suptitle and the legend before either is drawn.

    Without this the suptitle lands on the panel titles and the figure legend
    lands on the x-axis labels -- bbox_inches="tight" grows the canvas but does
    not move anything apart.
    """
    fig.tight_layout(rect=[0, bottom, 1, top])


def family_legend(fig, ncol=4, y=0.015) -> None:
    fig.legend(handles=[Patch(facecolor=TYPE_COLOR[t], label=t) for t in TYPE_ORDER],
               loc="lower center", ncol=ncol, bbox_to_anchor=(0.5, y))


# --- figure 1 ----------------------------------------------------------------
def figure_backend_latency(by_model, by_model_frac, fraction, out: Path) -> None:
    """In-memory and mmap differ by a few percent; the pool is two orders away."""
    models = sorted(by_model, key=lambda m: med(by_model[m], "query_latency"))
    y = range(len(models))
    h = 0.26

    fig, ax = plt.subplots(figsize=(PAGE_FIT * 0.62, 0.34 * len(models) + 1.5))
    series = [
        ("in-memory", lambda m: med(by_model[m], "query_latency") / 1000),
        ("mmap", lambda m: med(by_model[m], "disk_backed_query_latency") / 1000),
        ("buffer pool", lambda m: med(by_model_frac[(m, fraction)], "bp_warm_query_latency") / 1000),
    ]
    for si, (name, fn) in enumerate(series):
        vals = [fn(m) for m in models]
        offs = [i + (1 - si) * h for i in y]
        ax.barh(offs, vals, height=h * 0.88, color=BACKEND_COLOR[name],
                label=name, zorder=3, linewidth=0)

    # One selective label per index: the multiple that matters, not 36 numbers.
    for i, m in enumerate(models):
        inmem = med(by_model[m], "query_latency") / 1000
        pool = med(by_model_frac[(m, fraction)], "bp_warm_query_latency") / 1000
        ax.text(pool * 1.25, i - h, f"{pool / inmem:.0f}×", va="center",
                ha="left", fontsize=7, color=INK_2)

    ax.set_yticks(list(y))
    ax.set_yticklabels(models)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlim(20, 34000)
    ax.set_xlabel("median query latency (µs, log)")
    ax.set_title("Only the buffer pool actually reaches the device", loc="left")
    recessive(ax)
    fig.suptitle(f"Query latency by storage backend  (pool at {fraction:g} of index file)",
                 x=0.005, ha="left", fontsize=10, fontweight="bold")
    layout(fig, bottom=0.11, top=0.93)
    # Below the panel, not inside it: at this aspect every interior corner is
    # either bar or value label.
    fig.legend(handles=[Patch(facecolor=BACKEND_COLOR[n], label=n) for n, _ in series],
               loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.015))
    save(fig, out)


# --- figure 2 ----------------------------------------------------------------
def figure_index_ranking(by_model_frac, fractions, out: Path) -> None:
    """Page misses are the deterministic metric -- rank the indexes on it."""
    models = sorted({m for m, _ in by_model_frac},
                    key=lambda m: med(by_model_frac[(m, fractions[0])], "bp_page_misses_per_query"))
    n = len(fractions)
    fig, axes = plt.subplots(1, n, figsize=(PAGE_FIT, 3.4), sharey=True)

    for ax, fr in zip(axes, fractions):
        vals = [med(by_model_frac[(m, fr)], "bp_page_misses_per_query") for m in models]
        ax.barh(range(len(models)), vals, height=0.68, zorder=3, linewidth=0,
                color=[TYPE_COLOR[INDEX_TYPE[m]] for m in models])
        for i, v in enumerate(vals):
            ax.text(v + max(vals) * 0.02, i, f"{v:.0f}", va="center",
                    fontsize=6.5, color=INK_2)
        ax.set_title(f"budget = {fr:g} of file", loc="left")
        ax.set_xlabel("page misses per query")
        ax.set_xlim(0, max(vals) * 1.18)
        recessive(ax)

    axes[0].set_yticks(range(len(models)))
    axes[0].set_yticklabels(models)
    axes[0].invert_yaxis()
    fig.suptitle("Index ranking by page misses, and how it holds as memory shrinks",
                 x=0.005, ha="left", fontsize=10, fontweight="bold")
    layout(fig, bottom=0.13, top=0.90)
    family_legend(fig)
    save(fig, out)


# --- figure 3 ----------------------------------------------------------------
def spearman(a, b):
    def rank(x):
        order = sorted(range(len(x)), key=lambda i: x[i])
        r = [0] * len(x)
        for pos, i in enumerate(order):
            r[i] = pos
        return r
    ra, rb = rank(a), rank(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((ra[i] - ma) * (rb[i] - mb) for i in range(n))
    den = (sum((x - ma) ** 2 for x in ra) * sum((x - mb) ** 2 for x in rb)) ** 0.5
    return num / den if den else float("nan")


def figure_latency_vs_misses(by_model_frac, fractions, out: Path) -> None:
    """Under O_DIRECT, latency is misses times the device cost -- and little else."""
    models = sorted({m for m, _ in by_model_frac})
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(PAGE_FIT, 3.3))

    marks = {fractions[0]: "o", fractions[1]: "s", fractions[2]: "^"}
    for fr in fractions:
        xs = [med(by_model_frac[(m, fr)], "bp_page_misses_per_query") for m in models]
        ys = [med(by_model_frac[(m, fr)], "bp_warm_query_latency") / 1000 for m in models]
        ax_l.scatter(xs, ys, s=34, marker=marks[fr], zorder=3, linewidth=0.8,
                     edgecolor=SURFACE,
                     color=[TYPE_COLOR[INDEX_TYPE[m]] for m in models],
                     label=f"{fr:g}  (ρ={spearman(xs, ys):.3f})")
    ax_l.set_xlabel("page misses per query")
    ax_l.set_ylabel("warm query latency (µs)")
    ax_l.set_title("Latency is a straight function of misses", loc="left")
    ax_l.legend(title="budget fraction", loc="upper left", fontsize=7, title_fontsize=7)
    recessive(ax_l, axis="both")

    # Right: how much of the query is I/O, which is why the left panel is a line.
    shares = []
    for m in models:
        rows = by_model_frac[(m, fractions[1])]
        shares.append(100 * st.median([r["bp_page_misses_per_query"] * r["device_ns_per_page_p50"]
                                       / r["bp_warm_query_latency"] for r in rows]))
    order = sorted(range(len(models)), key=lambda i: shares[i])
    ax_r.barh([models[i] for i in order], [shares[i] for i in order], height=0.68,
              zorder=3, linewidth=0,
              color=[TYPE_COLOR[INDEX_TYPE[models[i]]] for i in order])
    for k, i in enumerate(order):
        ax_r.text(shares[i] - 1.2, k, f"{shares[i]:.0f}%", va="center", ha="right",
                  fontsize=6.5, color=SURFACE, fontweight="bold")
    ax_r.set_xlim(0, 100)
    ax_r.set_xlabel("share of query time spent on page reads (%)")
    ax_r.set_title(f"Because I/O is nearly all of it (budget {fractions[1]:g})", loc="left")
    recessive(ax_r)

    fig.suptitle("Why the direct-I/O latency ranking and the miss ranking agree",
                 x=0.005, ha="left", fontsize=10, fontweight="bold")
    layout(fig, bottom=0.14, top=0.90)
    family_legend(fig)
    save(fig, out)


# --- figure 4 ----------------------------------------------------------------
def figure_budget_sensitivity(by_model_frac, fractions, out: Path) -> None:
    """Which indexes degrade fastest when the budget is cut."""
    models = sorted({m for m, _ in by_model_frac})
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(PAGE_FIT, 3.3))
    xs = sorted(fractions)

    for m in models:
        col = TYPE_COLOR[INDEX_TYPE[m]]
        ax_l.plot(xs, [med(by_model_frac[(m, f)], "bp_page_misses_per_query") for f in xs],
                  color=col, linewidth=2, marker="o", markersize=4, zorder=3, alpha=0.85)
        ax_r.plot(xs, [med(by_model_frac[(m, f)], "bp_warm_hit_rate") * 100 for f in xs],
                  color=col, linewidth=2, marker="o", markersize=4, zorder=3, alpha=0.85)

    # Direct-label the extremes only -- 12 labels on each panel would be noise.
    worst = max(models, key=lambda m: med(by_model_frac[(m, xs[0])], "bp_page_misses_per_query"))
    best = min(models, key=lambda m: med(by_model_frac[(m, xs[0])], "bp_page_misses_per_query"))
    for m in (worst, best):
        v = med(by_model_frac[(m, xs[0])], "bp_page_misses_per_query")
        ax_l.annotate(m, (xs[0], v), textcoords="offset points", xytext=(9, 0),
                      fontsize=7, color=INK_2, va="center", ha="left")

    for ax, lab, title in ((ax_l, "page misses per query", "Misses rise as the budget shrinks"),
                           (ax_r, "warm hit rate (%)", "Hit rate falls in lockstep")):
        ax.set_xscale("log")
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{x:g}" for x in xs])
        # Budget shrinks left to right, so the lines climb in the direction the
        # titles read. Plotting it the other way had every panel title running
        # against its own slope.
        ax.set_xlim(max(xs) * 1.5, min(xs) * 0.45)
        ax.set_xlabel("buffer pool size (fraction of index file)")
        ax.set_ylabel(lab)
        ax.set_title(title, loc="left")
        recessive(ax, axis="both")

    fig.suptitle("Cost of a smaller memory budget", x=0.005, ha="left",
                 fontsize=10, fontweight="bold")
    layout(fig, bottom=0.14, top=0.90)
    family_legend(fig)
    save(fig, out)


# --- figure 5 ----------------------------------------------------------------
def figure_build_cost(by_model, out: Path) -> None:
    """What R4 asked for: which part of build time is learning, which is workload."""
    models = sorted(by_model, key=lambda m: med(by_model[m], "build_total_s"))
    y = list(range(len(models)))

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(PAGE_FIT, 3.6))

    # Left: absolute seconds. Linear, not log -- a stacked bar on a log axis
    # encodes segment size by position rather than length and cannot be read.
    for ax, normalize in ((ax_l, False), (ax_r, True)):
        left = [0.0]*len(models)
        for label, key in PHASE_KEY:
            vals = [med(by_model[m], key) for m in models]
            if normalize:
                tot = [med(by_model[m], "build_total_s") for m in models]
                vals = [100*v/t if t else 0.0 for v, t in zip(vals, tot)]
            ax.barh(y, vals, left=left, height=0.68, zorder=3, linewidth=0,
                    color=PHASE_COLOR[label], label=label if not normalize else None)
            left = [a+b for a, b in zip(left, vals)]
        recessive(ax)

    ax_l.set_xlabel("build time (s)")
    ax_l.set_title("Absolute cost", loc="left")
    for i, m in enumerate(models):
        t = med(by_model[m], "build_total_s")
        ax_l.text(t + 3, i, f"{t:.0f}s", va="center", fontsize=6.5, color=INK_2)
    ax_l.set_xlim(0, max(med(by_model[m], "build_total_s") for m in models)*1.16)

    ax_r.set_xlim(0, 100)
    ax_r.set_xlabel("share of build time (%)")
    ax_r.set_title("Composition", loc="left")

    for ax in (ax_l, ax_r):
        ax.set_yticks(y)
        ax.set_yticklabels(models)
        ax.invert_yaxis()
    ax_r.tick_params(labelleft=False)

    fig.suptitle("Where build time goes: learning, workload-awareness, or plain construction",
                 x=0.005, ha="left", fontsize=10, fontweight="bold")
    layout(fig, bottom=0.16, top=0.90)
    fig.legend(handles=[Patch(facecolor=PHASE_COLOR[l], label=l) for l, _ in PHASE_KEY],
               loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.015))
    save(fig, out)

# --- table view --------------------------------------------------------------
def write_table(by_model, by_model_frac, fractions, out: Path) -> None:
    """The table view the accessibility pass requires -- every plotted number."""
    with out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["index", "family", "in_memory_us", "mmap_us", "fraction",
                    "pool_warm_us", "misses_per_query", "pages_per_query",
                    "warm_hit_rate", "io_share_pct",
                    "build_total_s", "build_learning_s", "build_workload_aware_s",
                    "build_construct_s", "build_serialize_s", "build_accounting"])
        for m in sorted(by_model):
            for fr in fractions:
                rows = by_model_frac[(m, fr)]
                w.writerow([
                    m, INDEX_TYPE[m],
                    f"{med(by_model[m], 'query_latency') / 1000:.1f}",
                    f"{med(by_model[m], 'disk_backed_query_latency') / 1000:.1f}",
                    f"{fr:g}",
                    f"{med(rows, 'bp_warm_query_latency') / 1000:.1f}",
                    f"{med(rows, 'bp_page_misses_per_query'):.1f}",
                    f"{med(rows, 'bp_pages_requested_per_query'):.1f}",
                    f"{med(rows, 'bp_warm_hit_rate'):.3f}",
                    f"{100 * st.median([r['bp_page_misses_per_query'] * r['device_ns_per_page_p50'] / r['bp_warm_query_latency'] for r in rows]):.1f}",
                    f"{med(by_model[m], 'build_total_s'):.2f}",
                    f"{med(by_model[m], 'build_learning_s'):.2f}",
                    f"{med(by_model[m], 'build_workload_awareness_s'):.2f}",
                    f"{med(by_model[m], 'build_construct_s'):.2f}",
                    f"{med(by_model[m], 'build_serialize_s'):.2f}",
                    by_model[m][0]["build_accounting"],
                ])
    print(f"wrote {out}")


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    results_dir = Path(sys.argv[1])
    outdir = Path(sys.argv[2]) if len(sys.argv) > 2 else results_dir.parent / "figures"
    outdir.mkdir(parents=True, exist_ok=True)

    main_rows, bp_rows = load(results_dir)
    by_model = defaultdict(list)
    for r in main_rows:
        by_model[r["model"]].append(r)
    by_model_frac = defaultdict(list)
    for r in bp_rows:
        by_model_frac[(r["model"], r["bufferpool_fraction"])].append(r)

    fractions = sorted({f for _, f in by_model_frac}, reverse=True)
    style()

    figure_backend_latency(by_model, by_model_frac, fractions[1],
                           outdir / "backend_latency_by_index")
    figure_index_ranking(by_model_frac, fractions, outdir / "index_ranking")
    figure_latency_vs_misses(by_model_frac, fractions, outdir / "why_latency_tracks_misses")
    figure_budget_sensitivity(by_model_frac, fractions, outdir / "budget_sensitivity")
    figure_build_cost(by_model, outdir / "build_cost")
    write_table(by_model, by_model_frac, fractions, outdir / "index_analysis.csv")


if __name__ == "__main__":
    main()
