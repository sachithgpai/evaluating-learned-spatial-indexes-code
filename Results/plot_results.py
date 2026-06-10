from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator
import numpy as np
import pandas as pd
import seaborn as sb
from sklearn.metrics import top_k_accuracy_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from tqdm import tqdm

try:
    import colorcet as cc

    COLOR_SEQUENCE = cc.glasbey_dark
except ImportError:
    COLOR_SEQUENCE = sb.color_palette("tab20", n_colors=12).as_hex()


N_POINTS = 8_000_000
DEFAULT_SELECTIVITY = 0.1024
REDEMPTION_BASE_MODEL = "CUR"
MAX_USEFUL_LATENCY_RATIO = 1.5
INDEX_USEFULNESS_TOP_K = 3
INDEX_USEFULNESS_MAX_DEPTH = 2
INDEX_USEFULNESS_MAX_RULES_PER_INDEX = 2
INDEX_USEFULNESS_MIN_SAMPLES_LEAF = 10
INDEX_USEFULNESS_FEATURES = [
    "data_storage",
    "selectivity",
    "dataset_entropy_id",
    "query_entropy_id",
]
DECISION_TREE_FEATURES = [
    "dataset_entropy_id",
    "query_entropy_id",
    "selectivity",
    "data_storage",
]
DECISION_TREE_COMMON_PARAMS = {
    "random_state": 123,
    "splitter": "best",
    "criterion": "entropy",
    "class_weight": "balanced",
}
DECISION_TREE_SUBTREE_PARAMS = {
    "in-memory": {
        "max_depth": 4,
        "max_leaf_nodes": 5,
        "min_samples_leaf": 5,
        "ccp_alpha": 0.001,
    },
    "disk-backed": {
        "max_depth": 4,
        "max_leaf_nodes": 5,
        "min_samples_leaf": 5,
        "ccp_alpha": 0.001,
    },
}

VALIDATION_RESULT_FILENAME = "ValidationResults.json"
REAL_WORLD_VALIDATION_REPORT = "REAL_WORLD_VALIDATION_ANALYSIS.txt"
VALIDATION_BLOCK_LOOKUP_KEYS = [
    "model",
    "dataset_entropy_id",
    "query_entropy_id",
    "selectivity",
]
VALIDATION_GROUP_COLUMNS = [
    "data_sample_num",
    "dataset_entropy_id",
    "query_entropy_id",
    "selectivity",
]
VALIDATION_REGRET_THRESHOLDS = [1.10, 1.25, 1.50]
VALIDATION_WORST_CASES = 10

BLOCK_SIZE_ARRAY = [32, 64, 128, 256, 512, 1024, 2048, 4096]
SELECTIVITY_ARRAY = [0.0064, 0.0256, 0.1024, 0.4096, 1.6384]
MODEL_LIST = [
    "CUR",
    "FLOOD",
    "RSTAR",
    "STR",
    "GRID",
    "ZIndex",
    "ZM",
    "WAZI",
    "KD",
    "QD",
    "RSMI",
    "RW",
]
ORDER_LIST = [
    "GRID",
    "FLOOD",
    "KD",
    "QD",
    "STR",
    "CUR",
    "RSMI",
    "RSTAR",
    "RW",
    "ZIndex",
    "ZM",
    "WAZI",
]
INDEX_TYPE = {
    "CUR": "DATA",
    "FLOOD": "GRID",
    "RSTAR": "DATA",
    "STR": "DATA",
    "GRID": "GRID",
    "ZIndex": "ORDER",
    "ZM": "ORDER",
    "WAZI": "ORDER",
    "KD": "SPACE",
    "QD": "SPACE",
    "RSMI": "DATA",
    "RW": "DATA",
}
TYPE_ORDER = ["GRID", "SPACE", "DATA", "ORDER"]
TYPE_TITLES = [
    "Grid-based partition",
    "Space partition",
    "Data partition",
    "Order-based partition",
]

LATEX_MODEL_LABELS = {
    "CUR": r"\textsc{CUR}",
    "FLOOD": r"\textsc{Flood}",
    "RSTAR": r"\textsc{R\text{*}tree}",
    "STR": r"\textsc{STR}",
    "GRID": r"\textsc{GridFile}",
    "ZIndex": r"\textsc{Zindex}",
    "ZM": r"\textsc{ZMindex}",
    "WAZI": r"\textsc{WAZI}",
    "KD": r"\textsc{KDtree}",
    "QD": r"\textsc{QDtree}",
    "RSMI": r"\textsc{Rsmi}",
    "RW": r"\textsc{RWtree}",
}
PLAIN_MODEL_LABELS = {
    "CUR": "CUR",
    "FLOOD": "Flood",
    "RSTAR": "R*tree",
    "STR": "STR",
    "GRID": "GridFile",
    "ZIndex": "Zindex",
    "ZM": "ZMindex",
    "WAZI": "WAZI",
    "KD": "KDtree",
    "QD": "QDtree",
    "RSMI": "Rsmi",
    "RW": "RWtree",
}

PALETTE = dict(
    zip(MODEL_LIST, sb.color_palette(COLOR_SEQUENCE, n_colors=len(MODEL_LIST)).as_hex())
)
MARKERS = dict(zip(MODEL_LIST, ["o", "^", "v", "<", ">", "s", "D", "p", "h", "X", "H", "P"]))

COLUMN_FIT = 241.14749 / 72.26999
PAGE_FIT = 506.295 / 72.26999


@dataclass
class DecisionTreeSubtree:
    storage_value: int
    storage_label: str
    classifier: DecisionTreeClassifier
    feature_columns: list[str]
    training_accuracy: float
    training_weight_sum: float


@dataclass
class EntropyMappingSummary:
    classifier: DecisionTreeClassifier
    training_pairs: pd.DataFrame
    validation_pairs: pd.DataFrame
    validation_row_count: int


def configure_matplotlib(use_tex: bool) -> None:
    if use_tex:
        plt.rcParams["text.latex.preamble"] = (
            r"\RequirePackage[T1]{fontenc} "
            r"\RequirePackage[tt=true]{libertine} "
            r"\RequirePackage[varqu]{zi4} "
            r"\RequirePackage[libertine]{newtxmath}"
        )

    params = {
        "text.usetex": use_tex,
        "font.family": "libertine" if use_tex else "serif",
        "font.size": 7,
        "xtick.major.size": 2.5,
        "xtick.minor.size": 1.2,
        "ytick.major.size": 2.5,
        "ytick.minor.size": 1.2,
        "xtick.major.pad": 0,
        "ytick.major.pad": 0,
        "legend.handlelength": 1,
        "legend.columnspacing": 0.3,
        "legend.handletextpad": 0.2,
        "legend.borderpad": 0,
        "legend.framealpha": 0,
        "lines.linewidth": 0.75,
        "savefig.pad_inches": 0.02,
        "axes.labelpad": 0,
        "markers.fillstyle": "full",
        "lines.markeredgewidth": 0.1,
        "lines.markersize": 5,
        "axes.titlepad": 0.2,
    }
    plt.rcParams.update(params)


def sci_notation(number: float, sig_fig: int = 1) -> str:
    if number < 0:
        number = -number
    ret_string = f"{number:.{sig_fig}e}"
    significand, exponent = ret_string.split("e")
    return "$" + significand + r"\times 10^" + str(int(exponent)) + "$"


def load_results(result_path: Path, pickle_path: Path) -> pd.DataFrame:
    if pickle_path.is_file() and pickle_path.stat().st_mtime >= result_path.stat().st_mtime:
        return pd.read_pickle(pickle_path)

    rows = []
    with result_path.open() as json_file:
        for _, line in tqdm(enumerate(json_file), desc=f"Loading {result_path.name}"):
            data = json.loads(line)
            row = {}
            for feature in [
                "block_size",
                "build_time",
                "data_sample_num",
                "dataset_entropy",
                "dataset_entropy_id",
                "query_entropy",
                "query_entropy_id",
                "selectivity",
                "model",
                "query_latency",
                "result_size",
                "refinement_latency",
                "number_of_refined_blocks",
                "number_of_points_scanned",
                "disk_backed_result_size",
                "disk_backed_query_latency",
            ]:
                row[feature] = data[feature]

            row["avg_block_size"] = data["block_size_quantiles"][5]
            row["type"] = INDEX_TYPE[data["model"]]
            row["scan_latency"] = row["query_latency"] - row["refinement_latency"]
            row["query_latency_per_point"] = row["query_latency"] * 5000.0 / row["result_size"]
            row["disk_backed_query_latency_per_point"] = (
                row["disk_backed_query_latency"] * 5000.0 / row["result_size"]
            )
            row["disk_backed_scan_latency"] = (
                row["disk_backed_query_latency"] - row["refinement_latency"]
            )
            row["area_or_count_based"] = data["area_or_count_based"]
            rows.append(row)

    loaded_df = pd.DataFrame(rows)
    loaded_df["selectivity"] = loaded_df["selectivity"].astype(int) / 10000.0
    loaded_df.to_pickle(pickle_path)
    return loaded_df


def prepare_results(df: pd.DataFrame, query_gen_policy: str) -> pd.DataFrame:
    df = df[df["area_or_count_based"] == query_gen_policy].copy()
    df = (
        df.groupby(
            [
                "model",
                "block_size",
                "data_sample_num",
                "dataset_entropy_id",
                "query_entropy_id",
                "selectivity",
                "type",
                "area_or_count_based",
            ]
        )
        .mean(numeric_only=True)
        .reset_index()
    )
    df["target"] = 8.0 * 10e4 * df["selectivity"]
    df["difference_from_target"] = (df["target"] - (df["result_size"] / 5000)) / df["target"]
    return df


def compute_optimal_blocksize_dfs(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    optimal_setting_memory = df.copy()
    index = optimal_setting_memory.groupby(
        ["model", "data_sample_num", "dataset_entropy_id", "query_entropy_id", "selectivity"]
    )["query_latency"].idxmin()
    optimal_setting_memory = optimal_setting_memory.loc[index].reset_index(drop=True)

    optimal_setting_disk = df.copy()
    index = optimal_setting_disk.groupby(
        ["model", "data_sample_num", "dataset_entropy_id", "query_entropy_id", "selectivity"]
    )["disk_backed_query_latency"].idxmin()
    optimal_setting_disk = optimal_setting_disk.loc[index].reset_index(drop=True)

    optimal_measurable_memory = (
        df.copy()
        .groupby(
            ["model", "dataset_entropy_id", "query_entropy_id", "selectivity", "block_size", "type"]
        )
        .mean(numeric_only=True)
        .reset_index()
    )
    index = optimal_measurable_memory.groupby(
        ["model", "dataset_entropy_id", "query_entropy_id", "selectivity", "type"]
    )["query_latency"].idxmin()
    optimal_measurable_memory = optimal_measurable_memory.loc[index].reset_index(drop=True)

    optimal_measurable_disk = (
        df.copy()
        .groupby(
            ["model", "dataset_entropy_id", "query_entropy_id", "selectivity", "block_size", "type"]
        )
        .mean(numeric_only=True)
        .reset_index()
    )
    index = optimal_measurable_disk.groupby(
        ["model", "dataset_entropy_id", "query_entropy_id", "selectivity", "type"]
    )["disk_backed_query_latency"].idxmin()
    optimal_measurable_disk = optimal_measurable_disk.loc[index].reset_index(drop=True)

    return (
        optimal_setting_memory,
        optimal_setting_disk,
        optimal_measurable_memory,
        optimal_measurable_disk,
    )


def build_best_vs_average_df(
    df: pd.DataFrame, optimal_measurable_memory: pd.DataFrame
) -> pd.DataFrame:
    optimal_df = optimal_measurable_memory[
        optimal_measurable_memory["selectivity"] == DEFAULT_SELECTIVITY
    ].copy()
    optimal_df["optimal"] = "Block-optimal"

    average_df = df[df["selectivity"] == DEFAULT_SELECTIVITY].copy()
    average_df["optimal"] = "Block-averaged"

    combined_df = pd.concat([optimal_df, average_df], ignore_index=True)
    return combined_df.drop(
        columns=[
            "dataset_entropy_id",
            "query_entropy_id",
            "selectivity",
            "block_size",
            "data_sample_num",
            "build_time",
            "dataset_entropy",
            "query_entropy",
            "result_size",
            "refinement_latency",
            "number_of_refined_blocks",
            "number_of_points_scanned",
            "disk_backed_result_size",
            "disk_backed_query_latency",
            "avg_block_size",
            "scan_latency",
            "query_latency_per_point",
            "disk_backed_query_latency_per_point",
            "disk_backed_scan_latency",
            "target",
            "difference_from_target",
        ],
        errors="ignore",
    )


def build_data_query_entropy_df(optimal_measurable_memory: pd.DataFrame) -> pd.DataFrame:
    temp_df = optimal_measurable_memory.copy()
    temp_df = temp_df.drop(
        columns=[
            "block_size",
            "build_time",
            "data_sample_num",
            "dataset_entropy",
            "query_entropy",
            "result_size",
            "refinement_latency",
            "number_of_refined_blocks",
            "number_of_points_scanned",
            "disk_backed_result_size",
            "disk_backed_query_latency",
            "type",
            "scan_latency",
            "difference_from_target",
            "target",
            "avg_block_size",
            "disk_backed_query_latency_per_point",
            "disk_backed_scan_latency",
            "query_latency",
            "selectivity",
        ],
        errors="ignore",
    )
    return temp_df.groupby(["model", "dataset_entropy_id", "query_entropy_id"]).mean().reset_index()


def build_scan_refinement_df(optimal_measurable_memory: pd.DataFrame) -> pd.DataFrame:
    temp_df = optimal_measurable_memory.copy()
    temp_df = (
        temp_df.groupby(["model", "selectivity", "type"]).mean(numeric_only=True).reset_index()
    )
    temp_df["query_latency"] = temp_df["query_latency"] / 1_000_000
    temp_df["scan_latency"] = temp_df["scan_latency"] / 1_000_000
    temp_df["refinement_latency"] = temp_df["refinement_latency"] / 1_000_000
    return temp_df


def build_access_stats_df(optimal_measurable_memory: pd.DataFrame) -> pd.DataFrame:
    temp_df = optimal_measurable_memory.copy()
    temp_df["false_positives"] = temp_df["number_of_points_scanned"] - (
        temp_df["result_size"] / 5000.0
    )
    temp_df["percent_false_positives"] = temp_df["false_positives"] * 100 / N_POINTS
    temp_df["selectivity_cat"] = temp_df["selectivity"].astype("category")
    temp_df["percent_num_blocks"] = (
        temp_df["number_of_refined_blocks"] * 100 / (N_POINTS / temp_df["avg_block_size"])
    )
    temp_df["scan_latency_per_point"] = temp_df["scan_latency"] / temp_df["result_size"]
    temp_df["refinement_latency_per_block"] = (
        temp_df["refinement_latency"] / temp_df["number_of_refined_blocks"]
    )
    return temp_df


def build_disk_delay_df(df: pd.DataFrame) -> pd.DataFrame:
    temp_df = df.copy()
    temp_df["scan_latency_delay_ratios"] = (
        temp_df["disk_backed_query_latency"] - temp_df["refinement_latency"]
    ) / temp_df["scan_latency"]
    temp_df = temp_df[temp_df["scan_latency_delay_ratios"] < 7]
    temp_df = temp_df[temp_df["scan_latency_delay_ratios"] > 1]
    return temp_df


def build_build_time_latex_table(optimal_measurable_memory: pd.DataFrame) -> str:
    temp_df = optimal_measurable_memory[
        optimal_measurable_memory["selectivity"] == DEFAULT_SELECTIVITY
    ].copy()
    build_time_df = (
        temp_df.groupby("model")["build_time"].agg(["mean", "std"]).reset_index()
    )
    build_time_df["Coeff-Var-"] = build_time_df["std"]
    build_time_df = build_time_df.rename(
        columns={
            "model": "model-",
            "mean": "build_time-mean",
        }
    )
    return build_time_df[
        ["model-", "build_time-mean", "Coeff-Var-"]
    ].to_latex(index=False)


def build_redemption_latex_table(
    optimal_measurable_memory: pd.DataFrame,
    base_model: str = REDEMPTION_BASE_MODEL,
) -> str:
    temp_df = (
        optimal_measurable_memory.copy()
        .groupby(["model", "selectivity", "dataset_entropy_id", "query_entropy_id"])
        .mean(numeric_only=True)
        .reset_index()
    )
    temp_df = temp_df.drop(
        columns=[
            "block_size",
            "dataset_entropy",
            "query_entropy",
            "data_sample_num",
            "refinement_latency",
            "scan_latency",
            "result_size",
            "avg_block_size",
            "number_of_refined_blocks",
            "number_of_points_scanned",
            "disk_backed_result_size",
            "query_latency_per_point",
            "disk_backed_query_latency_per_point",
            "disk_backed_scan_latency",
            "target",
            "difference_from_target",
        ],
        errors="ignore",
    )

    redemption_frames = []
    for _, group_df in temp_df.groupby(["selectivity", "dataset_entropy_id", "query_entropy_id"]):
        if base_model not in set(group_df["model"]):
            continue

        group_df = group_df.set_index("model")
        base_query_latency = group_df.loc[base_model, "query_latency"]
        base_build_time = group_df.loc[base_model, "build_time"]
        base_disk_latency = group_df.loc[base_model, "disk_backed_query_latency"]

        group_df = group_df.drop(index=base_model).copy()
        group_df["build_time_faster_than_base"] = (
            group_df["build_time"] < base_build_time
        )
        group_df["query_latency_faster_than_base"] = (
            group_df["query_latency"] < base_query_latency
        )
        group_df["disk_backed_query_latency_faster_than_base"] = (
            group_df["disk_backed_query_latency"] < base_disk_latency
        )
        group_df["redemption"] = (
            1e9
            * (group_df["build_time"] - base_build_time)
            / (base_query_latency - group_df["query_latency"])
        )
        group_df["disk_redemption"] = (
            1e9
            * (group_df["build_time"] - base_build_time)
            / (base_disk_latency - group_df["disk_backed_query_latency"])
        )
        redemption_frames.append(group_df.reset_index())

    if not redemption_frames:
        raise ValueError(f"Could not build redemption table: missing base model {base_model!r}.")

    redemption_df = pd.concat(redemption_frames, ignore_index=True)
    redemption_df = redemption_df.groupby("model").mean(numeric_only=True).reset_index()
    redemption_df["model_idx"] = redemption_df["model"].map(
        {model: ix for ix, model in enumerate(ORDER_LIST)}
    )
    redemption_df = redemption_df.dropna(subset=["model_idx"])
    redemption_df = redemption_df.round(1)
    redemption_df = redemption_df.round(
        {
            "build_time_faster_than_base": 0,
            "query_latency_faster_than_base": 0,
            "disk_backed_query_latency_faster_than_base": 0,
        }
    )
    redemption_df = redemption_df.set_index("model_idx").sort_index()

    redemption_df["redemption_str"] = redemption_df.apply(
        _convert_to_redemption_str,
        axis=1,
    )
    redemption_df["disk_redemption_str"] = redemption_df.apply(
        _convert_to_disk_redemption_str,
        axis=1,
    )

    table_df = redemption_df[
        ["model", "build_time", "redemption_str", "disk_redemption_str"]
    ].T
    table_df.index = ["model", "build time", "redemption str", "disk redemption str"]
    return table_df.to_latex(index=True, header=False, escape=False)


def _convert_to_redemption_str(row: pd.Series) -> str:
    if row["build_time_faster_than_base"] > 0.5:
        if row["query_latency_faster_than_base"] > 0.5:
            return "(+)"
        return "(-)" + sci_notation(row["redemption"])

    if row["query_latency_faster_than_base"] > 0.5:
        return "(+)" + sci_notation(row["redemption"])
    return "(-)"


def _convert_to_disk_redemption_str(row: pd.Series) -> str:
    if row["build_time_faster_than_base"] > 0.5:
        if row["disk_backed_query_latency_faster_than_base"] > 0.5:
            return "(+)"
        return "(-)" + sci_notation(row["disk_redemption"])

    if row["disk_backed_query_latency_faster_than_base"] > 0.5:
        return "(+)" + sci_notation(row["disk_redemption"])
    return "(-)"


def write_latex_tables(
    optimal_measurable_memory: pd.DataFrame,
    figures_dir: Path,
) -> list[Path]:
    table_outputs = {
        "BuildTime_Table.txt": build_build_time_latex_table(optimal_measurable_memory),
        "BuildTime_Redemption_Table.txt": build_redemption_latex_table(
            optimal_measurable_memory
        ),
    }

    written_paths = []
    for filename, latex_table in table_outputs.items():
        output_path = figures_dir / filename
        output_path.write_text(latex_table, encoding="utf-8")
        written_paths.append(output_path)

    return written_paths


def build_decision_tree_inputs(
    optimal_setting_memory: pd.DataFrame,
    optimal_setting_disk: pd.DataFrame,
    top_k: int = INDEX_USEFULNESS_TOP_K,
) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    latency_weighted_data = build_latency_ranked_df(
        optimal_setting_memory,
        optimal_setting_disk,
        DECISION_TREE_FEATURES,
    )
    latency_weighted_data = latency_weighted_data[
        latency_weighted_data["latency_rank"] <= top_k
    ].copy()
    sample_weights = np.power(
        np.reciprocal(latency_weighted_data["latency_ratio"].to_numpy()), 2
    )
    latency_weighted_data = latency_weighted_data.drop(
        columns=[
            "data_sample_num",
            "query_latency",
            "best_query_latency",
            "latency_ratio",
            "latency_rank",
        ]
    )
    x_combined = latency_weighted_data.drop(columns=["model"])
    y_combined = latency_weighted_data["model"]
    return x_combined, y_combined, sample_weights


def build_latency_ranked_df(
    optimal_setting_memory: pd.DataFrame,
    optimal_setting_disk: pd.DataFrame,
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    memory_df = _query_latency_frame(
        optimal_setting_memory,
        latency_column="query_latency",
        data_storage=0,
        feature_columns=feature_columns,
    )
    disk_df = _query_latency_frame(
        optimal_setting_disk,
        latency_column="disk_backed_query_latency",
        data_storage=1,
        feature_columns=feature_columns,
    )

    ranked_df = pd.concat([memory_df, disk_df], ignore_index=True)
    ranked_df = _bucket_workload_features(ranked_df)
    group_columns = list(feature_columns) + ["data_sample_num"]
    latency_group = ranked_df.groupby(group_columns)["query_latency"]
    ranked_df["best_query_latency"] = latency_group.transform("min")
    ranked_df["latency_ratio"] = (
        ranked_df["query_latency"] / ranked_df["best_query_latency"]
    )
    ranked_df["latency_rank"] = latency_group.rank(method="first", ascending=True)
    return ranked_df


def _query_latency_frame(
    df: pd.DataFrame,
    latency_column: str,
    data_storage: int,
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    output_df = df.copy()
    output_df["data_storage"] = data_storage
    output_df = output_df[
        ["model", "data_sample_num", latency_column, *feature_columns]
    ]
    return output_df.rename(columns={latency_column: "query_latency"})


def _bucket_workload_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["dataset_entropy_id"] = df["dataset_entropy_id"].replace({1: 2, 5: 4})
    df["query_entropy_id"] = df["query_entropy_id"].replace({1: 2, 5: 4})
    df["selectivity"] = (
        df["selectivity"]
        .replace({0.0064: 1, 0.0256: 1, 0.1024: 2, 0.4096: 3, 1.6384: 3})
        .astype(int)
    )
    return df


def model_legend_handles(models: Sequence[str]) -> list[Line2D]:
    handles = []
    for model in models:
        handles.append(
            Line2D(
                [0],
                [0],
                color=PALETTE[model],
                marker=MARKERS[model],
                linestyle="",
                markerfacecolor="none",
                markeredgewidth=0.5,
                markersize=4,
            )
        )
    return handles


def save_figure(fig: plt.Figure, output_path: Path, dpi: int = 400) -> None:
    fig.savefig(output_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)


def plot_gran_block_size_selectivity(
    df: pd.DataFrame,
    optimal_setting_memory: pd.DataFrame,
    figures_dir: Path,
    model_labels: dict[str, str],
) -> None:
    labels_arr, handles_arr = [], []

    fig = plt.figure(figsize=(0.9 * PAGE_FIT, 0.9 * 4))
    ax1 = plt.subplot(3, 4, 1)
    ax2 = plt.subplot(3, 4, 2, sharey=ax1, sharex=ax1)
    ax3 = plt.subplot(3, 4, 3, sharey=ax1, sharex=ax1)
    ax4 = plt.subplot(3, 4, 4, sharey=ax1, sharex=ax1)
    ax1.set_ylim(top=0.45, bottom=0.075)

    ax21 = plt.subplot(3, 4, 5)
    ax22 = plt.subplot(3, 4, 6, sharey=ax21, sharex=ax21)
    ax23 = plt.subplot(3, 4, 7, sharey=ax21, sharex=ax21)
    ax24 = plt.subplot(3, 4, 8, sharey=ax21, sharex=ax21)
    legend_ax = plt.subplot(9, 1, 7)
    axs = [ax1, ax2, ax3, ax4]
    axs2 = [ax21, ax22, ax23, ax24]

    for ix, index_type in enumerate(TYPE_ORDER):
        temp_df = df[df["type"] == index_type].copy()
        temp_df = temp_df[temp_df["selectivity"] == DEFAULT_SELECTIVITY]
        temp_df = temp_df.groupby(["block_size", "model"]).mean(numeric_only=True).reset_index()
        temp_df["query_latency"] = temp_df["query_latency"] / 1_000_000

        ax = sb.lineplot(
            ax=axs[ix],
            data=temp_df,
            x="avg_block_size",
            y="query_latency",
            hue="model",
            errorbar=None,
            palette=PALETTE,
            markers=MARKERS,
            style="model",
            markerfacecolor="none",
            markeredgecolor=None,
            linewidth=0.3,
            markeredgewidth=0.5,
            dashes=False,
        )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_ylabel("Query Latency (ms)" if ix == 0 else None)
        ax.set_xlabel("Avg. Block Size")
        ax.set_xticks(BLOCK_SIZE_ARRAY[::2], labels=map(str, BLOCK_SIZE_ARRAY[::2]))
        ax.set_yticks([0.1, 0.2, 0.4], labels=map(str, [0.1, 0.2, 0.4]))
        ax.set_yticks([], minor=True)
        ax.set_title(TYPE_TITLES[ix], pad=1)

        handles, labels = ax.get_legend_handles_labels()
        ax.legend().set_visible(False)
        labels_arr += labels
        handles_arr += handles

    for ix, index_type in enumerate(TYPE_ORDER):
        temp_df = optimal_setting_memory[optimal_setting_memory["type"] == index_type].copy()
        temp_df = temp_df.groupby(["selectivity", "model"]).mean(numeric_only=True).reset_index()

        ax = sb.lineplot(
            ax=axs2[ix],
            data=temp_df,
            x="selectivity",
            y="avg_block_size",
            hue="model",
            errorbar=None,
            palette=PALETTE,
            markers=MARKERS,
            style="model",
            markerfacecolor="none",
            markeredgecolor=None,
            linewidth=0.3,
            markeredgewidth=0.5,
            dashes=False,
            legend=False,
        )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)
        ax.set_ylabel("Optimal block sizes" if ix == 0 else None)
        ax.set_xlabel("Selectivity")
        ax.set_xticks(SELECTIVITY_ARRAY[::2], labels=map(str, SELECTIVITY_ARRAY[::2]))

    legend_ax.axis("off")
    legend_ax.legend(
        handles_arr,
        [model_labels[label] for label in labels_arr],
        ncols=len(labels_arr),
        loc="center",
        columnspacing=1.2,
        handlelength=0.0,
        handletextpad=0.5,
    )

    plt.subplots_adjust(hspace=0.4, wspace=0.15)
    save_figure(fig, figures_dir / "GRAN_BlockSize_Selectivity.pdf")


def plot_gran_best_block_vs_average(
    best_vs_average_df: pd.DataFrame,
    figures_dir: Path,
    model_labels: dict[str, str],
) -> None:
    best_vs_average_df = best_vs_average_df.copy()
    best_vs_average_df["query_latency"] = best_vs_average_df["query_latency"] / 1_000_000

    fig, ax1 = plt.subplots(figsize=(COLUMN_FIT, 1))
    sb.barplot(
        ax=ax1,
        data=best_vs_average_df,
        x="optimal",
        y="query_latency",
        hue="model",
        palette=PALETTE,
        errorbar=("sd", 1),
        hue_order=ORDER_LIST,
    )
    handles, labels = ax1.get_legend_handles_labels()

    ax1.set_yscale("log")
    ax1.set_xlabel("")
    ax1.set_ylabel("Query Latency (ms)")
    ax1.set_ylim(bottom=0.055)
    ax1.yaxis.set_minor_locator(NullLocator())
    ax1.legend(
        handles,
        [model_labels[model] for model in labels],
        ncols=4,
        loc="upper left",
        columnspacing=0.5,
        handletextpad=0.1,
        labelspacing=0.2,
        borderaxespad=0.3,
        fontsize=7,
    )

    save_figure(fig, figures_dir / "GRAN_BestBlockVsAvg_barplot.pdf")


def plot_data_query_entropy_percent_change(
    data_query_entropy_df: pd.DataFrame,
    figures_dir: Path,
    model_labels: dict[str, str],
) -> None:
    grouped_df = data_query_entropy_df.groupby("model")
    fig, axs_arr = plt.subplots(
        2, 7, figsize=(0.9 * PAGE_FIT, 0.8 * 2.2), sharex=True, sharey=True
    )

    axs = []
    for row in range(2):
        for column in range(6):
            axs.append(axs_arr[row][column])

    axs_arr[0][6].set_axis_off()
    axs_arr[1][6].set_axis_off()

    for ix, model in enumerate(ORDER_LIST):
        model_df = grouped_df.get_group(model)
        arr = model_df.pivot(
            columns="dataset_entropy_id",
            index="query_entropy_id",
            values="query_latency_per_point",
        ).values[::-1]
        base = arr[4][0]
        arr = (arr - base) / base * 100

        ax = sb.heatmap(
            data=arr,
            ax=axs[ix],
            cmap="coolwarm",
            cbar=False,
            vmax=15,
            vmin=-15,
            center=0,
        )
        ax.set_aspect("equal")
        ax.set_title(model_labels[model], fontsize=7)
        ax.set_xticks([0.5, 1.5, 2.5, 3.5, 4.5], labels=map(str, [1, 2, 3, 4, 5]))
        ax.set_yticks([4.5, 3.5, 2.5, 1.5, 0.5], labels=map(str, [1, 2, 3, 4, 5]))

    cmap = cm.coolwarm
    norm = colors.Normalize(vmin=-200, vmax=200)
    cbar_legend_ax = fig.add_subplot(1, 28, 25)
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_legend_ax, orientation="vertical")

    fig.supylabel("Query Skew", fontsize=7, x=0.085)
    fig.supxlabel("Data Skew", fontsize=7, y=-0.005, x=0.45)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    save_figure(fig, figures_dir / "DATAQUERYENTROPY_percent_change_per_index.pdf")


def plot_analysis_refinement_scan(
    scan_refinement_df: pd.DataFrame,
    figures_dir: Path,
    model_labels: dict[str, str],
) -> None:
    fig = plt.figure(figsize=(PAGE_FIT, 2.4))

    ax1 = plt.subplot(2, 4, 1)
    ax1.set_ylim(bottom=0.00085, top=0.06)
    ax1.set_xlim(left=0.006, right=1.5)
    ax2 = plt.subplot(2, 4, 2, sharey=ax1, sharex=ax1)
    ax3 = plt.subplot(2, 4, 3, sharey=ax1, sharex=ax1)
    ax4 = plt.subplot(2, 4, 4, sharey=ax1, sharex=ax1)
    axs = [ax1, ax2, ax3, ax4]

    for ix, index_type in enumerate(TYPE_ORDER):
        temp_df = scan_refinement_df[scan_refinement_df["type"] == index_type].copy()
        sb.lineplot(
            ax=axs[ix],
            data=temp_df,
            x="scan_latency",
            y="refinement_latency",
            hue="model",
            errorbar=None,
            palette=PALETTE,
            markers=MARKERS,
            style="model",
            markerfacecolor="none",
            markeredgecolor=None,
            linewidth=0.4,
            markeredgewidth=0.5,
            dashes=False,
            legend=False,
            style_order=ORDER_LIST,
            orient="x",
        )
        axs[ix].set_title(TYPE_TITLES[ix], fontsize=7)
        axs[ix].set_xscale("log")
        axs[ix].set_yscale("log")
        axs[ix].set_ylabel(None)
        axs[ix].set_xlabel("Scan (ms)", fontsize=7)

    axs[0].set_ylabel("Refinement (ms)", fontsize=7)

    legend_ax = plt.subplot(2, 1, 2)
    legend_ax.axis("off")
    legend_ax.legend(
        model_legend_handles(ORDER_LIST),
        [model_labels[model] for model in ORDER_LIST],
        ncols=12,
        loc="upper center",
        columnspacing=1,
        handlelength=0.0,
        handletextpad=0.6,
        fontsize=7,
    )
    plt.subplots_adjust(hspace=0.25, wspace=0.225)
    save_figure(fig, figures_dir / "ANALYSIS_Refinement_Scan_lineplot_Grouped_PAGEFIT.pdf")


def plot_access_patterns(
    access_stats_df: pd.DataFrame,
    figures_dir: Path,
    model_labels: dict[str, str],
    use_tex: bool,
) -> None:
    plt.close("all")
    access_stats_df = access_stats_df.copy()
    latency_columns = ["scan_latency_per_point", "refinement_latency_per_block"]
    access_stats_df[latency_columns] = access_stats_df[latency_columns] / 1_000_000

    fig1, axs1 = plt.subplots(3, 4, sharex=True, sharey=True, figsize=(0.48 * PAGE_FIT, 2))
    fig2, axs2 = plt.subplots(3, 4, sharex=True, sharey=True, figsize=(0.48 * PAGE_FIT, 2))

    selectivity_categories = list(access_stats_df["selectivity_cat"].cat.categories)
    selectivity_palette = dict(
        zip(selectivity_categories, sb.color_palette("flare", n_colors=len(selectivity_categories)))
    )
    legend_handles = [
        Line2D([0], [0], marker="o", color=color, linestyle="", markersize=4)
        for color in selectivity_palette.values()
    ]
    legend_labels = [str(selectivity) for selectivity in selectivity_categories]

    for ix, model in enumerate(ORDER_LIST):
        temp_df = access_stats_df[access_stats_df["model"] == model]
        ax1 = sb.scatterplot(
            ax=axs1[ix // 4][ix % 4],
            data=temp_df,
            x="percent_false_positives",
            y="scan_latency_per_point",
            hue="selectivity_cat",
            palette=selectivity_palette,
            legend=False,
            style="model",
            lw=0,
            # s=1.5,
            s=7.5,
        )
        ax1.set_title(model_labels[model], fontsize=7)
        ax1.set_xlabel(None)
        ax1.set_ylabel(None)
        ax1.tick_params(which="minor", labelbottom=False, labelleft=False)

        ax2 = sb.scatterplot(
            ax=axs2[ix // 4][ix % 4],
            data=temp_df,
            x="percent_num_blocks",
            y="refinement_latency_per_block",
            hue="selectivity_cat",
            palette=selectivity_palette,
            legend=False,
            style="model",
            lw=0,
            # s=1.5,
            s=7.5,
        )
        ax2.set_title(model_labels[model], fontsize=7)
        ax2.set_xlabel(None)
        ax2.set_ylabel(None)
        ax2.tick_params(which="minor", labelbottom=False, labelleft=False)

    axs1[0][0].set_xscale("log")
    axs1[0][0].set_yscale("log")
    axs1[0][0].set_xlim(right=0.95, left=0.005)
    

    percent_label = r"\%" if use_tex else "%"
    fig1.subplots_adjust(hspace=0.375, wspace=0.175)
    fig1.supxlabel(f"{percent_label} False Positives Points", fontsize=7.25)
    fig1.supylabel("Scan Latency Per Result Point (ms)", fontsize=7.25)
    save_figure(fig1, figures_dir / "ANALYSIS_FalsePositve_ScanLatency.pdf")

    axs2[0][0].set_xscale("log")
    axs2[0][0].set_yscale("log")
    axs2[0][0].set_ylim(top=0.007, bottom=0.000005)

    fig2.supxlabel(f"{percent_label} Blocks Accessed Per Query", fontsize=7.25)
    fig2.supylabel("Refinement Latency Per Block (ms)", fontsize=7.25)
    fig2.subplots_adjust(hspace=0.375, wspace=0.175)
    save_figure(fig2, figures_dir / "ANALYSIS_PercentBlocksAccessed_RefinementLatency.pdf")

    legend_fig, legend_ax = plt.subplots(1, 1, figsize=(PAGE_FIT, 0.1))
    legend_ax.legend(
        legend_handles,
        legend_labels,
        ncols=len(legend_labels),
        loc="center",
        columnspacing=3,
        handlelength=0.0,
        handletextpad=1,
    )
    legend_ax.axis("off")
    save_figure(legend_fig, figures_dir / "ANALYSIS_Scan_Refinement_Legend.pdf")


def plot_gran_block_size_selectivity_disk_backed(
    df: pd.DataFrame,
    optimal_setting_disk: pd.DataFrame,
    figures_dir: Path,
    model_labels: dict[str, str],
) -> None:
    labels_arr, handles_arr = [], []
    fig = plt.figure(figsize=(0.9 * PAGE_FIT, 0.9 * 4))
    ax1 = plt.subplot(3, 4, 1)
    ax2 = plt.subplot(3, 4, 2, sharey=ax1, sharex=ax1)
    ax3 = plt.subplot(3, 4, 3, sharey=ax1, sharex=ax1)
    ax4 = plt.subplot(3, 4, 4, sharey=ax1, sharex=ax1)

    ax21 = plt.subplot(3, 4, 5)
    ax22 = plt.subplot(3, 4, 6, sharey=ax21, sharex=ax21)
    ax23 = plt.subplot(3, 4, 7, sharey=ax21, sharex=ax21)
    ax24 = plt.subplot(3, 4, 8, sharey=ax21, sharex=ax21)
    legend_ax = plt.subplot(9, 1, 7)
    axs = [ax1, ax2, ax3, ax4]
    axs2 = [ax21, ax22, ax23, ax24]

    for ix, index_type in enumerate(TYPE_ORDER):
        temp_df = df[df["type"] == index_type].copy()
        temp_df = temp_df[temp_df["selectivity"] == DEFAULT_SELECTIVITY]
        temp_df = temp_df.groupby(["block_size", "model"]).mean(numeric_only=True).reset_index()
        temp_df["disk_backed_query_latency"] = (
            temp_df["disk_backed_query_latency"] / 1_000_000
        )

        ax = sb.lineplot(
            ax=axs[ix],
            data=temp_df,
            x="avg_block_size",
            y="disk_backed_query_latency",
            hue="model",
            errorbar=None,
            palette=PALETTE,
            markers=MARKERS,
            style="model",
            markerfacecolor="none",
            markeredgecolor=None,
            linewidth=0.3,
            markeredgewidth=0.5,
            dashes=False,
        )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_ylabel("Query Latency (ms)" if ix == 0 else None, fontsize=7)
        ax.set_xlabel("Avg. Block Size", fontsize=7)
        ax.set_xticks(BLOCK_SIZE_ARRAY[::2], labels=map(str, BLOCK_SIZE_ARRAY[::2]))
        ax.set_yticks([1, 10, 100], labels=map(str, [1, 10, 100]))
        ax.set_title(TYPE_TITLES[ix], fontsize=7)

        handles, labels = ax.get_legend_handles_labels()
        ax.legend().set_visible(False)
        labels_arr += labels
        handles_arr += handles

    for ix, index_type in enumerate(TYPE_ORDER):
        temp_df = optimal_setting_disk[optimal_setting_disk["type"] == index_type].copy()
        temp_df = temp_df.groupby(["selectivity", "model"]).mean(numeric_only=True).reset_index()

        ax = sb.lineplot(
            ax=axs2[ix],
            data=temp_df,
            x="selectivity",
            y="avg_block_size",
            hue="model",
            errorbar=None,
            palette=PALETTE,
            markers=MARKERS,
            style="model",
            markerfacecolor="none",
            markeredgecolor=None,
            linewidth=0.3,
            markeredgewidth=0.5,
            dashes=False,
            legend=False,
        )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)
        ax.set_ylabel("Optimal block sizes" if ix == 0 else None, fontsize=7)
        ax.set_xlabel("Selectivity", fontsize=7)
        ax.set_xticks(SELECTIVITY_ARRAY[::2], labels=map(str, SELECTIVITY_ARRAY[::2]))

    legend_ax.axis("off")
    legend_ax.legend(
        handles_arr,
        [model_labels[label] for label in labels_arr],
        ncols=len(labels_arr),
        loc="center",
        columnspacing=1.2,
        handlelength=0.0,
        handletextpad=0.5,
    )

    plt.subplots_adjust(hspace=0.4, wspace=0.15)
    save_figure(fig, figures_dir / "GRAN_BlockSize_Selectivity_DiskBacked.pdf")


def plot_disk_backed_delay(disk_delay_df: pd.DataFrame, figures_dir: Path) -> float:
    vert_line = disk_delay_df["scan_latency_delay_ratios"].mean()

    fig = plt.figure(figsize=(COLUMN_FIT * 0.75, 0.8))
    ax = fig.add_subplot()
    ax.hist(disk_delay_df["scan_latency_delay_ratios"], bins=np.arange(1, 7, 0.1))
    ax.axvline(x=vert_line, linestyle="dashdot", color="black")
    ax.set_yticklabels([])
    ax.set_xlabel("Delay factor in disk-based scan latency")
    ax.set_ylabel("Frequency")
    plt.subplots_adjust(hspace=0.35, wspace=0.15)

    save_figure(fig, figures_dir / "DiskBacked.pdf")
    return float(vert_line)


def plot_decision_tree(
    x_combined: pd.DataFrame,
    y_combined: pd.Series,
    sample_weights: np.ndarray,
    figures_dir: Path,
    top_k: int = INDEX_USEFULNESS_TOP_K,
) -> tuple[float, dict[str, DecisionTreeSubtree]]:
    tree_specs = [
        (
            0,
            "in-memory",
            "DECISION_TREE_IN_MEMORY.pdf",
            DECISION_TREE_SUBTREE_PARAMS["in-memory"],
        ),
        (
            1,
            "disk-backed",
            "DECISION_TREE_DISK_BACKED.pdf",
            DECISION_TREE_SUBTREE_PARAMS["disk-backed"],
        ),
    ]
    weighted_accuracies = []
    subtrees = {}
    class_order_lines = [
        "Decision tree class order",
        "=========================",
        "",
        f"Training rows include only indexes with latency rank <= {top_k}.",
        "The value array in each subtree follows that subtree's class order.",
        "",
    ]

    for storage_value, storage_label, output_filename, tree_params in tree_specs:
        storage_mask = x_combined["data_storage"] == storage_value
        subtree_x = x_combined.loc[storage_mask].drop(columns=["data_storage"])
        subtree_y = y_combined.loc[storage_mask]
        subtree_weights = sample_weights[storage_mask.to_numpy()]

        subtree_accuracy, classifier = _plot_single_decision_subtree(
            subtree_x,
            subtree_y,
            subtree_weights,
            figures_dir / output_filename,
            tree_params,
            top_k=top_k,
        )
        weighted_accuracies.append(
            (subtree_accuracy, float(np.sum(subtree_weights)), storage_label)
        )
        subtrees[storage_label] = DecisionTreeSubtree(
            storage_value=storage_value,
            storage_label=storage_label,
            classifier=classifier,
            feature_columns=list(subtree_x.columns),
            training_accuracy=subtree_accuracy,
            training_weight_sum=float(np.sum(subtree_weights)),
        )

        class_order_lines.append(f"{storage_label} subtree ({output_filename})")
        class_order_lines.append("-" * (len(class_order_lines[-1])))
        class_order_lines.append(f"parameters: {tree_params}")
        class_order_lines.extend(
            f"{idx}: {model}" for idx, model in enumerate(classifier.classes_)
        )
        class_order_lines.append("")

    (figures_dir / "DECISION_TREE_CLASS_ORDER.txt").write_text(
        "\n".join(class_order_lines),
        encoding="utf-8",
    )

    total_weight = sum(weight for _, weight, _ in weighted_accuracies)
    weighted_accuracy = float(
        sum(accuracy * weight for accuracy, weight, _ in weighted_accuracies)
        / total_weight
    )
    return weighted_accuracy, subtrees


def _plot_single_decision_subtree(
    x: pd.DataFrame,
    y: pd.Series,
    sample_weights: np.ndarray,
    output_path: Path,
    tree_params: dict[str, object],
    top_k: int,
) -> tuple[float, DecisionTreeClassifier]:
    classifier = DecisionTreeClassifier(
        **DECISION_TREE_COMMON_PARAMS,
        **tree_params,
    )
    classifier.fit(x, y, sample_weight=sample_weights)
    print(f"{output_path.stem} classifier classes_: {list(classifier.classes_)}")

    fig = plt.figure(figsize=(30, 30))
    plot_tree(
        classifier,
        feature_names=x.columns,
        class_names=classifier.classes_,
        filled=True,
    )
    save_figure(fig, output_path, dpi=800)

    y_proba = classifier.predict_proba(x)
    accuracy = top_k_accuracy_score(
        y,
        y_proba,
        k=min(top_k, len(classifier.classes_)),
        labels=classifier.classes_,
    )
    return float(accuracy), classifier


def write_index_usefulness_rules(
    optimal_setting_memory: pd.DataFrame,
    optimal_setting_disk: pd.DataFrame,
    figures_dir: Path,
    top_k: int = INDEX_USEFULNESS_TOP_K,
    max_latency_ratio: float = MAX_USEFUL_LATENCY_RATIO,
    max_depth: int = INDEX_USEFULNESS_MAX_DEPTH,
    max_rules_per_index: int = INDEX_USEFULNESS_MAX_RULES_PER_INDEX,
) -> Path:
    usefulness_df = build_index_usefulness_df(
        optimal_setting_memory,
        optimal_setting_disk,
        top_k=top_k,
        max_latency_ratio=max_latency_ratio,
    )
    rows = []
    for model in ORDER_LIST:
        model_df = usefulness_df[usefulness_df["model"] == model].copy()
        rows.extend(
            _index_usefulness_rule_rows(
                model,
                model_df,
                max_depth=max_depth,
                max_rules=max_rules_per_index,
            )
        )

    output_path = figures_dir / "INDEX_USEFULNESS_RULES.txt"
    output_path.write_text(
        _format_index_usefulness_rules_text(
            rows,
            top_k=top_k,
            max_latency_ratio=max_latency_ratio,
            max_depth=max_depth,
            max_rules_per_index=max_rules_per_index,
        ),
        encoding="utf-8",
    )
    return output_path


def build_index_usefulness_df(
    optimal_setting_memory: pd.DataFrame,
    optimal_setting_disk: pd.DataFrame,
    top_k: int,
    max_latency_ratio: float,
) -> pd.DataFrame:
    usefulness_df = build_latency_ranked_df(
        optimal_setting_memory,
        optimal_setting_disk,
        INDEX_USEFULNESS_FEATURES,
    )
    usefulness_df["useful"] = (
        (usefulness_df["latency_rank"] <= top_k)
        | (usefulness_df["latency_ratio"] <= max_latency_ratio)
    )
    return usefulness_df


def _index_usefulness_rule_rows(
    model: str,
    model_df: pd.DataFrame,
    max_depth: int,
    max_rules: int,
) -> list[dict[str, object]]:
    total_scenarios = len(model_df)
    useful_scenarios = int(model_df["useful"].sum())
    if useful_scenarios == 0:
        return [
            {
                "model": model,
                "rule": "No useful region found",
                "precision": 0.0,
                "coverage": 0.0,
                "support": 0,
                "mean_rank": np.nan,
                "mean_ratio": np.nan,
                "useful_scenarios": 0,
                "total_scenarios": total_scenarios,
            }
        ]

    if useful_scenarios == total_scenarios:
        return [
            {
                "model": model,
                "rule": "All measured scenarios",
                "precision": 1.0,
                "coverage": 1.0,
                "support": total_scenarios,
                "mean_rank": model_df["latency_rank"].mean(),
                "mean_ratio": model_df["latency_ratio"].mean(),
                "useful_scenarios": useful_scenarios,
                "total_scenarios": total_scenarios,
            }
        ]

    x = model_df[INDEX_USEFULNESS_FEATURES]
    y = model_df["useful"].astype(int)
    min_samples_leaf = min(
        INDEX_USEFULNESS_MIN_SAMPLES_LEAF,
        max(2, len(model_df) // 10),
    )
    classifier = DecisionTreeClassifier(
        random_state=123,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced",
    )
    classifier.fit(x, y)

    leaf_ids = classifier.apply(x)
    rule_rows = []
    for leaf_id in sorted(np.unique(leaf_ids)):
        mask = leaf_ids == leaf_id
        leaf_df = model_df[mask]
        support = len(leaf_df)
        useful_count = int(leaf_df["useful"].sum())
        if useful_count == 0:
            continue

        useful_leaf_df = leaf_df[leaf_df["useful"]]
        rule_rows.append(
            {
                "model": model,
                "rule": _format_tree_leaf_rule(
                    classifier,
                    leaf_id=int(leaf_id),
                    feature_names=INDEX_USEFULNESS_FEATURES,
                ),
                "precision": useful_count / support,
                "coverage": useful_count / useful_scenarios,
                "support": support,
                "mean_rank": useful_leaf_df["latency_rank"].mean(),
                "mean_ratio": useful_leaf_df["latency_ratio"].mean(),
                "useful_scenarios": useful_scenarios,
                "total_scenarios": total_scenarios,
            }
        )

    if not rule_rows:
        return [
            {
                "model": model,
                "rule": "No compact positive rule found",
                "precision": 0.0,
                "coverage": 0.0,
                "support": 0,
                "mean_rank": np.nan,
                "mean_ratio": np.nan,
                "useful_scenarios": useful_scenarios,
                "total_scenarios": total_scenarios,
            }
        ]

    return sorted(
        rule_rows,
        key=lambda row: (row["precision"], row["coverage"], row["support"]),
        reverse=True,
    )[:max_rules]


def _format_tree_leaf_rule(
    classifier: DecisionTreeClassifier,
    leaf_id: int,
    feature_names: Sequence[str],
) -> str:
    raw_conditions = _tree_leaf_conditions(
        classifier.tree_,
        node_id=0,
        leaf_id=leaf_id,
        feature_names=feature_names,
        path=[],
    )
    if not raw_conditions:
        return "All scenarios"

    allowed_values = {
        feature: set(_usefulness_feature_values(feature))
        for feature in feature_names
    }
    for feature, operator, threshold in raw_conditions:
        allowed_values[feature] &= set(
            _condition_allowed_values(feature, operator, threshold)
        )

    conditions = []
    for feature in feature_names:
        feature_values = set(_usefulness_feature_values(feature))
        if allowed_values[feature] == feature_values:
            continue
        conditions.append(
            _format_allowed_usefulness_values(feature, sorted(allowed_values[feature]))
        )

    return " and ".join(conditions) if conditions else "All scenarios"


def _tree_leaf_conditions(
    tree,
    node_id: int,
    leaf_id: int,
    feature_names: Sequence[str],
    path: list[tuple[str, str, float]],
) -> list[tuple[str, str, float]] | None:
    if node_id == leaf_id:
        return path

    left_id = tree.children_left[node_id]
    right_id = tree.children_right[node_id]
    if left_id == right_id:
        return None

    feature = feature_names[tree.feature[node_id]]
    threshold = tree.threshold[node_id]
    left_path = _tree_leaf_conditions(
        tree,
        node_id=left_id,
        leaf_id=leaf_id,
        feature_names=feature_names,
        path=[*path, (feature, "<=", threshold)],
    )
    if left_path is not None:
        return left_path

    return _tree_leaf_conditions(
        tree,
        node_id=right_id,
        leaf_id=leaf_id,
        feature_names=feature_names,
        path=[*path, (feature, ">", threshold)],
    )


def _usefulness_feature_values(feature: str) -> list[int]:
    return {
        "data_storage": [0, 1],
        "selectivity": [1, 2, 3],
        "dataset_entropy_id": [2, 3, 4],
        "query_entropy_id": [2, 3, 4],
    }[feature]


def _condition_allowed_values(feature: str, operator: str, threshold: float) -> list[int]:
    feature_values = _usefulness_feature_values(feature)
    boundary = int(np.floor(threshold))
    if operator == "<=":
        return [value for value in feature_values if value <= boundary]
    return [value for value in feature_values if value > boundary]


def _format_allowed_usefulness_values(feature: str, values: Sequence[int]) -> str:
    labels = [_usefulness_feature_value_label(feature, value) for value in values]
    return f"{_usefulness_feature_label(feature)} in {{{', '.join(labels)}}}"


def _usefulness_feature_label(feature: str) -> str:
    return {
        "data_storage": "storage",
        "selectivity": "selectivity",
        "dataset_entropy_id": "data skew",
        "query_entropy_id": "query skew",
    }[feature]


def _usefulness_feature_value_label(feature: str, value: int) -> str:
    labels = {
        "data_storage": {
            0: "memory",
            1: "disk",
        },
        "selectivity": {
            1: "low",
            2: "medium",
            3: "high",
        },
        "dataset_entropy_id": {
            2: "low",
            3: "medium",
            4: "high",
        },
        "query_entropy_id": {
            2: "low",
            3: "medium",
            4: "high",
        },
    }
    return labels[feature][value]


def _format_index_usefulness_rules_text(
    rows: list[dict[str, object]],
    top_k: int,
    max_latency_ratio: float,
    max_depth: int,
    max_rules_per_index: int,
) -> str:
    lines = [
        "Index usefulness rules",
        "======================",
        f"Useful means the index is among the top-{top_k} performers "
        f"or has latency <= {max_latency_ratio:.2f}x the best index for a scenario.",
        f"Rules are extracted from one-vs-rest binary trees with max_depth={max_depth}.",
        f"At most {max_rules_per_index} positive rules are shown per index.",
        "",
        "Feature buckets",
        "===============",
        "storage: memory, disk",
        "selectivity: low={0.0064, 0.0256}, medium={0.1024}, high={0.4096, 1.6384}",
        "data skew: low={entropy ids 1,2}, medium={3}, high={4,5}",
        "query skew: low={entropy ids 1,2}, medium={3}, high={4,5}",
        "",
        "| Index | Useful region | Precision | Coverage | Support | Mean rank | Mean latency ratio | Useful/Total |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {model} | {rule} | {precision:.2f} | {coverage:.2f} | {support} | "
            "{mean_rank} | {mean_ratio} | {useful_scenarios}/{total_scenarios} |".format(
                model=row["model"],
                rule=row["rule"],
                precision=row["precision"],
                coverage=row["coverage"],
                support=row["support"],
                mean_rank=_format_optional_float(row["mean_rank"]),
                mean_ratio=_format_optional_float(row["mean_ratio"]),
                useful_scenarios=row["useful_scenarios"],
                total_scenarios=row["total_scenarios"],
            )
        )
    return "\n".join(lines) + "\n"


def _format_optional_float(value: object) -> str:
    if pd.isna(value):
        return "-"
    return f"{float(value):.2f}"


def write_real_world_validation_analysis(
    synthetic_raw_df: pd.DataFrame,
    validation_raw_df: pd.DataFrame,
    optimal_measurable_memory: pd.DataFrame,
    optimal_measurable_disk: pd.DataFrame,
    decision_tree_subtrees: dict[str, DecisionTreeSubtree],
    figures_dir: Path,
    query_gen_policy: str,
    top_k: int = INDEX_USEFULNESS_TOP_K,
) -> Path:
    remapped_validation_raw_df, entropy_summary = remap_dataset_entropy_ids(
        synthetic_raw_df,
        validation_raw_df,
    )
    validation_df = prepare_results(remapped_validation_raw_df, query_gen_policy)

    memory_df, memory_filter_summary = filter_to_synthetic_optimal_block_sizes(
        validation_df,
        optimal_measurable_memory,
    )
    disk_df, disk_filter_summary = filter_to_synthetic_optimal_block_sizes(
        validation_df,
        optimal_measurable_disk,
    )

    memory_evaluation = evaluate_real_world_decision_tree(
        memory_df,
        decision_tree_subtrees["in-memory"],
        latency_column="query_latency",
        top_k=top_k,
    )
    disk_evaluation = evaluate_real_world_decision_tree(
        disk_df,
        decision_tree_subtrees["disk-backed"],
        latency_column="disk_backed_query_latency",
        top_k=top_k,
    )

    output_path = figures_dir / REAL_WORLD_VALIDATION_REPORT
    output_path.write_text(
        format_real_world_validation_report(
            entropy_summary,
            memory_filter_summary,
            disk_filter_summary,
            memory_evaluation,
            disk_evaluation,
            top_k=top_k,
        ),
        encoding="utf-8",
    )
    return output_path


def remap_dataset_entropy_ids(
    synthetic_df: pd.DataFrame,
    validation_df: pd.DataFrame,
) -> tuple[pd.DataFrame, EntropyMappingSummary]:
    training_pairs = (
        synthetic_df[["dataset_entropy", "dataset_entropy_id"]]
        .dropna()
        .drop_duplicates()
        .sort_values(["dataset_entropy_id", "dataset_entropy"])
        .reset_index(drop=True)
    )
    training_pairs["dataset_entropy_id"] = training_pairs["dataset_entropy_id"].astype(int)

    classifier = DecisionTreeClassifier(random_state=123, criterion="entropy")
    classifier.fit(training_pairs[["dataset_entropy"]], training_pairs["dataset_entropy_id"])

    remapped_df = validation_df.copy()
    remapped_df["dataset_entropy_id_original"] = remapped_df["dataset_entropy_id"]
    remapped_df["dataset_entropy_id"] = classifier.predict(
        remapped_df[["dataset_entropy"]]
    ).astype(int)

    validation_pairs = (
        remapped_df[
            [
                "data_sample_num",
                "dataset_entropy",
                "dataset_entropy_id_original",
                "dataset_entropy_id",
            ]
        ]
        .drop_duplicates()
        .sort_values("data_sample_num")
        .reset_index(drop=True)
    )

    return remapped_df, EntropyMappingSummary(
        classifier=classifier,
        training_pairs=training_pairs,
        validation_pairs=validation_pairs,
        validation_row_count=len(remapped_df),
    )


def filter_to_synthetic_optimal_block_sizes(
    validation_df: pd.DataFrame,
    optimal_measurable_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    lookup = (
        optimal_measurable_df[[*VALIDATION_BLOCK_LOOKUP_KEYS, "block_size"]]
        .drop_duplicates()
        .rename(columns={"block_size": "synthetic_optimal_block_size"})
    )
    duplicate_lookup_key_count = int(
        lookup.duplicated(VALIDATION_BLOCK_LOOKUP_KEYS, keep=False).sum()
    )
    if duplicate_lookup_key_count:
        lookup = (
            lookup.sort_values([*VALIDATION_BLOCK_LOOKUP_KEYS, "synthetic_optimal_block_size"])
            .groupby(VALIDATION_BLOCK_LOOKUP_KEYS, as_index=False)
            .first()
        )

    merged_df = validation_df.merge(
        lookup,
        on=VALIDATION_BLOCK_LOOKUP_KEYS,
        how="left",
        indicator=True,
    )
    matched_mask = merged_df["_merge"] == "both"
    retained_mask = (
        matched_mask
        & (merged_df["block_size"] == merged_df["synthetic_optimal_block_size"])
    )
    filtered_df = merged_df.loc[retained_mask].drop(columns=["_merge"]).reset_index(drop=True)

    unmatched_keys = (
        merged_df.loc[~matched_mask, VALIDATION_BLOCK_LOOKUP_KEYS]
        .drop_duplicates()
        .sort_values(VALIDATION_BLOCK_LOOKUP_KEYS)
        .head(10)
        .reset_index(drop=True)
    )
    summary = {
        "input_rows": len(validation_df),
        "input_scenarios": _count_validation_scenarios(validation_df),
        "matched_rows": int(matched_mask.sum()),
        "unmatched_rows": int((~matched_mask).sum()),
        "retained_rows": len(filtered_df),
        "retained_scenarios": _count_validation_scenarios(filtered_df),
        "lookup_rows": len(lookup),
        "duplicate_lookup_key_count": duplicate_lookup_key_count,
        "unmatched_keys": unmatched_keys,
    }
    return filtered_df, summary


def evaluate_real_world_decision_tree(
    block_optimized_df: pd.DataFrame,
    subtree: DecisionTreeSubtree,
    latency_column: str,
    top_k: int,
) -> dict[str, object]:
    ranked_df = rank_validation_indexes(block_optimized_df, latency_column)
    if ranked_df.empty:
        scenario_df = pd.DataFrame()
        return {
            "subtree": subtree,
            "scenario_df": scenario_df,
            "leaf_summary_df": pd.DataFrame(),
            "metrics": summarize_validation_metrics(scenario_df, top_k),
        }

    scenario_features = (
        ranked_df[[*VALIDATION_GROUP_COLUMNS, "dataset_entropy"]]
        .drop_duplicates(subset=VALIDATION_GROUP_COLUMNS)
        .sort_values(VALIDATION_GROUP_COLUMNS)
        .reset_index(drop=True)
    )
    grouped_rankings = {
        key: group.sort_values(["query_latency", "model"]).reset_index(drop=True)
        for key, group in ranked_df.groupby(VALIDATION_GROUP_COLUMNS, sort=False)
    }

    tree_features = scenario_features[VALIDATION_GROUP_COLUMNS].copy()
    tree_features["data_storage"] = subtree.storage_value
    tree_features = _bucket_workload_features(tree_features)
    x = tree_features[subtree.feature_columns]
    probabilities = subtree.classifier.predict_proba(x)
    leaf_ids = subtree.classifier.apply(x)
    classes = np.asarray(subtree.classifier.classes_)
    top_k_limit = min(top_k, len(classes))

    rows = []
    for row_ix, scenario in scenario_features.iterrows():
        scenario_key = tuple(scenario[column] for column in VALIDATION_GROUP_COLUMNS)
        scenario_ranking = grouped_rankings[scenario_key]
        latency_by_model = scenario_ranking.set_index("model")["query_latency"].to_dict()
        rank_by_model = scenario_ranking.set_index("model")["latency_rank"].to_dict()

        actual_best = scenario_ranking.iloc[0]
        actual_best_model = actual_best["model"]
        actual_best_latency = float(actual_best["query_latency"])

        probability_order = np.argsort(probabilities[row_ix])[::-1]
        predicted_models = [str(classes[class_ix]) for class_ix in probability_order]
        predicted_probabilities = [
            float(probabilities[row_ix][class_ix]) for class_ix in probability_order
        ]
        predicted_top_k_models = predicted_models[:top_k_limit]
        predicted_top1_model = predicted_top_k_models[0]
        predicted_top1_latency = latency_by_model.get(predicted_top1_model, np.nan)
        predicted_top_k_latencies = [
            latency_by_model[model]
            for model in predicted_top_k_models
            if model in latency_by_model
        ]
        predicted_top_k_latency = (
            min(predicted_top_k_latencies) if predicted_top_k_latencies else np.nan
        )

        rows.append(
            {
                "data_sample_num": scenario["data_sample_num"],
                "dataset_entropy": scenario["dataset_entropy"],
                "dataset_entropy_id": scenario["dataset_entropy_id"],
                "query_entropy_id": scenario["query_entropy_id"],
                "selectivity": scenario["selectivity"],
                "leaf_id": int(leaf_ids[row_ix]),
                "leaf_prediction_distribution": _format_probability_distribution(
                    predicted_models,
                    predicted_probabilities,
                ),
                "predicted_top_k_models": ", ".join(predicted_top_k_models),
                "actual_best_model": actual_best_model,
                "actual_best_latency": actual_best_latency,
                "predicted_top1_model": predicted_top1_model,
                "predicted_top1_probability": predicted_probabilities[0],
                "predicted_top1_actual_rank": rank_by_model.get(
                    predicted_top1_model,
                    np.nan,
                ),
                "actual_best_predicted_rank": (
                    predicted_models.index(actual_best_model) + 1
                    if actual_best_model in predicted_models
                    else np.nan
                ),
                "top1_hit": predicted_top1_model == actual_best_model,
                "top_k_hit": actual_best_model in predicted_top_k_models,
                "top1_latency_regret": predicted_top1_latency / actual_best_latency,
                "top_k_oracle_latency_regret": (
                    predicted_top_k_latency / actual_best_latency
                ),
            }
        )

    scenario_df = pd.DataFrame(rows)
    leaf_summary_df = summarize_validation_leaves(scenario_df)
    return {
        "subtree": subtree,
        "scenario_df": scenario_df,
        "leaf_summary_df": leaf_summary_df,
        "metrics": summarize_validation_metrics(scenario_df, top_k),
    }


def rank_validation_indexes(df: pd.DataFrame, latency_column: str) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    ranked_df = df.copy()
    ranked_df["query_latency"] = ranked_df[latency_column]
    ranked_df = ranked_df.sort_values(
        [*VALIDATION_GROUP_COLUMNS, "query_latency", "model"]
    ).reset_index(drop=True)
    latency_group = ranked_df.groupby(VALIDATION_GROUP_COLUMNS)["query_latency"]
    ranked_df["best_query_latency"] = latency_group.transform("min")
    ranked_df["latency_ratio"] = ranked_df["query_latency"] / ranked_df["best_query_latency"]
    ranked_df["latency_rank"] = latency_group.rank(method="first", ascending=True)
    return ranked_df


def summarize_validation_metrics(
    scenario_df: pd.DataFrame,
    top_k: int,
) -> dict[str, float]:
    metrics = {
        "scenario_count": float(len(scenario_df)),
        "leaf_count": float(scenario_df["leaf_id"].nunique()) if not scenario_df.empty else 0.0,
        "top1_hit_rate": np.nan,
        "top_k_hit_rate": np.nan,
        "mean_top1_regret": np.nan,
        "median_top1_regret": np.nan,
        "p90_top1_regret": np.nan,
        "max_top1_regret": np.nan,
        "mean_top_k_oracle_regret": np.nan,
        "median_top_k_oracle_regret": np.nan,
        "p90_top_k_oracle_regret": np.nan,
        "max_top_k_oracle_regret": np.nan,
        "mean_predicted_top1_actual_rank": np.nan,
        "median_actual_best_predicted_rank": np.nan,
    }
    for threshold in VALIDATION_REGRET_THRESHOLDS:
        metrics[f"top1_within_{threshold:.2f}x"] = np.nan
        metrics[f"top_k_within_{threshold:.2f}x"] = np.nan

    if scenario_df.empty:
        return metrics

    metrics["top1_hit_rate"] = float(scenario_df["top1_hit"].mean())
    metrics["top_k_hit_rate"] = float(scenario_df["top_k_hit"].mean())
    metrics["mean_top1_regret"] = float(scenario_df["top1_latency_regret"].mean())
    metrics["median_top1_regret"] = float(scenario_df["top1_latency_regret"].median())
    metrics["p90_top1_regret"] = _nan_percentile(
        scenario_df["top1_latency_regret"],
        90,
    )
    metrics["max_top1_regret"] = float(scenario_df["top1_latency_regret"].max())
    metrics["mean_top_k_oracle_regret"] = float(
        scenario_df["top_k_oracle_latency_regret"].mean()
    )
    metrics["median_top_k_oracle_regret"] = float(
        scenario_df["top_k_oracle_latency_regret"].median()
    )
    metrics["p90_top_k_oracle_regret"] = _nan_percentile(
        scenario_df["top_k_oracle_latency_regret"],
        90,
    )
    metrics["max_top_k_oracle_regret"] = float(
        scenario_df["top_k_oracle_latency_regret"].max()
    )
    metrics["mean_predicted_top1_actual_rank"] = float(
        scenario_df["predicted_top1_actual_rank"].mean()
    )
    metrics["median_actual_best_predicted_rank"] = float(
        scenario_df["actual_best_predicted_rank"].median()
    )
    for threshold in VALIDATION_REGRET_THRESHOLDS:
        metrics[f"top1_within_{threshold:.2f}x"] = float(
            (scenario_df["top1_latency_regret"] <= threshold).mean()
        )
        metrics[f"top_k_within_{threshold:.2f}x"] = float(
            (scenario_df["top_k_oracle_latency_regret"] <= threshold).mean()
        )

    return metrics


def summarize_validation_leaves(scenario_df: pd.DataFrame) -> pd.DataFrame:
    if scenario_df.empty:
        return pd.DataFrame()

    rows = []
    for leaf_id, leaf_df in scenario_df.groupby("leaf_id"):
        rows.append(
            {
                "leaf_id": int(leaf_id),
                "scenario_count": len(leaf_df),
                "prediction_distribution": leaf_df[
                    "leaf_prediction_distribution"
                ].iloc[0],
                "actual_winner_counts": _format_value_counts(
                    leaf_df["actual_best_model"]
                ),
                "top1_hit_rate": float(leaf_df["top1_hit"].mean()),
                "top_k_hit_rate": float(leaf_df["top_k_hit"].mean()),
                "median_top1_regret": float(leaf_df["top1_latency_regret"].median()),
                "median_top_k_oracle_regret": float(
                    leaf_df["top_k_oracle_latency_regret"].median()
                ),
                "mean_predicted_top1_actual_rank": float(
                    leaf_df["predicted_top1_actual_rank"].mean()
                ),
            }
        )

    return pd.DataFrame(rows).sort_values("leaf_id").reset_index(drop=True)


def format_real_world_validation_report(
    entropy_summary: EntropyMappingSummary,
    memory_filter_summary: dict[str, object],
    disk_filter_summary: dict[str, object],
    memory_evaluation: dict[str, object],
    disk_evaluation: dict[str, object],
    top_k: int,
) -> str:
    lines = [
        "Real-world validation analysis",
        "==============================",
        "",
        "The decision trees are trained on synthetic block-optimized results. "
        "Real-world rows are first assigned a synthetic dataset entropy id, then "
        "filtered to the synthetic-optimal block size before scoring.",
        "",
    ]
    lines.extend(_format_entropy_mapping_section(entropy_summary))
    lines.extend(
        _format_block_filter_section("In-memory block-size policy", memory_filter_summary)
    )
    lines.extend(
        _format_block_filter_section("Disk-backed block-size policy", disk_filter_summary)
    )
    lines.extend(_format_validation_evaluation_section(memory_evaluation, top_k))
    lines.extend(_format_validation_evaluation_section(disk_evaluation, top_k))
    return "\n".join(lines) + "\n"


def _format_entropy_mapping_section(summary: EntropyMappingSummary) -> list[str]:
    training_ranges = (
        summary.training_pairs.groupby("dataset_entropy_id")["dataset_entropy"]
        .agg(["min", "max", "count"])
        .reset_index()
    )
    validation_pairs = summary.validation_pairs
    train_min = summary.training_pairs["dataset_entropy"].min()
    train_max = summary.training_pairs["dataset_entropy"].max()
    below_training_range = int((validation_pairs["dataset_entropy"] < train_min).sum())
    above_training_range = int((validation_pairs["dataset_entropy"] > train_max).sum())
    mapped_counts = validation_pairs["dataset_entropy_id"].value_counts().sort_index()
    thresholds = sorted(
        threshold
        for threshold in summary.classifier.tree_.threshold
        if threshold != -2
    )

    lines = [
        "Dataset entropy id remapping",
        "----------------------------",
        f"Validation rows remapped: {summary.validation_row_count}",
        f"Unique real-world datasets remapped: {len(validation_pairs)}",
        "Classifier: DecisionTreeClassifier(random_state=123, criterion='entropy')",
        "Synthetic training entropy ranges:",
        "| Synthetic entropy id | Min entropy | Max entropy | Unique points |",
        "|---:|---:|---:|---:|",
    ]
    for _, row in training_ranges.iterrows():
        lines.append(
            "| {dataset_entropy_id} | {min:.9f} | {max:.9f} | {count} |".format(
                dataset_entropy_id=int(row["dataset_entropy_id"]),
                min=row["min"],
                max=row["max"],
                count=int(row["count"]),
            )
        )

    lines.extend(
        [
            "",
            "Classifier thresholds over `dataset_entropy`: "
            + ", ".join(f"{threshold:.9f}" for threshold in thresholds),
            "Mapped real-world dataset counts: "
            + ", ".join(
                f"id {int(entropy_id)}={count}"
                for entropy_id, count in mapped_counts.items()
            ),
            (
                "Real-world entropy range coverage: "
                f"{below_training_range}/{len(validation_pairs)} unique datasets below "
                f"the synthetic minimum ({train_min:.9f}); "
                f"{above_training_range}/{len(validation_pairs)} above the synthetic maximum "
                f"({train_max:.9f})."
            ),
            "",
            "Real-world dataset mapping:",
            "| Dataset | Entropy | Original id | Mapped synthetic id |",
            "|---:|---:|---:|---:|",
        ]
    )
    for _, row in validation_pairs.iterrows():
        lines.append(
            "| {dataset} | {entropy:.9f} | {original_id} | {mapped_id} |".format(
                dataset=int(row["data_sample_num"]),
                entropy=row["dataset_entropy"],
                original_id=int(row["dataset_entropy_id_original"]),
                mapped_id=int(row["dataset_entropy_id"]),
            )
        )
    lines.append("")
    return lines


def _format_block_filter_section(
    title: str,
    summary: dict[str, object],
) -> list[str]:
    lines = [
        title,
        "-" * len(title),
        f"Lookup rows: {summary['lookup_rows']}",
        f"Validation rows before filtering: {summary['input_rows']}",
        f"Validation scenarios before filtering: {summary['input_scenarios']}",
        f"Rows with a synthetic block-size policy: {summary['matched_rows']}",
        f"Rows without a synthetic block-size policy: {summary['unmatched_rows']}",
        f"Rows retained at the synthetic-optimal block size: {summary['retained_rows']}",
        f"Scenarios retained: {summary['retained_scenarios']}",
        f"Duplicate lookup keys resolved: {summary['duplicate_lookup_key_count']}",
    ]
    unmatched_keys = summary["unmatched_keys"]
    if isinstance(unmatched_keys, pd.DataFrame) and not unmatched_keys.empty:
        lines.extend(
            [
                "",
                "First unmatched lookup keys:",
                "| Model | Dataset entropy id | Query entropy id | Selectivity |",
                "|---|---:|---:|---:|",
            ]
        )
        for _, row in unmatched_keys.iterrows():
            lines.append(
                "| {model} | {dataset_entropy_id} | {query_entropy_id} | {selectivity:.4f} |".format(
                    model=row["model"],
                    dataset_entropy_id=int(row["dataset_entropy_id"]),
                    query_entropy_id=int(row["query_entropy_id"]),
                    selectivity=row["selectivity"],
                )
            )
    lines.append("")
    return lines


def _format_validation_evaluation_section(
    evaluation: dict[str, object],
    top_k: int,
) -> list[str]:
    subtree = evaluation["subtree"]
    metrics = evaluation["metrics"]
    scenario_df = evaluation["scenario_df"]
    leaf_summary_df = evaluation["leaf_summary_df"]
    title = f"{subtree.storage_label} decision tree validation"
    lines = [
        title,
        "-" * len(title),
        f"Synthetic training top-{top_k} accuracy: {subtree.training_accuracy:.4f}",
        f"Synthetic training weight sum: {subtree.training_weight_sum:.2f}",
        f"Real-world scenarios: {int(metrics['scenario_count'])}",
        f"Leaves reached: {int(metrics['leaf_count'])}",
        f"Top-1 hit rate: {_format_optional_percent(metrics['top1_hit_rate'])}",
        f"Top-{top_k} hit rate: {_format_optional_percent(metrics['top_k_hit_rate'])}",
        "Top-1 latency regret: "
        f"mean={_format_optional_number(metrics['mean_top1_regret'])}, "
        f"median={_format_optional_number(metrics['median_top1_regret'])}, "
        f"p90={_format_optional_number(metrics['p90_top1_regret'])}, "
        f"max={_format_optional_number(metrics['max_top1_regret'])}",
        f"Top-{top_k} oracle latency regret: "
        f"mean={_format_optional_number(metrics['mean_top_k_oracle_regret'])}, "
        f"median={_format_optional_number(metrics['median_top_k_oracle_regret'])}, "
        f"p90={_format_optional_number(metrics['p90_top_k_oracle_regret'])}, "
        f"max={_format_optional_number(metrics['max_top_k_oracle_regret'])}",
        "Mean actual rank of predicted top-1 index: "
        f"{_format_optional_number(metrics['mean_predicted_top1_actual_rank'])}",
        "Median predicted rank of the actual best index: "
        f"{_format_optional_number(metrics['median_actual_best_predicted_rank'])}",
    ]
    for threshold in VALIDATION_REGRET_THRESHOLDS:
        lines.append(
            f"Within {threshold:.2f}x best latency: "
            f"top-1={_format_optional_percent(metrics[f'top1_within_{threshold:.2f}x'])}, "
            f"top-{top_k}={_format_optional_percent(metrics[f'top_k_within_{threshold:.2f}x'])}"
        )

    if not leaf_summary_df.empty:
        lines.extend(
            [
                "",
                "Leaf diagnostics:",
                (
                    "| Leaf | Scenarios | Tree prediction values | Actual winners | "
                    f"Top-1 hit | Top-{top_k} hit | Median top-1 regret | "
                    f"Median top-{top_k} regret | Mean predicted top-1 actual rank |"
                ),
                "|---:|---:|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for _, row in leaf_summary_df.iterrows():
            lines.append(
                "| {leaf_id} | {scenario_count} | {prediction_distribution} | "
                "{actual_winner_counts} | {top1_hit_rate} | {top_k_hit_rate} | "
                "{median_top1_regret} | {median_top_k_oracle_regret} | "
                "{mean_predicted_top1_actual_rank} |".format(
                    leaf_id=int(row["leaf_id"]),
                    scenario_count=int(row["scenario_count"]),
                    prediction_distribution=row["prediction_distribution"],
                    actual_winner_counts=row["actual_winner_counts"],
                    top1_hit_rate=_format_optional_percent(row["top1_hit_rate"]),
                    top_k_hit_rate=_format_optional_percent(row["top_k_hit_rate"]),
                    median_top1_regret=_format_optional_number(row["median_top1_regret"]),
                    median_top_k_oracle_regret=_format_optional_number(
                        row["median_top_k_oracle_regret"]
                    ),
                    mean_predicted_top1_actual_rank=_format_optional_number(
                        row["mean_predicted_top1_actual_rank"]
                    ),
                )
            )

    if not scenario_df.empty:
        worst_df = scenario_df.sort_values(
            "top1_latency_regret",
            ascending=False,
        ).head(VALIDATION_WORST_CASES)
        lines.extend(
            [
                "",
                f"Worst {len(worst_df)} top-1 regret scenarios:",
                (
                    "| Dataset | Entropy | Query entropy id | Selectivity | Leaf | "
                    "Actual best | Predicted top-1 | Top-1 actual rank | "
                    f"Top-1 regret | Top-{top_k} regret | Predicted top-{top_k} |"
                ),
                "|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---|",
            ]
        )
        for _, row in worst_df.iterrows():
            lines.append(
                "| {data_sample_num} | {dataset_entropy:.9f} | {query_entropy_id} | "
                "{selectivity:.4f} | {leaf_id} | {actual_best_model} | "
                "{predicted_top1_model} | {predicted_top1_actual_rank} | "
                "{top1_latency_regret} | {top_k_oracle_latency_regret} | "
                "{predicted_top_k_models} |".format(
                    data_sample_num=int(row["data_sample_num"]),
                    dataset_entropy=row["dataset_entropy"],
                    query_entropy_id=int(row["query_entropy_id"]),
                    selectivity=row["selectivity"],
                    leaf_id=int(row["leaf_id"]),
                    actual_best_model=row["actual_best_model"],
                    predicted_top1_model=row["predicted_top1_model"],
                    predicted_top1_actual_rank=_format_optional_number(
                        row["predicted_top1_actual_rank"]
                    ),
                    top1_latency_regret=_format_optional_number(
                        row["top1_latency_regret"]
                    ),
                    top_k_oracle_latency_regret=_format_optional_number(
                        row["top_k_oracle_latency_regret"]
                    ),
                    predicted_top_k_models=row["predicted_top_k_models"],
                )
            )
    lines.append("")
    return lines


def _count_validation_scenarios(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    return len(df[VALIDATION_GROUP_COLUMNS].drop_duplicates())


def _format_probability_distribution(
    models: Sequence[str],
    probabilities: Sequence[float],
) -> str:
    return ", ".join(
        f"{model}={probability:.3f}"
        for model, probability in zip(models, probabilities)
        if probability > 0
    )


def _format_value_counts(values: pd.Series) -> str:
    value_counts = values.value_counts().sort_index()
    return ", ".join(f"{value}={count}" for value, count in value_counts.items())


def _nan_percentile(values: pd.Series, percentile: float) -> float:
    clean_values = values.dropna()
    if clean_values.empty:
        return np.nan
    return float(np.percentile(clean_values, percentile))


def _format_optional_number(value: object, digits: int = 3) -> str:
    if pd.isna(value):
        return "-"
    return f"{float(value):.{digits}f}"


def _format_optional_percent(value: object) -> str:
    if pd.isna(value):
        return "-"
    return f"{100.0 * float(value):.1f}%"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paper figures from Results.json.")
    repo_root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=repo_root / "Results" / "20260604-8M",
        help="Directory containing Results.json and Results.pkl.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=None,
        help="Directory where generated PDFs should be written. Defaults to DATA_DIR/figures.",
    )
    parser.add_argument(
        "--query-gen-policy",
        default="count",
        help="Filter for the area_or_count_based column.",
    )
    parser.add_argument(
        "--no-tex",
        action="store_true",
        help="Disable LaTeX rendering for environments without a TeX installation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.figures_dir is None:
        args.figures_dir = args.data_dir / "figures"

    use_tex = not args.no_tex
    model_labels = LATEX_MODEL_LABELS if use_tex else PLAIN_MODEL_LABELS

    configure_matplotlib(use_tex)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    result_path = args.data_dir / "Results.json"
    pickle_path = args.data_dir / "Results.pkl"

    raw_df = load_results(result_path, pickle_path)
    df = prepare_results(raw_df, args.query_gen_policy)

    (
        optimal_setting_memory,
        optimal_setting_disk,
        optimal_measurable_memory,
        optimal_measurable_disk,
    ) = compute_optimal_blocksize_dfs(df)
    latex_table_paths = write_latex_tables(optimal_measurable_memory, args.figures_dir)
    for table_path in latex_table_paths:
        print(f"Wrote LaTeX table: {table_path}")

    best_vs_average_df = build_best_vs_average_df(df, optimal_measurable_memory)
    data_query_entropy_df = build_data_query_entropy_df(optimal_measurable_memory)
    scan_refinement_df = build_scan_refinement_df(optimal_measurable_memory)
    access_stats_df = build_access_stats_df(optimal_measurable_memory)
    disk_delay_df = build_disk_delay_df(df)
    x_combined, y_combined, decision_tree_weights = build_decision_tree_inputs(
        optimal_setting_memory,
        optimal_setting_disk,
    )

    plot_gran_block_size_selectivity(
        df, optimal_setting_memory, args.figures_dir, model_labels
    )
    plot_gran_best_block_vs_average(best_vs_average_df, args.figures_dir, model_labels)
    plot_data_query_entropy_percent_change(
        data_query_entropy_df, args.figures_dir, model_labels
    )
    plot_analysis_refinement_scan(scan_refinement_df, args.figures_dir, model_labels)
    plot_access_patterns(access_stats_df, args.figures_dir, model_labels, use_tex)
    plot_gran_block_size_selectivity_disk_backed(
        df, optimal_setting_disk, args.figures_dir, model_labels
    )
    delay_factor = plot_disk_backed_delay(disk_delay_df, args.figures_dir)
    decision_tree_accuracy, decision_tree_subtrees = plot_decision_tree(
        x_combined, y_combined, decision_tree_weights, args.figures_dir
    )
    usefulness_rules_path = write_index_usefulness_rules(
        optimal_setting_memory,
        optimal_setting_disk,
        args.figures_dir,
    )

    validation_path = args.data_dir / VALIDATION_RESULT_FILENAME
    validation_report_path = None
    if validation_path.is_file():
        validation_pickle_path = validation_path.with_suffix(".pkl")
        validation_raw_df = load_results(validation_path, validation_pickle_path)
        validation_report_path = write_real_world_validation_analysis(
            raw_df,
            validation_raw_df,
            optimal_measurable_memory,
            optimal_measurable_disk,
            decision_tree_subtrees,
            args.figures_dir,
            args.query_gen_policy,
        )

    print(f"Disk-backed scan delay mean: {delay_factor:.4f}")
    print(
        f"Decision tree top-{INDEX_USEFULNESS_TOP_K} accuracy: "
        f"{decision_tree_accuracy:.4f}"
    )
    print(f"Wrote index usefulness rules: {usefulness_rules_path}")
    if validation_report_path is not None:
        print(f"Wrote real-world validation analysis: {validation_report_path}")
    else:
        print(f"Skipped real-world validation analysis: {validation_path} not found")

    # Keep this local to main so future table-writing code has the same cleaned input.
    _ = optimal_measurable_disk


if __name__ == "__main__":
    main()
