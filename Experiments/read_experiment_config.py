#!/usr/bin/env python3
"""Read experiment settings and print Bash-friendly task-list values.

Output format, one value per line:
1. resolved experiment name
2. space-separated block sizes
3. space-separated selectivity tags
4. number of dataset samples
5. number of data entropy variants
6. number of query entropy variants
7. single-query-workload-per-sample flag, 1 or 0
8. paged-backend enable flag, 1 or 0
9. comma-separated buffer-pool fractions
10. page size in bytes
11. record size in bytes

Lines 8-11 are appended rather than inserted: create_tasklist.sh reads these by
index, so putting a new value anywhere but the end would silently shift every
line below it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SELECTIVITY_SCALE = 1_000_000

# Defaults match the built-in constants, so a config predating the buffer pool
# still produces a runnable task list.
DEFAULT_BUFFER_POOL_FRACTIONS = [1.0, 0.25, 0.05, 0.01, 0.001]
DEFAULT_PAGE_BYTES = 4096
DEFAULT_RECORD_BYTES = 16


def require_positive_int(config: dict[str, Any], key: str, context: str) -> int:
    if key not in config:
        raise SystemExit(f"Missing '{key}' in {context}.")

    value = int(config[key])
    if value < 1:
        raise SystemExit(f"'{key}' in {context} must be >= 1.")
    return value


def selectivity_tags(target_fractions: list[Any]) -> list[str]:
    return [
        f"{int(round(float(fraction) * SELECTIVITY_SCALE)):05d}"
        for fraction in target_fractions
    ]


def config_bool(config: dict[str, Any], key: str, fallback: bool = False) -> bool:
    value = config.get(key, fallback)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return bool(value)


def require_positive_int_value(value: Any, key: str, context: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise SystemExit(f"'{key}' in {context} must be >= 1.")
    return parsed


def buffer_pool_fractions(evaluation_config: dict[str, Any]) -> list[float]:
    """Validate the buffer-pool budget fractions and return them in sweep order."""
    fractions = evaluation_config.get("buffer_pool_fractions", DEFAULT_BUFFER_POOL_FRACTIONS)
    if not fractions:
        raise SystemExit("'evaluation.buffer_pool_fractions' must not be empty.")

    parsed = [float(fraction) for fraction in fractions]
    for fraction in parsed:
        if not 0.0 < fraction <= 1.0:
            raise SystemExit(
                f"'evaluation.buffer_pool_fractions' entries must be in (0, 1]; got {fraction}."
            )
    return parsed


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract task-list settings from experiment_config.json."
    )
    parser.add_argument("--config", required=True, help="Path to experiment_config.json.")
    parser.add_argument(
        "--experiment",
        default="",
        help="Experiment profile to read. Defaults to default_experiment.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config_path = Path(args.config)

    with open(config_path, "r", encoding="utf-8") as handle:
        project_config = json.load(handle)

    # Select the experiment profile first; all loop bounds come from that scope.
    experiment_name = args.experiment or project_config.get(
        "default_experiment",
        "synthetic",
    )
    experiments = project_config.get("experiments", {})
    if experiment_name not in experiments:
        known = ", ".join(sorted(experiments))
        raise SystemExit(
            f"Unknown experiment '{experiment_name}' in {config_path}. "
            f"Known experiments: {known}"
        )
    experiment_config = experiments[experiment_name]

    # Evaluation settings are shared by all experiment profiles.
    evaluation_config = project_config.get("evaluation", {})
    block_sizes = evaluation_config.get("block_sizes")
    if not block_sizes:
        raise SystemExit(f"Missing 'evaluation.block_sizes' in {config_path}.")

    target_fractions = experiment_config.get("target_fractions")
    if not target_fractions:
        raise SystemExit(
            f"Missing 'target_fractions' for experiment "
            f"'{experiment_name}' in {config_path}."
        )

    # Dataset and entropy loop bounds differ slightly between synthetic and real profiles.
    if "num_datasets" in experiment_config:
        num_dataset_samples = require_positive_int(
            experiment_config,
            "num_datasets",
            f"experiment '{experiment_name}'",
        )
    elif "num_samples" in experiment_config:
        num_dataset_samples = require_positive_int(
            experiment_config,
            "num_samples",
            f"experiment '{experiment_name}'",
        )
    else:
        raise SystemExit(
            f"Experiment '{experiment_name}' needs either "
            "'num_datasets' or 'num_samples'."
        )

    if "data_entropy_variants" in experiment_config:
        num_data_entropy_variants = require_positive_int(
            experiment_config,
            "data_entropy_variants",
            f"experiment '{experiment_name}'",
        )
    elif "num_samples" in experiment_config and "num_datasets" not in experiment_config:
        num_data_entropy_variants = 1
    else:
        num_data_entropy_variants = require_positive_int(
            experiment_config,
            "num_query_scales",
            f"experiment '{experiment_name}'",
        )

    if "query_entropy_variants" in experiment_config:
        num_query_entropy_variants = require_positive_int(
            experiment_config,
            "query_entropy_variants",
            f"experiment '{experiment_name}'",
        )
    else:
        num_query_entropy_variants = require_positive_int(
            experiment_config,
            "num_query_scales",
            f"experiment '{experiment_name}'",
        )

    # Keep this print order in sync with create_tasklist.sh.
    print(experiment_name)
    print(" ".join(str(int(block_size)) for block_size in block_sizes))
    print(" ".join(selectivity_tags(target_fractions)))
    print(num_dataset_samples)
    print(num_data_entropy_variants)
    print(num_query_entropy_variants)
    print(1 if config_bool(experiment_config, "single_query_workload_per_sample") else 0)

    # Lines 8-12: storage-backend settings. The fraction sweep runs inside the
    # evaluator binary, so this list is one env var rather than a task dimension --
    # keeping the task count unchanged.
    print(1 if config_bool(evaluation_config, "enable_paged_backend") else 0)
    print(",".join(repr(fraction) for fraction in buffer_pool_fractions(evaluation_config)))
    print(
        require_positive_int_value(
            evaluation_config.get("page_bytes", DEFAULT_PAGE_BYTES),
            "page_bytes",
            "evaluation",
        )
    )
    print(
        require_positive_int_value(
            evaluation_config.get("record_bytes", DEFAULT_RECORD_BYTES),
            "record_bytes",
            "evaluation",
        )
    )
    # Line 12: bypass the OS page cache on the buffer pool's read path. Requires
    # storage that accepts O_DIRECT -- point TEMP_BLOCKSTORE_DIR at node-local
    # disk, since Lustre refuses direct reads below 4096 bytes.
    print(1 if config_bool(evaluation_config, "direct_io") else 0)
    # Line 13: block sizes below this skip the buffer-pool passes entirely. A block
    # never shares a page, so with the defaults above a block of 256 records fills
    # a page exactly and anything smaller measures padding -- 8x write
    # amplification at 32 records, where the miss counts describe the page layout
    # rather than the index. Those tasks still produce their in-memory rows.
    print(int(evaluation_config.get("buffer_pool_min_block_size",
                                    DEFAULT_PAGE_BYTES // DEFAULT_RECORD_BYTES)))


if __name__ == "__main__":
    main()
