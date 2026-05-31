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
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SELECTIVITY_SCALE = 1_000_000


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


if __name__ == "__main__":
    main()
