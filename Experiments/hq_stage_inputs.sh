#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: bash hq_stage_inputs.sh <dataset_name>" >&2
    exit 1
fi

: "${LOCAL_SCRATCH:?LOCAL_SCRATCH must be set by the Slurm node-local storage environment.}"

dataset_name="$1"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
config_path="${EXPERIMENT_CONFIG:-${repo_root}/experiment_config.json}"

source_dataset="${repo_root}/Datasets/${dataset_name}"
source_rsmi="${script_dir}/${dataset_name}/TrainedIndexes/RSMI"

if [[ -z "${dataset_name}" ]]; then
    echo "Dataset name must not be empty." >&2
    exit 1
fi

if [[ ! -d "${source_dataset}" ]]; then
    echo "Dataset folder not found: ${source_dataset}" >&2
    exit 1
fi

if [[ ! -d "${source_rsmi}" ]]; then
    echo "RSMI model folder not found: ${source_rsmi}" >&2
    echo "Train RSMI first with hq_tasks_RSMI, then run this evaluation workflow." >&2
    exit 1
fi

if [[ ! -f "${config_path}" ]]; then
    echo "Experiment config not found: ${config_path}" >&2
    exit 1
fi

copy_dir() {
    local source_dir="$1"
    local target_dir="$2"

    mkdir -p "$(dirname "${target_dir}")"
    if command -v rsync >/dev/null 2>&1; then
        mkdir -p "${target_dir}"
        rsync -a --delete "${source_dir}/" "${target_dir}/"
    else
        rm -rf "${target_dir}"
        cp -a "${source_dir}" "${target_dir}"
    fi
}

mkdir -p "${LOCAL_SCRATCH}/Datasets"
mkdir -p "${LOCAL_SCRATCH}/Experiments/${dataset_name}/TrainedIndexes"
mkdir -p "${LOCAL_SCRATCH}/Experiments/${dataset_name}/TrainedIndexes/QDTree"
mkdir -p "${LOCAL_SCRATCH}/Experiments/${dataset_name}/TrainedIndexes/FLOOD"
mkdir -p "${LOCAL_SCRATCH}/Experiments/${dataset_name}/ResultsFolder"
mkdir -p "${LOCAL_SCRATCH}/output"
mkdir -p "${LOCAL_SCRATCH}/temp_blockstore"

find "${LOCAL_SCRATCH}/output" -maxdepth 1 -type f -name '*.json' -delete
find "${LOCAL_SCRATCH}/temp_blockstore" -maxdepth 1 -type f -delete

copy_dir "${source_dataset}" "${LOCAL_SCRATCH}/Datasets/${dataset_name}"
copy_dir "${source_rsmi}" "${LOCAL_SCRATCH}/Experiments/${dataset_name}/TrainedIndexes/RSMI"
cp "${config_path}" "${LOCAL_SCRATCH}/experiment_config.json"

echo "Staged ${dataset_name} on ${SLURMD_NODENAME:-$(hostname)} at ${LOCAL_SCRATCH}"
