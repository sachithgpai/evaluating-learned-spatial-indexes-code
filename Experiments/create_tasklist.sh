#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: bash create_tasklist.sh <dataset_name> [experiment_name]" >&2
    exit 1
fi

dataset_name="$1"

# Resolve project paths and the experiment profile to use.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
config_path="${EXPERIMENT_CONFIG:-${repo_root}/experiment_config.json}"
experiment_name="${2:-${EXPERIMENT_NAME:-}}"

evaluate_bin="${script_dir}/build_evaluate.out"
rsmi_script="${repo_root}/Indexes/RTree/RSMI.py"
config_reader="${script_dir}/read_experiment_config.py"

evaluate_tasks="${script_dir}/hq_tasks_evaluate"
rsmi_tasks="${script_dir}/hq_tasks_RSMI"

# Translate the JSON config into simple Bash values used by the loops below.
config_output="$(
    python3 "${config_reader}" \
        --config "${config_path}" \
        --experiment "${experiment_name}"
)"
mapfile -t config_lines <<< "${config_output}"
experiment_name="${config_lines[0]}"
IFS=' ' read -r -a block_sizes <<< "${config_lines[1]}"
IFS=' ' read -r -a selectivities <<< "${config_lines[2]}"
num_dataset_samples="${config_lines[3]}"
num_data_entropy_variants="${config_lines[4]}"
num_query_entropy_variants="${config_lines[5]}"

rm -f "${evaluate_tasks}" "${rsmi_tasks}"

# Write one evaluation task for every configured data/query/selectivity/block-size combination.
for ((data_ent_id = 1; data_ent_id <= num_data_entropy_variants; data_ent_id++)); do
    for ((data_sample_num = 1; data_sample_num <= num_dataset_samples; data_sample_num++)); do
        for block_size in "${block_sizes[@]}"; do
            for selectivity_id in "${!selectivities[@]}"; do
                selectivity="${selectivities[${selectivity_id}]}"
                for ((query_ent_id = 1; query_ent_id <= num_query_entropy_variants; query_ent_id++)); do
                    result_file="P_${block_size}_D_${data_sample_num}_DE_${data_ent_id}_Q_${query_ent_id}_S_${selectivity}.jsonl"
                    {
                        printf '%q ' \
                            env \
                            "PROJECT_ROOT=${repo_root}" \
                            "EXPERIMENT_CONFIG=${config_path}" \
                            "EXPERIMENT_NAME=${experiment_name}" \
                            "${evaluate_bin}" \
                            "${dataset_name}" \
                            "${data_sample_num}" \
                            "${data_ent_id}" \
                            "${block_size}" \
                            "${query_ent_id}" \
                            "${selectivity_id}" \
                            "${result_file}"
                        printf '\n'
                    } >> "${evaluate_tasks}"
                done
            done

            # RSMI is trained once per dataset entropy variant and block size.
            {
                printf '%q ' \
                    env \
                    "PROJECT_ROOT=${repo_root}" \
                    "EXPERIMENT_CONFIG=${config_path}" \
                    "EXPERIMENT_NAME=${experiment_name}" \
                    python3 \
                    "${rsmi_script}" \
                    "${dataset_name}" \
                    "${data_sample_num}" \
                    "${data_ent_id}" \
                    "${block_size}"
                printf '\n'
            } >> "${rsmi_tasks}"
        done
    done
done

# Ensure the evaluator and trainers have all expected output directories.
dataset_root="${script_dir}/${dataset_name}"
mkdir -p "${dataset_root}/TrainedIndexes/QDTree"
mkdir -p "${dataset_root}/TrainedIndexes/RSMI"
mkdir -p "${dataset_root}/TrainedIndexes/FLOOD"
mkdir -p "${dataset_root}/ResultsFolder"
mkdir -p "${dataset_root}/ResultsFolder_ExtendBlockSize"

# Several index implementations spill temporary block data under the repo root.
mkdir -p "${repo_root}/temp_blockstore"
