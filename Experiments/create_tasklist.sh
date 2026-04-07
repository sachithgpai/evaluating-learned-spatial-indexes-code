#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: bash create_tasklist.sh <dataset_name>" >&2
    exit 1
fi

dataset_name="$1"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

evaluate_bin="${script_dir}/build_evaluate.out"
rsmi_script="${repo_root}/Indexes/RTree/RSMI.py"

evaluate_tasks="${script_dir}/hq_tasks_evaluate"
rsmi_tasks="${script_dir}/hq_tasks_RSMI"

block_sizes=(32 64 128 256 512 1024 2048 4096)
selectivities=(00064 00256 01024 04096 16384)

rm -f "${evaluate_tasks}" "${rsmi_tasks}"

for data_ent_id in {1..5}; do
    for data_sample_num in {1..5}; do
        for block_size in "${block_sizes[@]}"; do
            for selectivity_id in "${!selectivities[@]}"; do
                selectivity="${selectivities[${selectivity_id}]}"
                for query_ent_id in {1..5}; do
                    result_file="P_${block_size}_D_${data_sample_num}_DE_${data_ent_id}_Q_${query_ent_id}_S_${selectivity}.jsonl"
                    {
                        printf '%q ' \
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

            {
                printf '%q ' \
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

dataset_root="${script_dir}/${dataset_name}"
mkdir -p "${dataset_root}/TrainedIndexes/QDTree"
mkdir -p "${dataset_root}/TrainedIndexes/RSMI"
mkdir -p "${dataset_root}/TrainedIndexes/FLOOD"
mkdir -p "${dataset_root}/ResultsFolder"
mkdir -p "${dataset_root}/ResultsFolder_ExtendBlockSize"

# Several index implementations spill temporary block data under the repo root.
mkdir -p "${repo_root}/temp_blockstore"
