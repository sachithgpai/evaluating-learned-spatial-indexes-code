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

evaluate_tasks="${script_dir}/hq_eval_tasks"
rsmi_tasks="${script_dir}/hq_tasks_RSMI"
evaluate_line_runner="${script_dir}/evaluate_line_n.sh"
rsmi_line_runner="${script_dir}/rsmi_line_n.sh"
slurm_evaluate_script="${script_dir}/slurm_evaluate_array.sh"
slurm_rsmi_script="${script_dir}/slurm_rsmi_array.sh"

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
single_query_workload_per_sample="${config_lines[6]:-0}"

# Storage-backend settings (lines 8-11). Defaults keep this script working against
# a config file that predates the buffer pool.
enable_paged_backend="${config_lines[7]:-0}"
buffer_pool_fractions="${config_lines[8]:-1.0}"
page_bytes="${config_lines[9]:-4096}"
record_bytes="${config_lines[10]:-16}"
direct_io="${config_lines[11]:-0}"

rm -f "${evaluate_tasks}" "${rsmi_tasks}"

evaluation_task_id=0

write_evaluation_task() {
    local data_sample_num="$1"
    local data_ent_id="$2"
    local block_size="$3"
    local query_ent_id="$4"
    local selectivity_id="$5"
    local result_file

    ((++evaluation_task_id))
    result_file="${evaluation_task_id}.jsonl"
    {
        printf '%q ' \
            env \
            "PROJECT_ROOT=${repo_root}" \
            "EXPERIMENT_CONFIG=${config_path}" \
            "EXPERIMENT_NAME=${experiment_name}" \
            "ENABLE_PAGED_BACKEND=${enable_paged_backend}" \
            "BUFFER_POOL_FRACTIONS=${buffer_pool_fractions}" \
            "PAGE_BYTES=${page_bytes}" \
            "RECORD_BYTES=${record_bytes}" \
            "BUFFER_POOL_DIRECT_IO=${direct_io}" \
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
}

write_rsmi_task() {
    local data_sample_num="$1"
    local data_ent_id="$2"
    local block_size="$3"

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
}

if [[ "${single_query_workload_per_sample}" == "1" ]]; then
    # Validation real workloads generate one selected data/query pair per sample.
    for ((data_sample_num = 1; data_sample_num <= num_dataset_samples; data_sample_num++)); do
        selected_workload_path="${repo_root}/Datasets/${dataset_name}/${data_sample_num}/queries/selected_workload.json"
        if [[ ! -f "${selected_workload_path}" ]]; then
            echo "Missing selected workload metadata: ${selected_workload_path}" >&2
            echo "Generate the real dataset first with single_query_workload_per_sample enabled." >&2
            exit 1
        fi

        selected_output="$(
            python3 -c 'import json, sys
with open(sys.argv[1], "r", encoding="utf-8") as handle:
    workload = json.load(handle)
for key in ("data_entropy_id", "query_entropy_id", "selectivity_id", "selectivity_tag"):
    if key not in workload:
        raise SystemExit(f"Missing {key} in {sys.argv[1]}")
print(int(workload["data_entropy_id"]))
print(int(workload["query_entropy_id"]))
print(int(workload["selectivity_id"]))
print(str(workload["selectivity_tag"]))' "${selected_workload_path}"
        )"
        mapfile -t selected_lines <<< "${selected_output}"
        data_ent_id="${selected_lines[0]}"
        query_ent_id="${selected_lines[1]}"
        selectivity_id="${selected_lines[2]}"
        selectivity_tag="${selected_lines[3]}"

        if (( selectivity_id < 0 || selectivity_id >= ${#selectivities[@]} )); then
            echo "Invalid selectivity_id ${selectivity_id} in ${selected_workload_path}" >&2
            exit 1
        fi
        if (( query_ent_id < 1 || query_ent_id > num_query_entropy_variants )); then
            echo "Invalid query_entropy_id ${query_ent_id} in ${selected_workload_path}" >&2
            exit 1
        fi
        if [[ "${selectivities[${selectivity_id}]}" != "${selectivity_tag}" ]]; then
            echo "Selected workload ${selected_workload_path} was generated with selectivity ${selectivity_tag}, but the current config has ${selectivities[${selectivity_id}]} at selectivity_id ${selectivity_id}." >&2
            exit 1
        fi

        for block_size in "${block_sizes[@]}"; do
            write_evaluation_task \
                "${data_sample_num}" \
                "${data_ent_id}" \
                "${block_size}" \
                "${query_ent_id}" \
                "${selectivity_id}"
            write_rsmi_task "${data_sample_num}" "${data_ent_id}" "${block_size}"
        done
    done
else
    # Write one evaluation task for every configured data/query/selectivity/block-size combination.
    for ((data_ent_id = 1; data_ent_id <= num_data_entropy_variants; data_ent_id++)); do
        for ((data_sample_num = 1; data_sample_num <= num_dataset_samples; data_sample_num++)); do
            for block_size in "${block_sizes[@]}"; do
                for selectivity_id in "${!selectivities[@]}"; do
                    for ((query_ent_id = 1; query_ent_id <= num_query_entropy_variants; query_ent_id++)); do
                        write_evaluation_task \
                            "${data_sample_num}" \
                            "${data_ent_id}" \
                            "${block_size}" \
                            "${query_ent_id}" \
                            "${selectivity_id}"
                    done
                done

                # RSMI is trained once per dataset entropy variant and block size.
                write_rsmi_task "${data_sample_num}" "${data_ent_id}" "${block_size}"
            done
        done
    done
fi


evaluation_task_count="$(wc -l < "${evaluate_tasks}" | tr -d '[:space:]')"
rsmi_task_count="$(wc -l < "${rsmi_tasks}" | tr -d '[:space:]')"

# Ensure the evaluator and trainers have all expected output directories.
dataset_root="${script_dir}/${dataset_name}"
mkdir -p "${dataset_root}/TrainedIndexes/QDTree"
mkdir -p "${dataset_root}/TrainedIndexes/RSMI"
mkdir -p "${dataset_root}/TrainedIndexes/FLOOD"
mkdir -p "${dataset_root}/ResultsFolder"

# Several index implementations spill temporary block data under the repo root.
mkdir -p "${repo_root}/temp_blockstore"

cat > "${slurm_evaluate_script}" <<EOF
#!/usr/bin/env bash
#SBATCH --account=project_2005865
#SBATCH --partition=large
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4000
#SBATCH --time=03:00:00
#SBATCH --job-name=evaluate-indexes
#SBATCH --array=1-${evaluation_task_count}
#SBATCH --output=slurm-evaluate-%A_%a.out
#SBATCH --error=slurm-evaluate-%A_%a.err

set -euo pipefail

script_dir="\${SLURM_SUBMIT_DIR:?Submit this script from the Experiments directory.}"
if [[ ! -f "\${script_dir}/evaluate_line_n.sh" ]]; then
    echo "evaluate_line_n.sh not found in SLURM_SUBMIT_DIR: \${script_dir}" >&2
    echo "Run: cd <repo>/Experiments && sbatch slurm_evaluate_array.sh" >&2
    exit 1
fi
task_list="\${EVALUATION_TASK_LIST:-\${script_dir}/hq_eval_tasks}"

# ---------------------------------------------------------------------------
# Put the paged block store on node-local NVMe.
#
# \${TMPDIR} is Roihu's per-job local NVMe area: created automatically, private
# to the job, and removed when it ends. It needs no --gres reservation here,
# unlike Puhti and Mahti. Use it rather than a hand-built path under /tmp, which
# duplicates what the site already provides and relies on the epilog's
# catch-all sweep for cleanup.
#
# Why not the default location: that is under PROJECT_ROOT on Lustre, a shared
# network filesystem that refuses O_DIRECT reads below 4096 bytes outright and
# serves the ones it accepts in ~310us with variance set by the rest of the
# cluster. Local NVMe answers in ~74us and is what a single-node DBMS actually
# runs on -- the setting the mmap critique this work responds to was made in.
#
# Resolved on the compute node rather than baked into the task list, since the
# task list is generated where \${TMPDIR} means something else entirely.
#
# One caveat worth carrying: \${TMPDIR} is a directory on a shared filesystem,
# not a private device. CSC documents the capacity as possibly "shared with
# other jobs or users on the same node", and it has no per-job bandwidth quota.
# Concurrent tasks on one node therefore contend for one NVMe queue. Page-miss
# counts are unaffected -- they are deterministic -- but per-miss latency is
# not. Run with --exclusive, or throttle the array with --array=1-N%k, when the
# latency numbers themselves matter; the device probe logged on every result
# row is what makes a contended task identifiable afterwards.
# ---------------------------------------------------------------------------
if [[ -z "\${TEMP_BLOCKSTORE_DIR:-}" ]]; then
    export TEMP_BLOCKSTORE_DIR="\${TMPDIR:-/tmp/\${USER}}/blockstore/"
fi
mkdir -p "\${TEMP_BLOCKSTORE_DIR}"

echo "TEMP_BLOCKSTORE_DIR=\${TEMP_BLOCKSTORE_DIR}"
df -hP "\${TEMP_BLOCKSTORE_DIR}" | tail -1

bash "\${script_dir}/evaluate_line_n.sh" "\${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}" "\${task_list}"
EOF

cat > "${slurm_rsmi_script}" <<EOF
#!/usr/bin/env bash
#SBATCH --account=project_2005865
#SBATCH --partition=test
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8000
#SBATCH --time=04:00:00
#SBATCH --job-name=train-rsmi
#SBATCH --array=1-${rsmi_task_count}
#SBATCH --output=slurm-rsmi-%A_%a.out
#SBATCH --error=slurm-rsmi-%A_%a.err

set -euo pipefail

script_dir="\${SLURM_SUBMIT_DIR:?Submit this script from the Experiments directory.}"
if [[ ! -f "\${script_dir}/rsmi_line_n.sh" ]]; then
    echo "rsmi_line_n.sh not found in SLURM_SUBMIT_DIR: \${script_dir}" >&2
    echo "Run: cd <repo>/Experiments && sbatch slurm_rsmi_array.sh" >&2
    exit 1
fi
task_list="\${RSMI_TASK_LIST:-\${script_dir}/hq_tasks_RSMI}"

bash "\${script_dir}/rsmi_line_n.sh" "\${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}" "\${task_list}"
EOF

chmod +x "${slurm_evaluate_script}"
chmod +x "${slurm_rsmi_script}"
chmod +x "${evaluate_line_runner}"
chmod +x "${rsmi_line_runner}"

echo "Wrote ${evaluation_task_count} evaluation tasks to ${evaluate_tasks}"
echo "Wrote ${rsmi_task_count} RSMI tasks to ${rsmi_tasks}"
echo "Wrote Slurm evaluation array script to ${slurm_evaluate_script}"
echo "Wrote Slurm RSMI array script to ${slurm_rsmi_script}"
