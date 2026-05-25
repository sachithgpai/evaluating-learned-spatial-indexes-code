#!/usr/bin/env bash
#SBATCH --account=project_2005865
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --mem=16000
#SBATCH --time=00:25:00
#SBATCH --gres=nvme:16
#SBATCH --job-name=hq-evaluate
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: sbatch hq_server_farm_jobs.sh <dataset_name> [experiment_name]" >&2
    exit 1
fi

dataset_name="$1"
experiment_name="${2:-${EXPERIMENT_NAME:-}}"

repo_root="$(cd "${SLURM_SUBMIT_DIR}/.." && pwd)"
task_list="${EVALUATION_TASK_LIST:-${SLURM_SUBMIT_DIR}/hq_tasks_evaluate}"
config_path="${EXPERIMENT_CONFIG:-${repo_root}/experiment_config.json}"
output_dir="${SLURM_SUBMIT_DIR}/output"

if [[ ! -f "${task_list}" ]]; then
    echo "Evaluation task list not found: ${task_list}" >&2
    echo "Run: bash create_tasklist.sh ${dataset_name}${experiment_name:+ ${experiment_name}}" >&2
    exit 1
fi

if [[ ! -f "${config_path}" ]]; then
    echo "Experiment config not found: ${config_path}" >&2
    exit 1
fi

task_count="$(wc -l < "${task_list}" | tr -d '[:space:]')"
if [[ -z "${task_count}" || "${task_count}" -lt 1 ]]; then
    echo "Evaluation task list is empty: ${task_list}" >&2
    exit 1
fi

if command -v module >/dev/null 2>&1; then
    module load hyperqueue
fi

HQ_BIN="$(command -v hq)"
node_count="${SLURM_JOB_NUM_NODES:-1}"
worker_count="${SLURM_NTASKS:-${node_count}}"

export EXPERIMENT_CONFIG="${config_path}"
if [[ -n "${experiment_name}" ]]; then
    export EXPERIMENT_NAME="${experiment_name}"
fi
export EVALUATION_TASK_LIST="${task_list}"
export HQ_SERVER_DIR="${SLURM_SUBMIT_DIR}/hq-server/${SLURM_JOB_ID:-manual}"

mkdir -p "${HQ_SERVER_DIR}"
mkdir -p "${output_dir}"

cleanup() {
    "$HQ_BIN" worker stop all >/dev/null 2>&1 || true
    "$HQ_BIN" server stop >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "Using HyperQueue binary: ${HQ_BIN}"
echo "Running on nodes: ${SLURM_JOB_NODELIST:-unknown}"
echo "SLURM_NTASKS: ${SLURM_NTASKS:-unknown}"
echo "SLURM_CPUS_PER_TASK: ${SLURM_CPUS_PER_TASK:-1}"
echo "Evaluation tasks: ${task_count}"

echo "Starting HyperQueue server..."
"${HQ_BIN}" server start &
sleep 10

echo "Starting HyperQueue workers..."
srun --export=ALL --overlap --cpu-bind=none --mpi=none "${HQ_BIN}" worker start \
    --manager slurm \
    --on-server-lost finish-running \
    --cpus="${SLURM_CPUS_PER_TASK:-1}" &

echo "Waiting for workers..."
"${HQ_BIN}" worker wait "${worker_count}"

echo "Staging dataset and RSMI models to LOCAL_SCRATCH on each node..."
srun --export=ALL --overlap --nodes="${node_count}" --ntasks="${node_count}" --ntasks-per-node=1 \
     -w "${SLURM_JOB_NODELIST}" \
    bash "${SLURM_SUBMIT_DIR}/hq_stage_inputs.sh" "${dataset_name}"

echo "Submitting HyperQueue evaluation array..."
"${HQ_BIN}" submit \
    --stdout=none \
    --stderr=none \
    --cpus=1 \
    --array=1-"${task_count}" \
    bash "${SLURM_SUBMIT_DIR}/hq_run_evaluate_task.sh" "${task_list}"

echo "Waiting for evaluation jobs to finish..."
job_wait_status=0
"${HQ_BIN}" job wait all || job_wait_status=$?

echo "Archiving LOCAL_SCRATCH outputs into ${output_dir}..."
srun --export=ALL --overlap --nodes="${node_count}" --ntasks="${node_count}" --ntasks-per-node=1 \
     -w "${SLURM_JOB_NODELIST}" \
    bash "${SLURM_SUBMIT_DIR}/hq_archive_outputs.sh" "${output_dir}"

if (( job_wait_status != 0 )); then
    echo "One or more HyperQueue jobs failed." >&2
    exit "${job_wait_status}"
fi

echo "Done."
