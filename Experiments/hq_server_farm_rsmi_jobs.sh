#!/usr/bin/env bash
#SBATCH --account=project_2005865
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --cpus-per-task=1
#SBATCH --mem=4000
#SBATCH --time=00:15:00
#SBATCH --job-name=hq-rsmi
#SBATCH --output=slurm-rsmi-%j.out
#SBATCH --error=slurm-rsmi-%j.err

set -euo pipefail

run_rsmi_task() {
    local task_list="$1"

    : "${HQ_TASK_ID:?HQ_TASK_ID must be set by HyperQueue array submission.}"

    if [[ ! "${HQ_TASK_ID}" =~ ^[0-9]+$ || "${HQ_TASK_ID}" -lt 1 ]]; then
        echo "HQ_TASK_ID must be a positive line number, got: ${HQ_TASK_ID}" >&2
        exit 1
    fi

    if [[ ! -f "${task_list}" ]]; then
        echo "RSMI task list not found: ${task_list}" >&2
        exit 1
    fi

    task_command="$(sed -n "${HQ_TASK_ID}p" "${task_list}")"
    if [[ -z "${task_command}" ]]; then
        echo "No command found on line ${HQ_TASK_ID} of ${task_list}" >&2
        exit 1
    fi

    eval "task_argv=(${task_command})"
    "${task_argv[@]}"
}

if [[ "${1:-}" == "--run-task" ]]; then
    if [[ $# -ne 2 ]]; then
        echo "Usage: bash hq_server_farm_rsmi_jobs.sh --run-task <hq_tasks_RSMI>" >&2
        exit 1
    fi
    run_rsmi_task "$2"
    exit 0
fi

if [[ $# -gt 1 ]]; then
    echo "Usage: sbatch hq_server_farm_rsmi_jobs.sh [hq_tasks_RSMI]" >&2
    exit 1
fi

task_list="${1:-${RSMI_TASK_LIST:-${SLURM_SUBMIT_DIR}/hq_tasks_RSMI}}"
echo $SLURM_SUBMIT_DIR $task_list 

if [[ "${task_list}" != /* ]]; then
    task_list="$(cd "$(dirname "${task_list}")" && pwd)/$(basename "${task_list}")"
fi

if [[ ! -f "${task_list}" ]]; then
    echo "RSMI task list not found: ${task_list}" >&2
    echo "Run create_tasklist.sh first to generate hq_tasks_RSMI." >&2
    exit 1
fi

task_count="$(wc -l < "${task_list}" | tr -d '[:space:]')"
if [[ -z "${task_count}" || "${task_count}" -lt 1 ]]; then
    echo "RSMI task list is empty: ${task_list}" >&2
    exit 1
fi

if command -v module >/dev/null 2>&1; then
    module load hyperqueue
fi

if ! HQ_BIN="$(command -v hq)"; then
    echo "HyperQueue binary not found in PATH." >&2
    exit 1
fi

worker_count="${SLURM_NTASKS:-${SLURM_JOB_NUM_NODES:-1}}"
export HQ_SERVER_DIR="${SLURM_SUBMIT_DIR}/hq-server-rsmi/${SLURM_JOB_ID:-manual}"
mkdir -p "${HQ_SERVER_DIR}"

cleanup() {
    "${HQ_BIN}" worker stop all >/dev/null 2>&1 || true
    "${HQ_BIN}" server stop >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "Using HyperQueue binary: ${HQ_BIN}"
echo "Running on nodes: ${SLURM_JOB_NODELIST:-unknown}"
echo "SLURM_NTASKS: ${SLURM_NTASKS:-unknown}"
echo "SLURM_CPUS_PER_TASK: ${SLURM_CPUS_PER_TASK:-1}"
echo "RSMI tasks: ${task_count}"

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

echo "Submitting HyperQueue RSMI training array..."
"${HQ_BIN}" submit \
    --stdout=none \
    --stderr=none \
    --cpus=1 \
    --array=1-"${task_count}" \
    bash "${SLURM_SUBMIT_DIR}/hq_server_farm_rsmi_jobs.sh" --run-task "${task_list}"

echo "Waiting for RSMI training jobs to finish..."
job_wait_status=0
"${HQ_BIN}" job wait all || job_wait_status=$?

if (( job_wait_status != 0 )); then
    echo "One or more RSMI HyperQueue jobs failed." >&2
    exit "${job_wait_status}"
fi

echo "Done."
