#!/usr/bin/env bash

set -euo pipefail

: "${LOCAL_SCRATCH:?LOCAL_SCRATCH must be set by the Slurm node-local storage environment.}"
: "${HQ_TASK_ID:?HQ_TASK_ID must be set by HyperQueue array submission.}"

if [[ ! "${HQ_TASK_ID}" =~ ^[0-9]+$ || "${HQ_TASK_ID}" -lt 1 ]]; then
    echo "HQ_TASK_ID must be a positive line number, got: ${HQ_TASK_ID}" >&2
    exit 1
fi

task_list="${1:-${EVALUATION_TASK_LIST:-${SLURM_SUBMIT_DIR:-$(pwd)}/hq_tasks_evaluate}}"
task_log_dir="${LOCAL_SCRATCH}/logs"
mkdir -p "${task_log_dir}"
exec >"${task_log_dir}/${HQ_TASK_ID}.out" 2>"${task_log_dir}/${HQ_TASK_ID}.err"

echo "Running HyperQueue task ${HQ_TASK_ID} on ${SLURMD_NODENAME:-$(hostname)}"
echo "Task list: ${task_list}"

if [[ ! -f "${task_list}" ]]; then
    echo "Evaluation task list not found: ${task_list}" >&2
    exit 1
fi

task_command="$(sed -n "${HQ_TASK_ID}p" "${task_list}")"
if [[ -z "${task_command}" ]]; then
    echo "No command found on line ${HQ_TASK_ID} of ${task_list}" >&2
    exit 1
fi

mkdir -p "${LOCAL_SCRATCH}/output"
mkdir -p "${LOCAL_SCRATCH}/temp_blockstore"

eval "task_argv=(${task_command})"

local_project_root="${LOCAL_SCRATCH}"
local_config="${LOCAL_SCRATCH}/experiment_config.json"
local_output="${LOCAL_SCRATCH}/output"
local_temp_blockstore="${LOCAL_SCRATCH}/temp_blockstore"

if [[ "${task_argv[0]}" == "env" ]]; then
    command_start=1
    saw_project_root=0
    saw_config=0

    while (( command_start < ${#task_argv[@]} )); do
        if [[ "${task_argv[command_start]}" != *=* ]]; then
            break
        fi

        case "${task_argv[command_start]}" in
            PROJECT_ROOT=*)
                task_argv[command_start]="PROJECT_ROOT=${local_project_root}"
                saw_project_root=1
                ;;
            EXPERIMENT_CONFIG=*)
                task_argv[command_start]="EXPERIMENT_CONFIG=${local_config}"
                saw_config=1
                ;;
        esac

        ((command_start++))
    done

    injected_env=(
        "EXPERIMENT_OUTPUT_DIR=${local_output}"
        "EXPERIMENT_RESULT_FILE=${HQ_TASK_ID}.json"
        "TEMP_BLOCKSTORE_DIR=${local_temp_blockstore}"
    )
    if (( saw_project_root == 0 )); then
        injected_env+=("PROJECT_ROOT=${local_project_root}")
    fi
    if (( saw_config == 0 )); then
        injected_env+=("EXPERIMENT_CONFIG=${local_config}")
    fi

    task_argv=(
        "${task_argv[@]:0:${command_start}}"
        "${injected_env[@]}"
        "${task_argv[@]:${command_start}}"
    )
else
    task_argv=(
        env
        "PROJECT_ROOT=${local_project_root}"
        "EXPERIMENT_CONFIG=${local_config}"
        "EXPERIMENT_OUTPUT_DIR=${local_output}"
        "EXPERIMENT_RESULT_FILE=${HQ_TASK_ID}.json"
        "TEMP_BLOCKSTORE_DIR=${local_temp_blockstore}"
        "${task_argv[@]}"
    )
fi

printf 'Command:'
printf ' %q' "${task_argv[@]}"
printf '\n'

cd "${LOCAL_SCRATCH}"
"${task_argv[@]}"
