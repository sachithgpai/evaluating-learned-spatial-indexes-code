#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: bash evaluate_line_n.sh <line_number> [task_list]" >&2
    exit 1
fi

line_number="$1"
if [[ ! "${line_number}" =~ ^[0-9]+$ || "${line_number}" -lt 1 ]]; then
    echo "line_number must be a positive integer, got: ${line_number}" >&2
    exit 1
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

default_task_list="${script_dir}/hq_eval_tasks"
if [[ ! -f "${default_task_list}" && -f "${script_dir}/hq_tasks_evaluate" ]]; then
    default_task_list="${script_dir}/hq_tasks_evaluate"
fi
task_list="${2:-${EVALUATION_TASK_LIST:-${default_task_list}}}"

if [[ "${task_list}" != /* ]]; then
    task_list="$(cd "$(dirname "${task_list}")" && pwd)/$(basename "${task_list}")"
fi

if [[ ! -f "${task_list}" ]]; then
    echo "Evaluation task list not found: ${task_list}" >&2
    exit 1
fi

task_command="$(sed -n "${line_number}p" "${task_list}")"
if [[ -z "${task_command}" ]]; then
    echo "No command found on line ${line_number} of ${task_list}" >&2
    exit 1
fi

mkdir -p "${repo_root}/temp_blockstore"

eval "task_argv=(${task_command})"

set_env_assignment() {
    local env_name="$1"
    local env_value="$2"
    local arg_index

    for ((arg_index = 0; arg_index < ${#env_assignments[@]}; arg_index++)); do
        if [[ "${env_assignments[arg_index]}" == "${env_name}="* ]]; then
            env_assignments[arg_index]="${env_name}=${env_value}"
            return
        fi
    done

    env_assignments+=("${env_name}=${env_value}")
}

if [[ "${task_argv[0]:-}" == "env" ]]; then
    command_start=1
    while (( command_start < ${#task_argv[@]} )); do
        if [[ "${task_argv[command_start]}" != *=* ]]; then
            break
        fi
        ((command_start++))
    done

    env_assignments=("${task_argv[@]:1:$((command_start - 1))}")
    command_tail=("${task_argv[@]:${command_start}}")

    set_env_assignment "PROJECT_ROOT" "${repo_root}"
    set_env_assignment "TEMP_BLOCKSTORE_DIR" "${repo_root}/temp_blockstore"

    task_argv=(
        env
        "${env_assignments[@]}"
        "${command_tail[@]}"
    )
else
    task_argv=(
        env
        "PROJECT_ROOT=${repo_root}"
        "TEMP_BLOCKSTORE_DIR=${repo_root}/temp_blockstore"
        "${task_argv[@]}"
    )
fi

printf 'Running evaluation task %s from %s\n' "${line_number}" "${task_list}"
printf 'Command:'
printf ' %q' "${task_argv[@]}"
printf '\n'

cd "${repo_root}"
"${task_argv[@]}"
