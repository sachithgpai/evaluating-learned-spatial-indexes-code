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

# Where the mmap and paged block stores are written.
#
# An inherited value wins, so a batch script can place the store on node-local
# NVMe -- required for O_DIRECT, which Lustre refuses below 4096 bytes. Failing
# that, fall back to ${TMPDIR} when the site provides one (on Roihu it is a
# per-job NVMe area, created and removed automatically), and only then to the
# historical location under the repo root.
blockstore_dir="${TEMP_BLOCKSTORE_DIR:-${TMPDIR:+${TMPDIR}/blockstore}}"
blockstore_dir="${blockstore_dir:-${repo_root}/temp_blockstore}"
mkdir -p "${blockstore_dir}"

# Clear any orphaned scratch files before this task runs.
#
# A task that completes leaves nothing behind: PagedDiskBackend unlinks its file
# during Build() and holds it open by descriptor, so the space is reclaimed when
# the process exits. A task that is KILLED mid-Build -- an OOM kill or a walltime
# kill -- never reaches that unlink, and its file is orphaned with no process
# holding it. Since one array element runs several task lines in sequence, those
# orphans would otherwise pile up for the rest of the element.
#
# Deliberately scoped to the contents, not the directory, and guarded: an empty
# or root-ish blockstore_dir must never turn this into a destructive rm. This
# assumes one task at a time per blockstore dir, which is how the array script
# drives it (TEMP_BLOCKSTORE_DIR is per-job under ${TMPDIR}). Point two
# concurrent tasks at ONE shared directory and this would delete a live store.
if [[ -d "${blockstore_dir}" && "${blockstore_dir}" != "/" ]]; then
    orphans="$(find "${blockstore_dir}" -mindepth 1 -maxdepth 1 -type f | wc -l)"
    if (( orphans > 0 )); then
        echo "clearing ${orphans} orphaned blockstore file(s) from ${blockstore_dir}"
        find "${blockstore_dir}" -mindepth 1 -maxdepth 1 -type f -delete
    fi
fi

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
    # The task line already names its own result file, and THAT is the
    # authoritative name. Deriving it from ${line_number} instead is wrong the
    # moment the list is not the full hq_eval_tasks: the line number is a
    # position in WHATEVER list was passed, so re-running failures from a
    # filtered sub-list sends results to the file belonging to an unrelated
    # task. Results are opened with ios_base::app, so that appends rather than
    # truncates -- two parameter sets silently merged into one file, with no
    # error and a still-plausible row count. Fall back to the line number only
    # when the task line carries no filename of its own.
    if [[ "${command_tail[-1]}" == *.jsonl ]]; then
        set_env_assignment "EXPERIMENT_RESULT_FILE" "${command_tail[-1]}"
    else
        set_env_assignment "EXPERIMENT_RESULT_FILE" "${line_number}.jsonl"
    fi
    set_env_assignment "TEMP_BLOCKSTORE_DIR" "${blockstore_dir}"

    task_argv=(
        env
        "${env_assignments[@]}"
        "${command_tail[@]}"
    )
else
    task_argv=(
        env
        "PROJECT_ROOT=${repo_root}"
        "EXPERIMENT_RESULT_FILE=${line_number}.jsonl"
        "TEMP_BLOCKSTORE_DIR=${blockstore_dir}"
        "${task_argv[@]}"
    )
fi

printf 'Running evaluation task %s from %s\n' "${line_number}" "${task_list}"
printf 'Command:'
printf ' %q' "${task_argv[@]}"
printf '\n'

cd "${repo_root}"
"${task_argv[@]}"
