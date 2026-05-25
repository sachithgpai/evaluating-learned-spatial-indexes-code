#!/usr/bin/env bash

set -euo pipefail

: "${LOCAL_SCRATCH:?LOCAL_SCRATCH must be set by the Slurm node-local storage environment.}"

archive_dir="${1:-${SLURM_SUBMIT_DIR:-$(pwd)}/output}"
mkdir -p "${archive_dir}"
node_name="${SLURMD_NODENAME:-$(hostname)}"
job_id="${SLURM_JOB_ID:-manual}"

shopt -s nullglob
json_files=("${LOCAL_SCRATCH}"/output/*.json)
log_files=("${LOCAL_SCRATCH}"/logs/*)

if (( ${#json_files[@]} == 0 )); then
    echo "No JSON outputs found on ${node_name}"
else
    for json_file in "${json_files[@]}"; do
        cp "${json_file}" "${archive_dir}/"
    done
    echo "Archived ${#json_files[@]} JSON output files from ${node_name} to ${archive_dir}"
fi

if (( ${#log_files[@]} > 0 )); then
    log_archive_dir="${archive_dir}/logs/${job_id}/${node_name}"
    mkdir -p "${log_archive_dir}"
    for log_file in "${log_files[@]}"; do
        [[ -f "${log_file}" ]] || continue
        cp "${log_file}" "${log_archive_dir}/"
    done
    echo "Archived ${#log_files[@]} task log files from ${node_name} to ${log_archive_dir}"
else
    echo "No task logs found on ${node_name}"
fi
