#!/usr/bin/env bash

set -euo pipefail

: "${LOCAL_SCRATCH:?LOCAL_SCRATCH must be set by the Slurm node-local storage environment.}"

archive_dir="${1:-${SLURM_SUBMIT_DIR:-$(pwd)}/output}"
mkdir -p "${archive_dir}"

shopt -s nullglob
json_files=("${LOCAL_SCRATCH}"/output/*.json)

if (( ${#json_files[@]} == 0 )); then
    echo "No JSON outputs found on ${SLURMD_NODENAME:-$(hostname)}"
    exit 0
fi

for json_file in "${json_files[@]}"; do
    cp "${json_file}" "${archive_dir}/"
done

echo "Archived ${#json_files[@]} JSON output files from ${SLURMD_NODENAME:-$(hostname)} to ${archive_dir}"
