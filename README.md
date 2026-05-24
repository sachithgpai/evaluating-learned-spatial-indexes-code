# Evaluating Learned Spatial Indexes

This repository hosts the code of the article:
"[Experiments & Analysis] Evaluating Learned Spatial Indexes" by Sachith Pai and Michael Mathioudakis, 2025.

The repository contains the C++ index implementations, workload generators, and experiment scripts used to reproduce the results.

## Repository Layout

- `Datasets/`
  Workload generation for synthetic data, real parquet-backed data, and OSM-to-parquet conversion.
- `Experiments/`
  Evaluator, task-list generation, and HyperQueue/Slurm wrappers.
- `Indexes/`
  Index implementations used by the evaluator.
- `experiment_config.json`
  Full experiment configuration.
- `small_experiment_config.json`
  Small configuration for smoke tests.
- `run_all_experiment.sh`
  Convenience script, run with `bash run_all_experiment.sh`, that generates the small synthetic workload, compiles the evaluator, and creates task lists. Run the generated task lists afterward to train and evaluate.

## Requirements

For the synthetic pipeline:

- Python 3 with `numpy`
- A C++17 compiler such as `g++`
- Python packages used by RSMI training: `torch`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`, and the `zCurve` module imported by `Indexes/RTree/RSMI.py`

For real workloads, also install `pyarrow`. For OSM PBF conversion, also install `osmium`.

HyperQueue is only needed when submitting task lists through `hq` or the Slurm wrappers.

## End-to-End Synthetic Pipeline

Run from the repository root. Use `experiment_config.json` for the full experiment:

```bash
REPO_ROOT="$(pwd)"
DATASET_NAME="dataset_synthetic_full"
CONFIG_PATH="${REPO_ROOT}/experiment_config.json"
EXPERIMENT_NAME="synthetic"

python3 Datasets/spatial_workload_generator.py synthetic \
  --config "${CONFIG_PATH}" \
  --experiment "${EXPERIMENT_NAME}" \
  --output-root "Datasets/${DATASET_NAME}"

cd "${REPO_ROOT}/Experiments"
g++ -std=c++17 evaluate_all_indexes.cpp -o build_evaluate.out
EXPERIMENT_CONFIG="${CONFIG_PATH}" bash create_tasklist.sh "${DATASET_NAME}" "${EXPERIMENT_NAME}"
```

This creates:

- `Experiments/hq_tasks_RSMI`
- `Experiments/hq_tasks_evaluate`
- `Experiments/<dataset_name>/TrainedIndexes/`
- `Experiments/<dataset_name>/ResultsFolder/`
- `temp_blockstore/`

Train RSMI first, then run the evaluator. For a small local run, the task files are plain shell command lists:

```bash
cd "${REPO_ROOT}/Experiments"
bash -e hq_tasks_RSMI
bash -e hq_tasks_evaluate
```

For HyperQueue, start a server and workers in your environment, then submit the same task files:

```bash
cd "${REPO_ROOT}/Experiments"
hq submit --each-line hq_tasks_RSMI
hq job wait all
hq submit --each-line hq_tasks_evaluate
hq job wait all
```

Local and direct HyperQueue runs write JSONL results under `Experiments/<dataset_name>/ResultsFolder/`.

### Quick Smoke Test

Use the small config when you only want to verify the pipeline:

```bash
REPO_ROOT="$(pwd)"
DATASET_NAME="dataset_synthetic_small"
CONFIG_PATH="${REPO_ROOT}/small_experiment_config.json"
EXPERIMENT_NAME="synthetic"

python3 Datasets/spatial_workload_generator.py synthetic \
  --config "${CONFIG_PATH}" \
  --experiment "${EXPERIMENT_NAME}" \
  --output-root "Datasets/${DATASET_NAME}"

cd "${REPO_ROOT}/Experiments"
g++ -std=c++17 evaluate_all_indexes.cpp -o build_evaluate.out
EXPERIMENT_CONFIG="${CONFIG_PATH}" bash create_tasklist.sh "${DATASET_NAME}" "${EXPERIMENT_NAME}"

bash -e hq_tasks_RSMI
bash -e hq_tasks_evaluate
```

## Slurm Pipeline

After generating the dataset, compiling `build_evaluate.out`, and creating task lists, train RSMI first:

```bash
cd "${REPO_ROOT}"
sbatch Experiments/hq_server_farm_rsmi_jobs.sh
```

Then run evaluation on node-local scratch:

```bash
EXPERIMENT_CONFIG="${CONFIG_PATH}" \
sbatch Experiments/hq_server_farm_jobs.sh "${DATASET_NAME}" "${EXPERIMENT_NAME}"
```

Keep `EXPERIMENT_CONFIG` set to the same config used by `create_tasklist.sh`, especially when using `small_experiment_config.json` or a custom config. The Slurm evaluation wrapper stages the dataset, trained RSMI files, and config to `$LOCAL_SCRATCH`. It writes per-task JSON files to `$LOCAL_SCRATCH/output/` and archives them into `Experiments/output/`.

## Real Workloads

For parquet-backed real workloads and OSM conversion, see `Datasets/README.md`. The rest of the pipeline is the same: generate `Datasets/<dataset_name>/`, compile the evaluator, run `create_tasklist.sh <dataset_name> real`, train RSMI, then evaluate.

## Authors and acknowledgment
The code is created and maintained by Sachith Pai (sachith.pai@helsinki.fi).
The work was supported by Michael Mathioudakis's Academy of Finland grants.


## Project status
Manuscript under review.
