# Evaluating Learned Spatial Indexes

This repository hosts the code of the article:
"[Experiments & Analysis] Evaluating Learned Spatial Indexes" by Sachith Pai and Michael Mathioudakis, 2025.

The repository holds the C++ implementation for Indexes used for experimentation and scripts to reproduce the results.


## Reproducing the experiments
Experiment sizes and selectivities are configured in `experiment_config.json` at the repository root.
The dataset generator, task-list generator, and evaluator all read this file.

### Full Synthetic Experiment
Run these commands from the repository root:

```bash
REPO_ROOT="$(pwd)"
DATASET_NAME="dataset_synthetic_full"
CONFIG_PATH="${REPO_ROOT}/experiment_config.json"

cd Datasets
python3 spatial_workload_generator.py synthetic \
  --config "${CONFIG_PATH}" \
  --output-root "${DATASET_NAME}"

cd ../Experiments
g++ -std=c++17 evaluate_all_indexes.cpp -o build_evaluate.out
EXPERIMENT_CONFIG="${CONFIG_PATH}" bash create_tasklist.sh "${DATASET_NAME}" synthetic
```

This creates `hq_tasks_RSMI` and `hq_tasks_evaluate` in `Experiments/`.
Train the RSMI models first, then run the full evaluator:

```bash
hq submit --each-line hq_tasks_RSMI
hq submit --each-line hq_tasks_evaluate
```

You can add scheduler options such as `--cpus=X` to either `hq submit` command.
Results are written under `Experiments/<dataset_name>/ResultsFolder_ExtendBlockSize/`.

### Quick Smoke Test
Use `small_experiment_config.json` for quick local checks:

```bash
REPO_ROOT="$(pwd)"
DATASET_NAME="dataset_synthetic_small"
CONFIG_PATH="${REPO_ROOT}/small_experiment_config.json"

cd Datasets
python3 spatial_workload_generator.py synthetic \
  --config "${CONFIG_PATH}" \
  --output-root "${DATASET_NAME}"

cd ../Experiments
g++ -std=c++17 evaluate_all_indexes.cpp -o build_evaluate.out
EXPERIMENT_CONFIG="${CONFIG_PATH}" bash create_tasklist.sh "${DATASET_NAME}" synthetic
```

For real-world parquet-backed datasets and OSM conversion, see `Datasets/README.md`.







## Authors and acknowledgment
The code is created and maintained by Sachith Pai (sachith.pai@helsinki.fi).
The work was supported by Michael Mathioudakis's Academy of Finland grants.


## Project status
Manuscript under review.
