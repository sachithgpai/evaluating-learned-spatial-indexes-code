# Evaluating Learned Spatial Indexes

This repository hosts the code of the article:
"[Experiments & Analysis] Evaluating Learned Spatial Indexes" by Sachith G Pai, Jun Yang & Michael Mathioudakis 2026.

The repository contains the C++ index implementations, workload generators, and experiment scripts used to reproduce the results.

## Repository Layout

- `Datasets/`
  Workload generation for synthetic data, real parquet-backed data, and OSM-to-parquet conversion.
- `Experiments/`
  Evaluator, task-list generation, and Slurm array wrappers.
- `Indexes/`
  Index implementations used by the evaluator.
- `experiment_config.json`
  Full experiment configuration.
- `small_experiment_config.json`
  Small configuration for smoke tests.
- `run_all_experiment.sh`
  Convenience script, run with `bash run_all_experiment.sh`, that generates the full synthetic workload from `experiment_config.json` into `Datasets/dataset_synthetic/`, compiles the evaluator, and creates task lists. Run the generated task lists afterward to train and evaluate.

## Requirements

For the synthetic pipeline:

- Python 3 with `numpy`
- A C++17 compiler such as `g++`
- Python packages used by RSMI training: `torch`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`, and the `zCurve` module imported by `Indexes/RTree/RSMI.py`
- Python packages used by QDTree training: `torch`, `torchrl` and `tensordict`, imported by `Indexes/QDTree/Train_QdTree.py` and `QdTree_environment.py`. Install the three together — they are released in lockstep and a mismatched set fails at import.

Versions in the requirements files float, so the artifact keeps installing on
current Python releases. The two exceptions are deliberate and commented in
place: `scikit-learn>=1.4` (for `root_mean_squared_error`) and an exact
`zCurve==0.0.4`, whose `interlace()` output *is* the WAZI/ZM point ordering and
so would silently restructure those indexes if it changed.

Install the Python dependencies with:

```bash
python3 -m pip install -r requirements-synthetic.txt
```

For real workloads and OSM PBF conversion, install the extended requirements:

```bash
python3 -m pip install -r requirements-real.txt
```

The real requirements include the synthetic requirements, plus `pyarrow` and `osmium`.

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
- `Experiments/hq_eval_tasks`
- `Experiments/slurm_rsmi_array.sh`
- `Experiments/slurm_evaluate_array.sh`
- `Experiments/<dataset_name>/TrainedIndexes/`
- `Experiments/<dataset_name>/ResultsFolder/`
- `temp_blockstore/`

`Experiments/evaluate_line_n.sh` and `Experiments/rsmi_line_n.sh` are checked into the
repository rather than generated; `create_tasklist.sh` only makes them executable.

Train RSMI first, then run the evaluator. For a small local run, the task files are plain shell command lists:

```bash
cd "${REPO_ROOT}/Experiments"
bash -e hq_tasks_RSMI
bash -e hq_eval_tasks
```

To run one evaluation task by line number:

```bash
cd "${REPO_ROOT}/Experiments"
bash evaluate_line_n.sh 1
```

Evaluation runs write JSONL results under `Experiments/<dataset_name>/ResultsFolder/`.
Each evaluation task writes `<line_number>.jsonl`, where the line number is its row in `hq_eval_tasks`.
Memory-mapped temporary blockstore files are created under `temp_blockstore/` and removed by the evaluator.

## Configuration

Experiment sizes, selectivities, block sizes and the storage-backend settings all
come from `experiment_config.json`; `create_tasklist.sh` reads it and bakes the
relevant values into each generated task command. The knobs worth knowing about
when running the evaluator by hand:

- `EXPERIMENT_SEED` (default `42`) — seeds the random searches FLOOD and QD perform. Fixed, so one task is reproducible; vary it deliberately to sample independent search paths.
- `ENABLE_PAGED_BACKEND` and `BUFFER_POOL_DIRECT_IO` — the binary defaults both off, but the shipped config turns both **on**, so a task list measures the paged backend with direct I/O while a bare `build_evaluate.out` invocation does not.
- `TEMP_BLOCKSTORE_DIR` — must name node-local storage for direct I/O; Lustre and NFS commonly refuse it.
- `PROJECT_ROOT` — override instead of rebuilding when the checkout moves.

`Experiments/README.md` documents all of them, with the binary default beside
each, and `Indexes/utils/tests/README.md` covers the ones specific to the
storage-backend unit tests.

## Plotting Results

The plotting script lives in `Results/plot_results.py`. First combine the per-task
JSONL files into `Results/<dataset_name>/Results.json`, then run the plotting script:

The plotting script is designed for the full experiment configuration used by the
paper. Small configurations are useful for pipeline checks, but some figures
assume the full set of data-skew, query-skew, selectivity, and block-size settings.

```bash
DATASET_NAME="dataset_synthetic_full"

mkdir -p "Results/${DATASET_NAME}"
cat "Experiments/${DATASET_NAME}/ResultsFolder/"*.jsonl \
  > "Results/${DATASET_NAME}/Results.json"

python3 Results/plot_results.py \
  --data-dir "Results/${DATASET_NAME}" \
  --figures-dir "Results/${DATASET_NAME}/figures" \
  --no-tex
```

The generated figures and derived result files are written under
`Results/<dataset_name>/figures/`. See `Results/README.md` for more details.

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
bash -e hq_eval_tasks
```

## Slurm Pipeline

After generating the dataset, compiling `build_evaluate.out`, and creating task lists, train RSMI first:

```bash
cd "${REPO_ROOT}/Experiments"
sbatch slurm_rsmi_array.sh
```

Then submit the generated evaluation array script:

```bash
EXPERIMENT_CONFIG="${CONFIG_PATH}" \
sbatch slurm_evaluate_array.sh
```

Keep `EXPERIMENT_CONFIG` set to the same config used by `create_tasklist.sh`, especially when using `small_experiment_config.json` or a custom config. The Slurm evaluation array reads data and writes results in the repository checkout directly; it does not stage data to node-local scratch or copy results back.
Run both Slurm commands from `Experiments/`; the generated scripts use `SLURM_SUBMIT_DIR` to find `*_line_n.sh` and the task lists.

## Real Workloads

For parquet-backed real workloads and OSM conversion, see `Datasets/README.md`. The rest of the pipeline is the same: generate `Datasets/<dataset_name>/`, compile the evaluator, run `create_tasklist.sh <dataset_name> real`, train RSMI, then evaluate.

## Authors and acknowledgment
The code is created and maintained by Sachith Pai (sachith.pai@helsinki.fi,sachithgpai@gmail.com).
The work was supported by Michael Mathioudakis's Academy of Finland grants.


## Project status
Manuscript under review.
