# Experiments

This folder contains the experiment runner for the repository.
It turns a generated dataset in `Datasets/<dataset_name>/...` into:

- task lists for batch execution
- trained index artifacts for methods that need offline training
- per-job JSONL result files

## Files

- `create_tasklist.sh`
  Generates `hq_tasks_evaluate` and `hq_tasks_RSMI`.
  It also creates the expected output folders under `Experiments/<dataset_name>/`.
  It reads experiment sizes, selectivities, and block sizes from `EXPERIMENT_CONFIG`, or from `../experiment_config.json` when `EXPERIMENT_CONFIG` is not set.

- `read_experiment_config.py`
  Extracts task-list loop bounds and selectivity tags from the selected config for `create_tasklist.sh`.

- `evaluate_all_indexes.cpp`
  Loads one dataset/query workload combination and evaluates the implemented index structures on that workload.
  The output is written as JSON lines into `Experiments/<dataset_name>/ResultsFolder/`.

- `hq_server_farm_jobs.sh`
  Slurm entrypoint for running the evaluation task list as a HyperQueue array on node-local scratch.
  It stages the dataset and trained RSMI models to `$LOCAL_SCRATCH`, runs each line of `hq_tasks_evaluate`,
  then copies `$LOCAL_SCRATCH/output/*.json` back to `Experiments/output/`.

- `hq_server_farm_rsmi_jobs.sh`
  Slurm entrypoint for training RSMI models with HyperQueue. It runs one array task per line in `hq_tasks_RSMI`
  without using `$LOCAL_SCRATCH`.

- `hq_stage_inputs.sh`, `hq_run_evaluate_task.sh`, `hq_archive_outputs.sh`
  Helper scripts used by `hq_server_farm_jobs.sh` for scratch staging, per-array-task execution, and result collection.

## Expected dataset layout

The experiment code expects a generated dataset folder under `Datasets/<dataset_name>/`, for example:

```text
Datasets/<dataset_name>/
  1/
    datapoints/
      1
      2
      3
      4
      5
      entropy_values
    queries/
      entropy_values
      otherDist/
        ...
```

`spatial_workload_generator.py` in the `Datasets` folder now generates this layout.

## End-to-End Workflow

First generate the dataset from the repository root. For example:

```bash
REPO_ROOT="$(pwd)"
DATASET_NAME="dataset_synthetic_full"
CONFIG_PATH="${REPO_ROOT}/experiment_config.json"
EXPERIMENT_NAME="synthetic"

python3 Datasets/spatial_workload_generator.py synthetic \
  --config "${CONFIG_PATH}" \
  --experiment "${EXPERIMENT_NAME}" \
  --output-root "Datasets/${DATASET_NAME}"
```

Then compile the evaluator and create task lists from inside `Experiments/`:

```bash
cd "${REPO_ROOT}/Experiments"
g++ -std=c++17 evaluate_all_indexes.cpp -o build_evaluate.out
EXPERIMENT_CONFIG="${CONFIG_PATH}" bash create_tasklist.sh "${DATASET_NAME}" "${EXPERIMENT_NAME}"
```

This creates:

- `hq_tasks_evaluate`
- `hq_tasks_RSMI`
- `Experiments/<dataset_name>/TrainedIndexes/`
- `Experiments/<dataset_name>/ResultsFolder/`
- `<repo>/temp_blockstore/`

RSMI models must be trained before evaluation. The generated task files are plain shell command lists, so a smoke test can run them directly:

```bash
bash -e hq_tasks_RSMI
bash -e hq_tasks_evaluate
```

For HyperQueue, start a server and workers in your environment, then submit the generated task files:

```bash
hq submit --each-line hq_tasks_RSMI
hq job wait all
hq submit --each-line hq_tasks_evaluate
hq job wait all
```

Direct shell and direct HyperQueue runs write results to `Experiments/<dataset_name>/ResultsFolder/`.

## Slurm Scratch Workflow

On Slurm clusters with node-local `$LOCAL_SCRATCH`, train the RSMI models first:

```bash
sbatch hq_server_farm_rsmi_jobs.sh
```

Then submit the evaluation farm:

```bash
EXPERIMENT_CONFIG=/path/to/the/same/config.json \
sbatch hq_server_farm_jobs.sh <dataset_name> [experiment_name]
```

The Slurm workflow copies `Datasets/<dataset_name>/`, `Experiments/<dataset_name>/TrainedIndexes/RSMI/`,
and the experiment config to each node's `$LOCAL_SCRATCH`. HyperQueue runs one array task per line in
`hq_tasks_evaluate`; task `N` writes to `$LOCAL_SCRATCH/output/N.json`. After all tasks complete, each
node copies its JSON files back into `Experiments/output/`.

The scratch workflow overrides only runtime paths. The task list itself can still be used directly with
`hq submit --each-line hq_tasks_evaluate` when local scratch staging is not needed.

Keep the same config for dataset generation, task-list generation, and Slurm evaluation. This matters for `small_experiment_config.json` and custom configs because the scratch wrapper copies the selected config to `$LOCAL_SCRATCH/experiment_config.json`.

## Evaluator CLI

The evaluator binary accepts:

```text
./build_evaluate.out <dataset_name> <data_sample_num> <data_ent_id> <block_size> <query_ent_id> <selectivity_id> [result_file]
```

Where:

- `data_sample_num` selects the outer dataset folder such as `Datasets/<dataset_name>/1/`
- `data_ent_id` selects the datapoint file inside `datapoints/`
- `query_ent_id` selects the query entropy variant inside `queries/otherDist/`
- `selectivity_id` maps to the order of `target_fractions` in the selected experiment config

If `result_file` is omitted, the evaluator now derives a unique JSONL filename automatically.

## Configuration

`create_tasklist.sh` defaults to the config's `default_experiment`.
Pass `[experiment_name]` or set `EXPERIMENT_NAME` to use another profile.
Set `EXPERIMENT_CONFIG=/path/to/experiment_config.json` if you want to generate tasks from a different config file.

The generated task commands pass `EXPERIMENT_CONFIG` and `EXPERIMENT_NAME` to `evaluate_all_indexes.cpp`, so the evaluator uses the same selectivity tags and query entropy counts as the task list.

Set `EXPERIMENT_OUTPUT_DIR` to write evaluator result files outside the default `Experiments/<dataset_name>/ResultsFolder/`.
Set `EXPERIMENT_RESULT_FILE` to override the optional `[result_file]` argument.
Set `TEMP_BLOCKSTORE_DIR` to move memory-mapped temporary blockstore files away from `<repo>/temp_blockstore/`.

## Output Locations

- RSMI training writes `.tree` and `.time` files to `Experiments/<dataset_name>/TrainedIndexes/RSMI/`.
- QDTree and FLOOD artifacts are created during evaluation under `Experiments/<dataset_name>/TrainedIndexes/`.
- Direct evaluator runs write JSONL files to `Experiments/<dataset_name>/ResultsFolder/`.
- Slurm scratch evaluation archives numbered JSON files to `Experiments/output/`.
