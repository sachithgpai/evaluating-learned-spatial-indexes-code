# Experiments

This folder contains the experiment runner for the repository.
It is responsible for turning a generated dataset in `Datasets/<dataset_name>/...` into:

- task lists for batch execution
- trained index artifacts for methods that need offline training
- per-job JSONL result files

## Files

- `create_tasklist.sh`
  Generates `hq_tasks_evaluate` and `hq_tasks_RSMI`.
  It also creates the expected output folders under `Experiments/<dataset_name>/`.

- `evaluate_all_indexes.cpp`
  Loads one dataset/query workload combination and evaluates the implemented index structures on that workload.
  The output is written as JSON lines into `Experiments/<dataset_name>/ResultsFolder_ExtendBlockSize/`.

## Expected dataset layout

The experiment code expects a dataset folder produced in the legacy layout, for example:

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
        1_00064_countbased_1
        1_00064_areabased_1
        ...
```

`spatial_workload_generator.py` in the `Datasets` folder now generates this layout.

## Workflow

From inside `Experiments/`:

```bash
g++ -std=c++17 evaluate_all_indexes.cpp -o build_evaluate.out
bash create_tasklist.sh <dataset_name>
```

This creates:

- `hq_tasks_evaluate`
- `hq_tasks_RSMI`
- `Experiments/<dataset_name>/TrainedIndexes/`
- `Experiments/<dataset_name>/ResultsFolder_ExtendBlockSize/`
- `<repo>/temp_blockstore/`

You can then submit the generated task files with HyperQueue:

```bash
hq submit --each-line hq_tasks_RSMI
hq submit --each-line hq_tasks_evaluate
```

## Evaluator CLI

The evaluator binary accepts:

```text
./build_evaluate.out <dataset_name> <data_sample_num> <data_ent_id> <block_size> <query_ent_id> <selectivity_id> [result_file]
```

Where:

- `data_sample_num` selects the outer dataset folder such as `Datasets/<dataset_name>/1/`
- `data_ent_id` selects the datapoint file inside `datapoints/`
- `query_ent_id` selects the query entropy variant inside `queries/otherDist/`
- `selectivity_id` maps to:
  - `0 -> 00064`
  - `1 -> 00256`
  - `2 -> 01024`
  - `3 -> 04096`
  - `4 -> 16384`

If `result_file` is omitted, the evaluator now derives a unique JSONL filename automatically.

## Current assumptions

The current experiment scripts still assume:

- 5 dataset samples
- 5 datapoint entropy variants per sample
- 5 query entropy variants per datapoint variant
- 5 selectivity levels with the legacy tags above

If you change those counts in the dataset generator, update `create_tasklist.sh` and the indexing logic in `evaluate_all_indexes.cpp` as well.
