# Dataset generation

`spatial_workload_generator.py` creates the dataset and query files used by the experiment runner.
It generates both:

- `otherDist/*_areabased_*` query files for QDTree training
- `otherDist/*_countbased_*` query files for evaluation

Experiment sizes and selectivities are read from the project-level config. By default this is `experiment_config.json` at the repository root, unless you pass `--config` or set `EXPERIMENT_CONFIG`.

The generated dataset folder must live under `Datasets/<dataset_name>/` for the experiment runner.

## Synthetic workloads

From the repository root:

```bash
python3 Datasets/spatial_workload_generator.py synthetic \
  --config experiment_config.json \
  --experiment synthetic \
  --output-root Datasets/dataset_synthetic
```

This writes the experiment input files under `Datasets/dataset_synthetic/`:

```text
Datasets/dataset_synthetic/
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
  2/
  3/
  4/
  5/
```

For a quick smoke dataset, use the small config:

```bash
python3 Datasets/spatial_workload_generator.py synthetic \
  --config small_experiment_config.json \
  --experiment synthetic \
  --output-root Datasets/dataset_synthetic_small
```

## Real workloads from parquet

If you already have a parquet with `lat` and `lon` columns:

```bash
python3 Datasets/spatial_workload_generator.py real \
  --config experiment_config.json \
  --experiment real \
  --parquet-path planet_latlon.parquet \
  --world-grid-path count_grid_0p05x0p1_deg.npz \
  --output-root Datasets/dataset_real \
  --build-world-grid
```

Each real sample is written under `Datasets/dataset_real/<sample_id>/`.

Use `--build-world-grid` when you want to rebuild the `.npz` count grid from parquet instead of reusing an existing one.

Use `--experiment <name>` to select a different experiment profile from the config. If omitted, the generator uses `synthetic` for synthetic mode and `real` for real mode.

After generation, run the experiment pipeline from `Experiments/`:

```bash
cd Experiments
g++ -std=c++17 evaluate_all_indexes.cpp -o build_evaluate.out
EXPERIMENT_CONFIG=../experiment_config.json bash create_tasklist.sh dataset_real real
```

Then train RSMI before running evaluation.

## Converting OSM PBF to parquet

`OSM-2d-Parquet.py` creates the parquet expected by the `real` subcommand.

```bash
python3 Datasets/OSM-2d-Parquet.py \
  planet-latest.osm.pbf \
  planet_latlon.parquet \
  --keep-every 10 \
  --batch-size 500000
```

The output parquet contains only two columns:

- `lat`
- `lon`

`--keep-every` lets you subsample the OSM node stream before writing parquet.

## Dependencies

Required:

- `numpy`

Needed for real-mode parquet support:

- `pyarrow`

Needed for OSM conversion:

- `osmium`
- `pyarrow`

Optional:

- `matplotlib` for datapoint and query overlay plots
