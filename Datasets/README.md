# Dataset generation

`spatial_workload_generator.py` creates the dataset and query files used by the experiment runner.
It generates both:

- `otherDist/*_areabased_*` query files for QDTree training
- `otherDist/*_countbased_*` query files for evaluation

Experiment sizes and selectivities are read from the project-level config at `../experiment_config.json`.
If you change that file, the workload generator, task-list script, and evaluator read the same values.

## Synthetic workloads

From inside the `Datasets` directory:

```bash
python3 spatial_workload_generator.py synthetic \
  --output-root dataset_synthetic
```

This writes the experiment input files under `dataset_synthetic/`:

```text
dataset_synthetic/
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

## Real workloads from parquet

If you already have a parquet with `lat` and `lon` columns:

```bash
python3 spatial_workload_generator.py real \
  --parquet-path planet_latlon.parquet \
  --world-grid-path count_grid_0p05x0p1_deg.npz \
  --output-root dataset_real \
  --build-world-grid
```

Each real sample is written under `dataset_real/<sample_id>/`.

Use `--build-world-grid` when you want to rebuild the `.npz` count grid from parquet instead of reusing an existing one.

Use `--experiment <name>` to select a different experiment profile from `../experiment_config.json`, or `--config <path>` to point at another config file.

## Converting OSM PBF to parquet

`OSM-2d-Parquet.py` creates the parquet expected by the `real` subcommand.

```bash
python3 OSM-2d-Parquet.py \
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
