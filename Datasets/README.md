# Dataset generation

`spatial_workload_generator.py` creates the dataset and query files used by the experiment runner.
It writes datapoints, query centers, area-based query boxes, and count-based query boxes.
The current evaluator reads the `otherDist/*_countbased_*` query files.
The generated `otherDist/*_areabased_*` files are not used by the default evaluation pipeline.

Experiment sizes and selectivities are read from the project-level config. By default this is `experiment_config.json` at the repository root, unless you pass `--config` or set `EXPERIMENT_CONFIG`.

The generated dataset folder must live under `Datasets/<dataset_name>/` for the experiment runner.
Use `--experiment <name>` to select a different experiment profile from the config. If omitted, the generator uses `synthetic` for synthetic mode and `real` for real mode.

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
  <sample_id>/
    datapoints/
      <data_entropy_id>
      entropy_values
    queries/
      entropy_values
      otherDist/
        <data_entropy_id>_<selectivity>_countbased_<query_entropy_id>
        <data_entropy_id>_<selectivity>_countbased_meta_<query_entropy_id>.txt
        <data_entropy_id>_<selectivity>_areabased_<query_entropy_id>
        <data_entropy_id>_querycenters_<query_entropy_id>.csv
```

For a quick smoke dataset, use the small config:

```bash
python3 Datasets/spatial_workload_generator.py synthetic \
  --config small_experiment_config.json \
  --experiment synthetic \
  --output-root Datasets/dataset_synthetic_small
```

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

When the selected real experiment profile has `single_query_workload_per_sample: true`,
the generator writes one randomly selected query workload per sample instead of the full
query-entropy/selectivity grid. The selected pair is recorded in:

```text
Datasets/dataset_real/<sample_id>/queries/selected_workload.json
```

That JSON includes the selected `data_entropy_id`, `query_entropy_id`,
normalized `query_entropy`, `selectivity_id`, `selectivity_tag`,
`target_fraction`, `target_count`, `n_queries`, and the generated query file paths.

Use `--build-world-grid` when you want to rebuild the `.npz` count grid from parquet instead of reusing an existing one.

After generation, run the experiment pipeline from `Experiments/`:

```bash
cd Experiments
g++ -std=c++17 evaluate_all_indexes.cpp -o build_evaluate.out
EXPERIMENT_CONFIG=../experiment_config.json bash create_tasklist.sh dataset_real real
```

Then train RSMI before running evaluation.

## Dependencies

For synthetic dataset generation:

- `numpy`

For real-mode parquet support:

- `pyarrow`

For OSM conversion:

- `osmium`
- `pyarrow`

Optional:

- `matplotlib` for datapoint and query overlay plots

From the repository root, `requirements-synthetic.txt` installs the synthetic
generation dependencies, and `requirements-real.txt` installs the real-workload
dependencies.
