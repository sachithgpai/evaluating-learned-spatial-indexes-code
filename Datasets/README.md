# Dataset generation

`spatial_workload_generator.py` replaces both `create_gmm_datasets.py` and `create_countbased_query.cpp`.
It keeps the legacy directory layout and filenames that the existing experiment code expects, while generating both:

- `otherDist/*_areabased_*` query files for QDTree training
- `otherDist/*_countbased_*` query files for evaluation

The default selectivities match the old pipeline:

- `00064` -> `0.000064`
- `00256` -> `0.000256`
- `01024` -> `0.001024`
- `04096` -> `0.004096`
- `16384` -> `0.016384`

The current experiment scripts still assume five data entropy variants, five query entropy variants, and the legacy selectivity tags above. If you change those values, update the experiment code as well.

## Synthetic workloads

From inside the `Datasets` directory:

```bash
python3 spatial_workload_generator.py synthetic \
  --output-root dataset_synthetic \
  --n-points 8000000 \
  --n-queries 1000 \
  --num-datasets 5 \
  --num-query-scales 5 \
  --num-query-clusters 5 \
  --synthetic-num-clusters 10 \
  --target-fractions 0.000064 0.000256 0.001024 0.004096 0.016384
```

This writes a legacy-compatible layout under `dataset_synthetic/`:

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
        1_00064_areabased_1
        1_00064_countbased_1
        ...
  2/
  3/
  4/
  5/
```

Mapping to the old scripts:

- outer folder `1..5` is `data_sample_num`
- datapoint file `1..5` is `data_ent_id`
- query suffix `..._<query_ent_id>` is the query entropy variant

## Real workloads from parquet

If you already have a parquet with `lat` and `lon` columns:

```bash
python3 spatial_workload_generator.py real \
  --parquet-path planet_latlon.parquet \
  --world-grid-path count_grid_0p05x0p1_deg.npz \
  --output-root dataset_real \
  --build-world-grid \
  --num-samples 5 \
  --real-target-points 8000000 \
  --n-queries 1000 \
  --num-query-scales 5 \
  --num-query-clusters 5 \
  --target-fractions 0.000064 0.000256 0.001024 0.004096 0.016384
```

Each real sample is written under `dataset_real/<sample_id>/`.
For real mode, the datapoint file is stored as `datapoints/1`, with matching `queries/entropy_values` and `queries/otherDist/1_<selectivity>_*_<query_ent_id>` files.

Use `--build-world-grid` when you want to rebuild the `.npz` count grid from parquet instead of reusing an existing one.

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

## Python API

```python
from spatial_workload_generator import GeneratorConfig, SpatialWorkloadGenerator

cfg = GeneratorConfig(
    synthetic_n_points=8_000_000,
    n_queries=1_000,
)

gen = SpatialWorkloadGenerator(cfg, "dataset_synthetic")
gen.run_synthetic()
```

For real workloads:

```python
from spatial_workload_generator import GeneratorConfig, SpatialWorkloadGenerator

cfg = GeneratorConfig(
    real_target_points=8_000_000,
    n_queries=1_000,
)

gen = SpatialWorkloadGenerator(cfg, "dataset_real")
gen.run_real(
    parquet_path="planet_latlon.parquet",
    world_grid_path="count_grid_0p05x0p1_deg.npz",
    build_world_grid=False,
)
```

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
