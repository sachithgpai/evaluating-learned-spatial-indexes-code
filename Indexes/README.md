# Indexes

This folder contains the index implementations used by the experiments.

The code is organized by index family:

- `KDTree/`
  Binary KD-tree structures.
- `WAZI/`
  Z-order based indexes, including the base ZTree, sampling-aware ZTree, and ZM index.
- `RTree/`
  Classical and learned R-tree style variants.
- `FLOOD/`
  FLOOD grid index and its random-search trainer.
- `QDTree/`
  Query-driven tree implementation and its trainer.
- `utils/`
  Shared geometry primitives, block storage, sorting helpers, and density estimators.

Notes:

- `utils/json.hpp` and `utils/pgm/` are vendored dependencies and should generally be treated as external code.
- Most index implementations share the `BlockStore` abstraction in `utils/local_model.h` for block materialization and scanning.
- There is no separate build step for this folder in the default pipeline. `Experiments/evaluate_all_indexes.cpp` includes these headers and is compiled from `Experiments/`.
- RSMI training is the separate Python pre-evaluation step generated in `Experiments/hq_tasks_RSMI`. QDTree and FLOOD training happen inside the evaluator.
- Query execution typically follows the same high-level pattern:
  1. projection to candidate cells or blocks
  2. refinement using bounding metadata
  3. scanning blocks to return matching points
