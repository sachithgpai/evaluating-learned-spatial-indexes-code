# QDTree

The default experiment pipeline does not require a separate QDTree build step.
`Experiments/evaluate_all_indexes.cpp` includes `qdtree.h`, trains QDTree during each evaluation task, and writes the trained policy file under:

```text
Experiments/<dataset_name>/TrainedIndexes/QDTree/
```

The older Python files in this directory are kept for the reinforcement-learning QDTree experiments and are not used by `create_tasklist.sh`.
