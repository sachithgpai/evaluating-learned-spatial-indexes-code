# Results

This folder is for consolidating experiment output and generating the paper figures.
The plotting script is designed for the full experiment configuration used by the
paper. Small configurations are useful for checking that the pipeline runs, but
some paper figures assume the full set of data-skew, query-skew, selectivity,
and block-size settings.

The experiment runner writes one JSONL file per evaluation task under:

```text
Experiments/<data_folder_name>/ResultsFolder/
```

For plotting, combine those individual files into:

```text
Results/<data_folder_name>/Results.json
```

For example, for a full synthetic dataset named `dataset_synthetic_full`:

```bash
mkdir -p Results/dataset_synthetic_full
cat Experiments/dataset_synthetic_full/ResultsFolder/*.jsonl \
  > Results/dataset_synthetic_full/Results.json
```

Then write all generated figures and derived result files under:

```text
Results/<data_folder_name>/figures/
```

## Expected Layout

The plotting script lives in this folder. Each experiment run should get its own
subfolder named after the dataset folder used by the experiment runner:

```text
Results/
  README.md
  plot_results.py
  <data_folder_name>/
    Results.json
    Results.pkl
    figures/
      ...
```

`Results.pkl` is an optional cache created by `plot_results.py`.

## Consolidate Evaluation JSONL Files

Run these commands from the repository root. Replace `<data_folder_name>` with the
dataset folder name used under `Experiments/`, for example `dataset_synthetic_full`.

```bash
DATA_FOLDER_NAME="<data_folder_name>"

mkdir -p "Results/${DATA_FOLDER_NAME}"
cat "Experiments/${DATA_FOLDER_NAME}/ResultsFolder/"*.jsonl \
  > "Results/${DATA_FOLDER_NAME}/Results.json"
```

Check that the consolidated file is non-empty:

```bash
wc -l "Results/${DATA_FOLDER_NAME}/Results.json"
```

The line count should match the total number of JSON result rows emitted by the
evaluation tasks.

## Generate Figures

Run the plotting script from the repository root:

```bash
DATA_FOLDER_NAME="<data_folder_name>"

python3 Results/plot_results.py \
  --data-dir "Results/${DATA_FOLDER_NAME}" \
  --figures-dir "Results/${DATA_FOLDER_NAME}/figures"
```

If the machine does not have a full LaTeX installation, add `--no-tex`:

```bash
python3 Results/plot_results.py \
  --data-dir "Results/${DATA_FOLDER_NAME}" \
  --figures-dir "Results/${DATA_FOLDER_NAME}/figures" \
  --no-tex
```

## End-to-End Example

```bash
DATA_FOLDER_NAME="dataset_synthetic_full"

mkdir -p "Results/${DATA_FOLDER_NAME}"
cat "Experiments/${DATA_FOLDER_NAME}/ResultsFolder/"*.jsonl \
  > "Results/${DATA_FOLDER_NAME}/Results.json"

python3 Results/plot_results.py \
  --data-dir "Results/${DATA_FOLDER_NAME}" \
  --figures-dir "Results/${DATA_FOLDER_NAME}/figures" \
  --no-tex
```
