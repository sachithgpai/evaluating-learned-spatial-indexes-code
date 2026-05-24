REPO_ROOT="$(pwd)"
DATASET_NAME="dataset_synthetic_small"
CONFIG_PATH="${REPO_ROOT}/small_experiment_config.json"

cd Datasets
python3 spatial_workload_generator.py synthetic \
  --config "${CONFIG_PATH}" \
  --output-root "${DATASET_NAME}"

cd ../Experiments
g++ -std=c++17 evaluate_all_indexes.cpp -o build_evaluate.out
EXPERIMENT_CONFIG="${CONFIG_PATH}" bash create_tasklist.sh "${DATASET_NAME}" synthetic