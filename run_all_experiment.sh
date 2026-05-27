
REPO_ROOT="$(pwd)"
DATASET_NAME="dataset_synthetic"
CONFIG_PATH="${REPO_ROOT}/experiment_config.json"
EXPERIMENT_NAME="synthetic"

python3 Datasets/spatial_workload_generator.py synthetic \
  --config "${CONFIG_PATH}" \
  --experiment "${EXPERIMENT_NAME}" \
  --output-root "Datasets/${DATASET_NAME}"

cd "${REPO_ROOT}/Experiments"
g++ -std=c++17 evaluate_all_indexes.cpp -o build_evaluate.out
EXPERIMENT_CONFIG="${CONFIG_PATH}" bash create_tasklist.sh "${DATASET_NAME}" "${EXPERIMENT_NAME}"