# Evaluating Learned Spatial Indexes

This repository hosts the code of the article:
"[Experiments & Analysis] Evaluating Learned Spatial Indexes" by Sachith Pai and Michael Mathioudakis, 2025.

The repository holds the C++ implementation for Indexes used for experimentation and scripts to reproduce the results.


## Reproducing the experiments
### Create a set synthetic dataset
Run the following commands to create a dataset to be used for your experiments.

```
cd Datasets
python create_gmm_datasets.py <dataset_name> <data_size> <query_size>
g++ -std=c++17 create_countbased_query.cpp -o create_countbased_query.out
./create_countbased_query.out <dataset_name>
cd ..
```


### Create the list of configurations to execute
Run the script to create the list of configurations to execute.
```
cd Experiments
bash create_tasklist.sh <dataset_name>
```
This script creates bash files `hq_tasks_evaluate` and `hq_tasks_RSMI` with list of jobs to run for generating the experimental results.


### Experimental Evaluation.

Since the number of individual jobs are very high, it is suitable to use the HyperQueue tool for this purpose. OR use any other job scheduler tool available to execute the jobs listed in the two `hq_tasks_...` files.

First we need to train and store the RSMI models to use in experiments.
```
hq submit --each-line hq_tasks_RSMI
```
You can add an option `--cpus=X` to define the number of CPUs cores to use for each job.
This will train and store the RSMI indexes for each data configuration in `Experiments/<dataset_name>/TrainedIndexes/RSMI`. These learned models will be used by the evaluation script.

Then we can run the main evaluation script:
```
g++ -std=c++17 evaluate_all_indexes.cpp -o build_evaluate.out
hq submit --each-line hq_tasks_evaluate
```

### Consodidating results and Plotting.







## Authors and acknowledgment
The code is created and maintained by Sachith Pai (sachith.pai@helsinki.fi).
The work was supported by Michael Mathioudakis's Academy of Finland grants.


## Project status
Manuscript under review.