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
```

### Create a set synthetic dataset



## Authors and acknowledgment
The code is created and maintained by Sachith Pai (sachith.pai@helsinki.fi).
The work was supported by the Academy of Finland.


## Project status
Manuscript under review.