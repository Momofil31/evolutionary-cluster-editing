
# Cluster Editing with Evolutionary Algorithms

This repository contains the code and experiments related to the final project for the course Bio-Inspired Artificial Intelligence at the University of Trento. Experiments results are available on [Weights & Biases](https://wandb.ai/filippomomesso/cluster-edit).


## Introduction

Cluster Editing is a significant combinatorial optimization challenge. While classical optimization techniques have been the main focus in literature, this research explores the potential of evolutionary algorithms in addressing the problem. The repository provides the implementations of the proposed evolutionary strategies, their integration with heuristic local search techniques, and the experimental results.

## Setup

### 1. Conda Environment

Before running the experiments, make sure to set up a conda environment. If you don't have conda installed, you can get it from [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution).

```bash
# Create a new conda environment
conda create --name cluster_editing python=3.9

# Activate the environment
conda activate cluster_editing
```

### 2. Install Dependencies

After setting up the conda environment, you can install the required packages using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### 3. Setup Weights & Biases Logger

The repository uses [Weights & Biases](https://wandb.ai/) for logging the experiments. To use it, you need to create a free account and set the API key in the environment. Create a `.env` file in the root directory of the repository and add the following line:

```bash
WANDB_API_KEY=<your_api_key>
```

## Running Experiments

The repository utilizes [Hydra](https://hydra.cc/) for managing configurations. 

To run evolutionary experiment:

```bash
python src/evolution_experiment.py experiment=experiment_configuration
```

To run local search only experiment:

```bash
python src/local_search_experiment.py experiment=experiment_configuration
```

Replace `experiment_configuration` with the name of the configuration file you want to use (excluding the `.yaml` extension). Configuration files are located in the `configs` directory. You can also override single configuration parameters from the command line. For example, to override the `num_gens` parameter, you can run:

```bash
python src/evolution_experiment.py experiment=experiment_configuration evolution.num_gens=25
```


To use WandB logger for logging the experiments, add the `logger=wandb` flag.

## Results

After running the experiments, the results will be saved in the `outputs` directory. Each experiment will have its own folder named by the timestamp. If you used the WandB logger, the results will also be available on your Weights and Biases dashboard.

## Contributing

If you'd like to contribute or have suggestions/fixes, please open a GitHub issue or submit a pull request.
