
# "Inference for Batched Bandits" 
This repo is to reproduce the main results from the "Inference for Batched Bandits" paper by Kelly W. Zhang, Lucas Janson, and Susan A. Murphy (https://arxiv.org/abs/2002.03217).

## Requirements
- python (tested for python 3.7)
- numpy
- scipy
- matplotlib

## Files
### `run_simulation.py`
Main file for running batched bandits simulations (runs multiple Monte Carlo runs in parallel). Includes the sampling algorithms (Thompson Sampling, epsilon-greedy, ect.). Saves the results from the run in the `simulations` folder, which is automatically created.

### `process.py`
This is the main file for performing inference on the data generated from the batched bandit simulations. Includes the OLS, BOLS, W-Decorrelated, and AW-AIPW estimators. Creates plots of Type-1 error and power.

### `write_nonstationary.py`
This script generates the expected rewards for both arms for the non-stationary simulations used in the paper. The script saves these files in the `nonstationary_means` directory, which is automatically created.

### `process_nonstationary.py`
This script generates the power plots for the non-stationary treatment effect setting (must be run after running `process.py`).

### `utils.py`
This file includes some useful functions used by the other main scripts.

### `run.sh`
This bash script can be run to reproduce the main results from the paper.


