# About
This repository includes scripts for optimizing ACSF descriptor hyperparameters using Bayesian optimization to minimize the information imbalance between descriptor distances, and Cartesian-Euclidean and/or geodesic distances.

# Usage
1. `datasets` directory contains the training data structures for ethanol, malonaldehyde, and aspirin used in this work. These structures were borrowed from the [MD22 dataset](http://www.sgdml.org/).
2. `descriptor_comparisons` contains scripts to (i) perform Bayesian optimization of descriptor hyperparameters (`bayesian_optimization.py`), (ii) generate the multi-descriptor heatmap data corresponding to Fig. 5 in the paper (`generate_heatmap_data.py`), and (iii) compute the information imbalance between two distance measures (`information_imbalance.py`), as described in [Glielmo et al. (2022)](https://academic.oup.com/pnasnexus/article/1/2/pgac039/6568571).
3. `distance_matrices` contains scripts to first use a farthest-point sampling algorithm to sample a set of dissimilar structures from the dataset (`locate_path.py`) and compute pair-wise geodesic distances between each of these sampled structures (`geodesic_length.py`).
