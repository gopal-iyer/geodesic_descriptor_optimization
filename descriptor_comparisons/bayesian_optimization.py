import numpy as np
import matplotlib.pyplot as plt
from skopt.plots import plot_gaussian_process
from skopt import gp_minimize
from skopt.space import Real, Integer
import ase
from ase.io import read, write
from dscribe.descriptors import ACSF
from information_imbalance import dist_quality
import sys

# system = '4_ethanol'
system = '5_malonaldehyde'
# system = '6_aspirin'

flip = True
minimize_combined = False
# allowed settings: (n_g2 = 2, use_g4 = False), (n_g2 = 4, use_g4 = False), (n_g2 = 4, use_g4 = True)
n_g2 = 4
use_g4 = True

theory_B = 'direct' # the theory against which you want to optimize the imbalance
if theory_B == 'direct':
    theory_C = 'geodesic'
# elif theory_B == 'geodesic':
#     theory_C = 'direct'

np.random.seed(42)

if use_g4:
    g4_code = 1
else:
    g4_code = 0
if not minimize_combined:
    if not flip:
        objective_code = 'difficult_objective'
    elif flip:
        objective_code = 'simple_objective'
else:
    objective_code = 'combined_objective'

save_file_code = f'{system}_{n_g2}g2_{g4_code}g4_{objective_code}'
log_file_name = f'log_{save_file_code}.txt'
descriptor_params_iterations_file = f'descriptor_params_{save_file_code}.npy'
imbalance_objective_iterations_file = f'imb_objective_{save_file_code}.npy'

f_log = open(log_file_name, 'w')
original_stdout = sys.stdout
sys.stdout = f_log

epsilon = 1e-6

dmat_B = np.load(f'../datasets/{system}/train/subset_100_{theory_B}_dmat.npy')
dmat_C = np.load(f'../datasets/{system}/train/subset_100_{theory_C}_dmat.npy')

Delta_B_to_C, Delta_C_to_B, imbalance_BC = dist_quality(dmat_B, dmat_C, theory_B, theory_C, system)

print("-----------------")

weaker_theory = None
stronger_theory = None

one_is_better_than_the_other = False
if Delta_B_to_C > Delta_C_to_B:
    weaker_theory = theory_B
    stronger_theory = theory_C
    one_is_better_than_the_other = True
elif Delta_C_to_B > Delta_B_to_C:
    weaker_theory = theory_C
    stronger_theory = theory_B
    one_is_better_than_the_other = True

if one_is_better_than_the_other:
    print(f"{weaker_theory} distance is found to be less informative than {stronger_theory} distance.")
    print(f"Since {stronger_theory} likely already contains the information in {weaker_theory},")
    print(f"it is likely that optimizing the descriptor to capture the information content in {weaker_theory}")
    print(f"will automatically capture the information in {stronger_theory}.")
    print(f"Optimizing imbalance with {weaker_theory}")
else:
    print(f"Both {theory_B} distance and {theory_C} distance are found to be equally informative.")
    print(f"Optimizing imbalance with just {theory_B}.")

print("-----------------")

fps_structures_file = open(f'../optimize_descriptors/{system}/path_100.txt', 'r')
lines = fps_structures_file.readlines()
    
structures_list = []
for line in lines:
    structure_fname = line[:-1]
    structure = read(f'../datasets/{system}/train/{structure_fname}')
    structures_list.append(structure)
fps_structures_file.close()

def f(x):
    rc = x[0] # range (1, 7) angstrom
    etas_g2 = np.logspace(np.log10(x[1]), np.log10(x[2]), num=n_g2) # range (epsilon, 1-epsilon) and (1+epsilon, 20) respectively, scan in log scale
    rs = x[3] # range (1, 7) angstrom
    if use_g4 and len(x) == 6:
        eta_g4 = x[4] # range 0.007 to 7, scan in log scale
        zeta_g4_max = x[5] # integer, scan 2 to 64 in log scale
        lamda_1 = 1
        lamda_2 = -1
    
    if n_g2 == 2:
        acsf = ACSF(
        species=["H", "C", "O"],
        r_cut=rc,
        g2_params=[[etas_g2[0], rs], [etas_g2[1], rs]],
        )
    
    elif n_g2 == 4 and not use_g4:
        acsf = ACSF(
        species=["H", "C", "O"],
        r_cut=rc,
        g2_params=[[etas_g2[0], rs], [etas_g2[1], rs], [etas_g2[2], rs], [etas_g2[3], rs]],
        )
    
    elif n_g2 == 4 and use_g4:
        acsf = ACSF(
        species=["H", "C", "O"],
        r_cut=rc,
        g2_params=[[etas_g2[0], rs], [etas_g2[1], rs], [etas_g2[2], rs], [etas_g2[3], rs]],
        g4_params=[[eta_g4, 1, lamda_1], [eta_g4, zeta_g4_max, lamda_1], [eta_g4, 1, lamda_2], [eta_g4, zeta_g4_max, lamda_2]],
        )
    
    descriptor_dmat = np.zeros((len(structures_list), len(structures_list)))
    
    for i in range(len(structures_list)):
        d_i = acsf.create(structures_list[i]).flatten()
        for j in range(i+1, len(structures_list)):
            d_j = acsf.create(structures_list[j]).flatten()
            descriptor_dmat[i, j] = descriptor_dmat[j, i] = np.linalg.norm(d_i - d_j)
    
    theory_A = 'descriptor'
    dmat_A = descriptor_dmat
    print("=================")
    Delta_A_to_B, Delta_B_to_A, imbalance_AB = dist_quality(dmat_A, dmat_B, theory_A, theory_B, system)
    print("-----------------")
    Delta_A_to_C, Delta_C_to_A, imbalance_AC = dist_quality(dmat_A, dmat_C, theory_A, theory_C, system)
    print("=================")
    if one_is_better_than_the_other:
        if theory_B == weaker_theory:
            if not flip:
                imbalance_to_optimize = imbalance_AB
            else:
                imbalance_to_optimize = imbalance_AC
        else:
            if not flip:
                imbalance_to_optimize = imbalance_AC
            else:
                imbalance_to_optimize = imbalance_AB
    else:
        imbalance_to_optimize = imbalance_AB
    
    if minimize_combined:
        imbalance_to_optimize = imbalance_AB + imbalance_AC
        print("Combined symmetric imbalance (A<->B + A<->C) = ", imbalance_to_optimize)
    
    return imbalance_to_optimize

space_no_g4  = [Real(1, 7, name='R_cut'),
                Real(epsilon, 1-epsilon, "log-uniform", name='etas_lower_bound'),
                Real(1+epsilon, 20, "log-uniform", name='etas_upper_bound'),
                Real(1, 7, name='R_s')]

space_with_g4  = [Real(1, 7, name='R_cut'),
                  Real(epsilon, 1-epsilon, "log-uniform", name='etas_lower_bound'),
                  Real(1+epsilon, 20, "log-uniform", name='etas_upper_bound'),
                  Real(1, 7, name='R_s'),
                  Real(0.007, 7, "log-uniform", name='eta_g4'),
                  Integer(2, 64, "log-uniform", name='zeta_g4_max')]

if use_g4:
    res = gp_minimize(f,                  # the function to minimize
                      space_with_g4,      # the bounds on each dimension of x
                      acq_func="EI",      # the acquisition function
                      n_calls=100,         # the number of evaluations of f
                      n_random_starts=50,  # the number of random initialization points
                    #   noise=0.1**2,       # the noise level (optional)
                      random_state=42)   # the random seed
else:
    res = gp_minimize(f,                  # the function to minimize
                      space_no_g4,      # the bounds on each dimension of x
                      acq_func="EI",      # the acquisition function
                      n_calls=100,         # the number of evaluations of f
                      n_random_starts=50,  # the number of random initialization points
                    #   noise=0.1**2,       # the noise level (optional)
                      random_state=42)   # the random seed

print(res)

np.save(descriptor_params_iterations_file, res.x_iters)
np.save(imbalance_objective_iterations_file, res.func_vals)

sys.stdout = original_stdout









