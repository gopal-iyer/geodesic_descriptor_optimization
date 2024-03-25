import numpy as np
import matplotlib.pyplot as plt
import ase
from ase.io import read, write
from dscribe.descriptors import ACSF
from information_imbalance import dist_quality
import sys
import re

system = '4_ethanol'
# system = '5_malonaldehyde'
# system = '6_aspirin'
n_g2 = 4
use_g4 = False

if use_g4:
    g4_code = 1
else:
    g4_code = 0

theory_B = 'direct' # the theory against which you want to optimize the imbalance
if theory_B == 'direct':
    theory_C = 'geodesic'
# elif theory_B == 'geodesic':
#     theory_C = 'direct'

fps_structures_file = open(f'../distance_matrices/{system}/path_100.txt', 'r')
lines = fps_structures_file.readlines()
    
structures_list = []
for line in lines:
    structure_fname = line[:-1]
    structure = read(f'../datasets/{system}/train/{structure_fname}')
    structures_list.append(structure)
fps_structures_file.close()

def extract_x_from_file(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            if line.strip().startswith('x:'):
                # Extract the list of numbers
                x_values = re.findall(r'[-+]?[\d.]+(?:e[-+]?\d+)?', line, re.IGNORECASE)
                # Convert to float and then to a numpy array
                x_array = np.array([float(value) for value in x_values])
                return x_array
    return None  # Return None if 'x' line is not found

def find_position_in_array(x, array_2d):
    for i, row in enumerate(array_2d):
        if np.array_equal(x, row):
            return i
    return -1

############## Settings ##############
def get_settings(code):
    if code == 0:
        flip = False
        minimize_combined = True
    elif code == 1:
        flip = False
        minimize_combined = False
    elif code == 2:
        flip = True
        minimize_combined = False
    
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
    
    if system == '4_ethanol' or system == '5_malonaldehyde':
        if objective_code == 'simple_objective':
            objective = 'direct'
        elif objective_code == 'difficult_objective':
            objective = 'geodesic'
        elif objective_code == 'combined_objective':
            objective = 'combined'
    elif system == '6_aspirin':
        if objective_code == 'simple_objective':
            objective = 'geodesic'
        elif objective_code == 'difficult_objective':
            objective = 'direct'
        elif objective_code == 'combined_objective':
            objective = 'combined'
    
    settings = {}
    settings["log_file_name"] = log_file_name
    settings["descriptor_params_iterations_file"] = descriptor_params_iterations_file
    settings["imbalance_objective_iterations_file"] = imbalance_objective_iterations_file
    settings["objective"] = objective
    
    return settings
########################################################

def make_descriptor(x, n_g2, use_g4):
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
    
    return acsf

codes = [0, 1, 2]

imbalance_matrix = {}

for i in range(len(codes)):
    settings_i = get_settings(i)
    optimal_descriptor_settings_i = extract_x_from_file(settings_i["log_file_name"])
    descriptor_i = make_descriptor(optimal_descriptor_settings_i, n_g2, use_g4)
    
    theory_A = settings_i["objective"]
    dmat_A = np.zeros((len(structures_list), len(structures_list)))
    for p in range(len(structures_list)):
        d_p = descriptor_i.create(structures_list[p]).flatten()
        for q in range(p+1, len(structures_list)):
            d_q = descriptor_i.create(structures_list[q]).flatten()
            dmat_A[p, q] = dmat_A[q, p] = np.linalg.norm(d_p - d_q)
    
    for j in range(len(codes)):
        settings_j = get_settings(j)
        optimal_descriptor_settings_j = extract_x_from_file(settings_j["log_file_name"])
        descriptor_j = make_descriptor(optimal_descriptor_settings_j, n_g2, use_g4)
        
        dict_key_i_to_j = f'Delta ({settings_i["objective"]} -> {settings_j["objective"]})'
        dict_key_j_to_i = f'Delta ({settings_j["objective"]} -> {settings_i["objective"]})'
        
        theory_B = settings_j["objective"]
        dmat_B = np.zeros((len(structures_list), len(structures_list)))
        for r in range(len(structures_list)):
            d_r = descriptor_j.create(structures_list[r]).flatten()
            for s in range(r+1, len(structures_list)):
                d_s = descriptor_j.create(structures_list[s]).flatten()
                dmat_B[r, s] = dmat_B[s, r] = np.linalg.norm(d_r - d_s)
            
        Delta_A_to_B, Delta_B_to_A, imbalance_AB = dist_quality(dmat_A, dmat_B, theory_A, theory_B, system)
        
        imbalance_matrix[dict_key_i_to_j] = Delta_A_to_B
        imbalance_matrix[dict_key_j_to_i] = Delta_B_to_A

print("---------------------------------------------------")

print(f"SYSTEM = {system}")
print("In the following, 'direct' refers to a descriptor that was Bayesian-optimized to")
print("minimize the symmetric information imbalance with the Euclidean distance,")
print("'geodesic' refers to a descriptor that was Bayesian-optimized to minimize")
print("the symmetric information imbalance with the geodesic distance, and")
print("'combined' refers to a descriptor that was Bayesian-optimized to minimize")
print("the sum of its information imbalance with both Euclidean and geodesic distances")
for key, value in imbalance_matrix.items():
    print(f"{key}: {value}")
