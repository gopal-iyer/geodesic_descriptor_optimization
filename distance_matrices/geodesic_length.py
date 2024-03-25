import numpy as np
import os
from ase.io import read, write
import sys
from copy import deepcopy
import pickle
import matplotlib.pyplot as plt
import re
import subprocess

sys.path.insert(1, '../datasets')
from distance_matrices import kabsch_rotation

def compute_geodesic_length(prefix, n_fps, fname_structure_1, fname_structure_2, n_images=27, maxiter=100, microiter=100):
    # Collecting structures to interpolate between
    path_structure_1 = os.path.join(f'../datasets/{prefix}/train/', fname_structure_1)
    path_structure_2 = os.path.join(f'../datasets/{prefix}/train/', fname_structure_2)
    
    P = structure_1 = read(path_structure_1)
    Q = structure_2 = read(path_structure_2)
    
    # Generating copies of the images that are centroid-shifted and rotated to ensure maximum alignment
    P_shift, Q_rot = kabsch_rotation(P, Q)
    
    # Files to initialize the geodesic interpolation
    dummy_1 = f'subset_{n_fps}_{prefix}_dummy_file_1.xyz'
    dummy_2 = f'subset_{n_fps}_{prefix}_dummy_file_2.xyz'
    input_file = f'subset_{n_fps}_{prefix}_endpoint_structures.xyz'
    output_file = f'subset_{n_fps}_{prefix}_temp_output_file.txt'
    
    write(dummy_1, P_shift)
    write(dummy_2, Q_rot)
    
    f1 = open(dummy_1, 'r')
    l1 = f1.readlines()
    f1.close()
    f2 = open(dummy_2, 'r')
    l2 = f2.readlines()
    f2.close()
    
    l_combined = l1 + l2
    
    f_combined = open(input_file, 'w')
    f_combined.writelines(l_combined)
    f_combined.close()
    
    fname_interpolated = f'../datasets/{prefix}/train/subset_{n_fps}_geo_path_structure_{fname_structure_1[10:-4]}_shift_{fname_structure_2[10:-4]}_rot.xyz'
    
    # Running geodesic interpolation
    if not os.path.exists(fname_interpolated):
        cmd = f'geodesic_interpolate {input_file} --output {fname_interpolated} --nimages {n_images} --maxiter {maxiter} --microiter {microiter} &> {output_file}'
        os.system(cmd)
        # result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        # with open(output_file, 'w') as file:
        #     file.write(result.stdout)  # Write standard output
        #     file.write(result.stderr)  # Write standard error (if any)
    
    geodesic_path = read(f'{fname_interpolated}@:') # Note that the '@:' at the end is important to ensure all structures are read
    
    # Regular expression to match the required values
    pattern = r"Final path length:\s+(\d+\.\d+)\s+Max RMSD in path:\s+(\d+\.\d+)"
    
    # Read the file and extract the values
    with open(output_file, 'r') as file:
        for line in file:
            match = re.search(pattern, line)
            if match:
                distance, max_rmsd = match.groups()
    
    geodesic_distance = distance
    
    direct_distance = np.linalg.norm(geodesic_path[-1].positions - geodesic_path[0].positions)
    # direct_mass_weighted_distance = np.sqrt(
    #                                   np.average(
    #                                       np.square(
    #                                           np.linalg.norm(geodesic_path[-1].positions - geodesic_path[0].positions, axis=-1)),
    #                                           weights=masses))
    
    fig_fname = f'../datasets/{prefix}/train/subset_{n_fps}_plot_geo_dist_structure_{fname_structure_1[10:-4]}_shift_{fname_structure_2[10:-4]}_rot.pdf'
    
    cmd = f'rm {dummy_1} {dummy_2} {input_file} {output_file} {fname_interpolated}'
    os.system(cmd)
    
    return geodesic_distance, direct_distance

def main(n_fps):
    # system = '4_ethanol'
    # system = '5_malonaldehyde'
    system = '6_aspirin'
    
    structures_subset_fname = f'{system}/path_{n_fps}.txt'
    f_sub = open(structures_subset_fname, 'r')
    structures_subset = f_sub.readlines()
    f_sub.close()
    structures_subset = [ss[:-1] for ss in structures_subset]
    
    i2f_fname = f'../datasets/{system}/train/idx2file_table.pkl'
    f2i_fname = f'../datasets/{system}/train/file2idx_table.pkl'
    
    with open(i2f_fname, 'rb') as f:
        i2f = pickle.load(f)
    with open(f2i_fname, 'rb') as f:
        f2i = pickle.load(f)
    
    structures_subset_indices = [f2i[ss] for ss in structures_subset]
    
    # geodesic_dmat = np.full((len(i2f), len(i2f)), np.nan)
    subset_geodesic_dmat = np.zeros((len(structures_subset_indices), len(structures_subset_indices)))
    subset_direct_dmat = np.zeros((len(structures_subset_indices), len(structures_subset_indices)))
    
    # only for plotting
    geodesic_distance_list = []
    direct_distance_list = []
    
    for i in range(len(structures_subset_indices)):
        for j in range(i+1, len(structures_subset_indices)):
            p = structures_subset_indices[i]
            q = structures_subset_indices[j]
            # geodesic_dmat[p, q], direct_D_pq = compute_geodesic_length(system, i2f[p], i2f[q], n_images=27)
            # geodesic_dmat[q, p] = geodesic_dmat[p, q]
            subset_geodesic_dmat[i, j], subset_direct_dmat[i, j] = compute_geodesic_length(system, n_fps, i2f[p], i2f[q], n_images=27)
            subset_geodesic_dmat[j, i] = subset_geodesic_dmat[i, j]
            subset_direct_dmat[j, i] = subset_direct_dmat[i, j]
            
            # collecting data for plotting
            geodesic_distance_list.append(subset_geodesic_dmat[i, j])
            # direct_distance_list.append(direct_D_pq)
            direct_distance_list.append(subset_direct_dmat[i, j])
            
    # np.fill_diagonal(geodesic_dmat, 0.0)

    subset_geodesic_dmat_fname = f'../datasets/{system}/train/subset_{n_fps}_geodesic_dmat.npy'
    np.save(subset_geodesic_dmat_fname, subset_geodesic_dmat)
    subset_direct_dmat_fname = f'../datasets/{system}/train/subset_{n_fps}_direct_dmat.npy'
    np.save(subset_direct_dmat_fname, subset_direct_dmat)
    
    plt.figure()
    plt.scatter(np.array(direct_distance_list), np.array(geodesic_distance_list), color='blue', marker='.')
    plt.xlabel('Direct Eucl. dist. init->final (Å)')
    plt.ylabel('Geodesic dist. (no units)')
    plt.savefig(f'../datasets/{system}/train/subset_{n_fps}_plot_direct_vs_geodesic_scatter.pdf')
    plt.clf()

if __name__ == '__main__':
    main(60)
