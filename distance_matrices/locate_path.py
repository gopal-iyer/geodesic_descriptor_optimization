import os
import numpy as np
import pickle
from copy import deepcopy

def main():
    # system = '4_ethanol'
    # system = '5_malonaldehyde'
    system = '6_aspirin'
    n_path_images = 80
    # max 40 (N_train) for H2
    # max 80 (N_train) for CO2
    # max 800 (N_train) for H2O
    # max 1000 (N_train) for ethanol
    # max 1000 (N_train) for malonaldehyde
    # max 1000 (N_train) for aspirin
    alpha = 0.0
    # 'alpha' is a factor that dictates what power law assigns the relative
    # importance of more recent structures compared to older ones when
    # performing farthest point sampling to grow the constructed path.
    # For example, alpha = 2.0 means the importance assigned to older structures
    # decays as a square law, while alpha = 0.0 means all structures have the
    # same importance. No prefactor is included to the decay law for now.
    # One could also implement exponential decay.
    
    assert n_path_images >= 2, "Number of path images must be greater than or equal to 2"
    
    dmat_fname = os.path.join(os.path.join('../datasets', system), 'train/dmat.npy')
    i2f_fname = os.path.join(os.path.join('../datasets', system), 'train/idx2file_table.pkl')
    f2i_fname = os.path.join(os.path.join('../datasets', system), 'train/file2idx_table.pkl')
    
    dmat = np.load(dmat_fname)
    with open(i2f_fname, 'rb') as f:
        i2f = pickle.load(f)
    with open(f2i_fname, 'rb') as f:
        f2i = pickle.load(f)
    
    structure_path = []
    structure_idx_path = []
    
    max_dist_pair = np.unravel_index(np.argmax(dmat), dmat.shape)
    
    i = max_dist_pair[0]
    j = max_dist_pair[1]
    
    sum_distances = np.linalg.norm(dmat[:, [i, j]], axis=1)
    sum_distances[[i, j]] = -np.inf # artificially setting points already seen to -ve infinity so they are masked out
    max_dist_idx = np.argmax(sum_distances)
    
    if dmat[i, max_dist_idx] >= dmat[j, max_dist_idx]:
        structure_path.append(i2f[j])
        structure_path.append(i2f[i])
        structure_idx_path.append(j)
        structure_idx_path.append(i)
    
    else:
        structure_path.append(i2f[i])
        structure_path.append(i2f[j])
        structure_idx_path.append(i)
        structure_idx_path.append(j)
    
    if n_path_images > 2:
        structure_path.append(i2f[max_dist_idx])
        structure_idx_path.append(max_dist_idx)
    
        for p in range(n_path_images - 3):
            # sum_distances = np.linalg.norm(dmat[:, structure_idx_path], axis=1)
            sum_distances = np.sqrt(np.average(np.square(dmat[:, structure_idx_path]), axis=1,
                                       weights=np.arange(1, len(structure_idx_path)+1, 1)**alpha))
            sum_distances[structure_idx_path] = -np.inf # artificially setting points already seen to -ve infinity so they are masked out
            max_dist_idx = np.argmax(sum_distances)
            structure_path.append(i2f[max_dist_idx])
            structure_idx_path.append(max_dist_idx)
    
    f = open(os.path.join(system, f'path_{n_path_images}.txt'), 'w')
    for structure in structure_path:
        f.write(structure)
        f.write('\n')
    f.close()

if __name__ == '__main__':
    main()
