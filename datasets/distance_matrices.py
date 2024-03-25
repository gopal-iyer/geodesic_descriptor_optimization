import os
import numpy as np
from ase.io import read
from scipy.spatial.transform import Rotation
from copy import deepcopy
import pickle

def kabsch_rotation(P, Q):
    """Aligns the points in Q to the points in P."""
    P_shift = deepcopy(P)
    Q_rot = deepcopy(Q)
    
    centroid_0 = np.average(P.get_positions(), axis=0, weights=P.get_masses())
    pos_0 = P.get_positions() - centroid_0
    P_shift.set_positions(pos_0)
    
    centroid = np.average(Q.get_positions(), axis=0, weights=Q.get_masses())
    pos = Q.get_positions() - centroid
    
    rotation_matrix, rmsd = Rotation.align_vectors(pos_0, pos)
    Q_rot.set_positions(np.matmul(rotation_matrix.as_matrix(), pos.T).T)
    
    return P_shift, Q_rot

def load_structures(directory):
    structures = []
    structure_names = []
    for filename in os.listdir(directory):
        if filename.startswith("structure_") and filename.endswith(".xyz"):
            path = os.path.join(directory, filename)
            structure = read(path)
            structures.append(structure)
            structure_names.append(filename)
    return structures, structure_names

def calculate_distance_matrix(structures, structure_names):
    num_structures = len(structures)
    distance_matrix = np.zeros((num_structures, num_structures))
    for i in range(num_structures):
        for j in range(i + 1, num_structures):
            P = structures[i]
            Q = structures[j]
            P_shift, Q_rot = kabsch_rotation(P, Q)
            dist = np.linalg.norm(P_shift.get_positions() - Q_rot.get_positions())
            distance_matrix[i, j] = distance_matrix[j, i] = dist
    return distance_matrix

def main():
    for directory in ['4_ethanol/train', '5_malonaldehyde/train', '6_aspirin/train']:
        structures, structure_names = load_structures(directory)
        distance_matrix = calculate_distance_matrix(structures, structure_names)
    
        dmat_fname = os.path.join(directory, 'dmat.npy')
        
        idx2file_table = {}
        file2idx_table = {}
        for i in range(len(structures)):
            idx2file_table[i] = structure_names[i]
            file2idx_table[structure_names[i]] = i
        idx2file_table_fname = os.path.join(directory, 'idx2file_table.pkl')
        file2idx_table_fname = os.path.join(directory, 'file2idx_table.pkl')
        
        np.save(dmat_fname, distance_matrix)
        with open(idx2file_table_fname, 'wb') as f:
            pickle.dump(idx2file_table, f)
        with open(file2idx_table_fname, 'wb') as f:
            pickle.dump(file2idx_table, f)

if __name__ == "__main__":
    main()
