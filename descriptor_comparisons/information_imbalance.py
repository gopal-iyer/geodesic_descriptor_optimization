import numpy as np
import matplotlib.pyplot as plt

def dist_quality(dmat_a, dmat_b, theory_A, theory_B, system, plot=False):
    fps_a = dmat_a
    fps_b = dmat_b
    assert len(fps_a) == len(fps_b), "Lengths of fingerprint sets do not match!"

    sorted_ranks_a = []
    sorted_ranks_b = []

    ordered_indices_a = np.arange(len(fps_a))
    ordered_indices_b = np.arange(len(fps_b))

    Delta_A_to_B = np.array([])
    Delta_B_to_A = np.array([])
    
    identical = 0
    non_identical = 0

    for i in range(len(fps_a)):
        dists_a = fps_a[i]
        dists_b = fps_b[i]

        zip_a = zip(dists_a, ordered_indices_a)
        sorted_a = sorted(zip_a)
        zip_b = zip(dists_b, ordered_indices_b)
        sorted_b = sorted(zip_b)

        shuffled_indices_a = np.array([r for _, r in sorted_a])
        shuffled_indices_b = np.array([r for _, r in sorted_b])
        
        if shuffled_indices_a.tolist() == shuffled_indices_b.tolist():
            identical += 1
        else:
            non_identical += 1

        # Find the rank of each structure using measures A and B based on its closeness to the reference structure.
        # Exclude the 'nearest' structure because this is actually the same as the reference structure.
        for j in range(len(shuffled_indices_a)):
            # if not np.where(shuffled_indices_a == j)[0][0] == 0:
            sorted_ranks_a += [np.where(shuffled_indices_a == j)[0][0]]
            # if not np.where(shuffled_indices_b == j)[0][0] == 0:
            sorted_ranks_b += [np.where(shuffled_indices_b == j)[0][0]]

        Delta_A_to_B = np.append(
                            Delta_A_to_B, sorted_ranks_b[
                            np.where(np.array(sorted_ranks_a) == 1)[0][0]
                            ])
        Delta_B_to_A = np.append(
                            Delta_B_to_A, sorted_ranks_a[
                            np.where(np.array(sorted_ranks_b) == 1)[0][0]])

    print(f'{identical} cases found where orders are identical')
    print(f'{non_identical} cases found where orders are non-identical')
    
    sorted_ranks_a = np.array(sorted_ranks_a)
    sorted_ranks_b = np.array(sorted_ranks_b)

    Delta_A_to_B = 2. * np.mean(Delta_A_to_B) / len(fps_a)
    Delta_B_to_A = 2. * np.mean(Delta_B_to_A) / len(fps_b)

    imbalance = (Delta_A_to_B + Delta_B_to_A) / np.sqrt(2)

    print(f"Delta_{theory_A}_to_{theory_B} = ", Delta_A_to_B)
    print(f"Delta_{theory_B}_to_{theory_A} = ", Delta_B_to_A)
    print(f"Symmetric_imbalance({theory_A}<->{theory_B}) = ", imbalance)
    if plot:
        plt.figure()
        plt.scatter(sorted_ranks_b, sorted_ranks_a, marker='.', color='r')
        plt.xlabel(f'$r$ ({theory_B})')
        plt.ylabel(f'$r$ ({theory_A})')
        plt.savefig(f'{system}_{theory_A}_vs_{theory_B}_information_imbalance.pdf')
        plt.close()

    return Delta_A_to_B, Delta_B_to_A, imbalance

def main():
    systems_group_1 = [
                       '4_ethanol',
                       #'5_malonaldehyde',
                       #'6_aspirin',
                      ]
    point_subset_sizes = [20, 40, 60, 80, 100]
    
    theory_A = 'direct'
    theory_B = 'geodesic'
    
    for system in systems_group_1:
        print(f'---   {system}   ---')
        for pts in point_subset_sizes:
            print(f'{pts} points')
            dmat_A = np.load(f'../datasets/{system}/train/subset_{pts}_{theory_A}_dmat.npy')
            dmat_B = np.load(f'../datasets/{system}/train/subset_{pts}_{theory_B}_dmat.npy')
            Delta_A_to_B, Delta_B_to_A, imbalance = dist_quality(dmat_A, dmat_B, theory_A, theory_B, system, plot=True)
            print(f'end')
        print(f'--- end {system} ---')
    
if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
