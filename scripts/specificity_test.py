import numpy as np
import pandas as pd
from itertools import permutations

# Final matrix values from your confirmed results
chakras = ["Heart", "Root", "Sacral", "Solar Plexus", "Third Eye", "Throat"]
glands = ["Adrenal", "Gonads", "Pancreas", "Pineal", "Thymus", "Thyroid"]

M = np.array([
    [0.0247, 0.0323, 0.0136, 0.0658, 0.0468, 0.0358],  # Heart
    [0.0263, 0.0314, 0.0199, 0.0511, 0.0420, 0.0281],  # Root
    [0.0477, 0.0603, 0.0350, 0.0992, 0.0783, 0.0685],  # Sacral
    [0.0230, 0.0189, 0.0087, 0.0125, 0.0137, 0.0129],  # Solar Plexus
    [0.0523, 0.0667, 0.0353, 0.0664, 0.0901, 0.0619],  # Third Eye
    [0.0532, 0.0619, 0.0272, 0.0953, 0.0913, 0.0688],  # Throat
])

# Proposed mapping:
# Heart->Thymus, Root->Adrenal, Sacral->Gonads, Solar Plexus->Pancreas, Third Eye->Pineal, Throat->Thyroid
proposed_cols = [4, 0, 1, 2, 3, 5]

# Observed diagonal mean
observed = np.mean([M[i, proposed_cols[i]] for i in range(len(chakras))])

# Off-diagonal mean
off_diag_values = []
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        if j != proposed_cols[i]:
            off_diag_values.append(M[i, j])
off_diag_mean = np.mean(off_diag_values)

# Exact permutation test over all 6! = 720 assignments
perm_means = []
for perm in permutations(range(len(glands))):
    perm_mean = np.mean([M[i, perm[i]] for i in range(len(chakras))])
    perm_means.append(perm_mean)

perm_means = np.array(perm_means)
p_value = np.mean(perm_means >= observed)

print("Observed proposed-pair mean:", round(observed, 4))
print("Off-diagonal mean:", round(off_diag_mean, 4))
print("Diagonal advantage:", round(observed - off_diag_mean, 4))
print("Permutation-test p-value:", round(p_value, 4))

# Rank of proposed gland within each row
print("\nRank of proposed gland within each chakra row:")
for i, chakra in enumerate(chakras):
    row = M[i]
    sorted_indices = np.argsort(row)[::-1]  # descending
    rank = list(sorted_indices).index(proposed_cols[i]) + 1
    print(f"{chakra}: rank {rank} of {len(glands)}")
    