import numpy as np
import pandas as pd
import os
from itertools import permutations

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
results_folder = "../results"
matrix_file = os.path.join(results_folder, "all_chakra_all_gland_cosine_matrix.csv")

# ------------------------------------------------------------
# Load the NEW cosine similarity matrix
# ------------------------------------------------------------
df = pd.read_csv(matrix_file, index_col=0)

# Make sure rows are in the intended chakra order
chakra_order = ["Heart", "Root", "Sacral", "Solar Plexus", "Third Eye", "Throat"]
gland_order = ["Adrenal", "Gonads", "Pancreas", "Pineal", "Thymus", "Thyroid"]

df = df.loc[chakra_order, gland_order]

chakras = list(df.index)
glands = list(df.columns)

M = df.values

# ------------------------------------------------------------
# Proposed mapping:
# Heart -> Thymus
# Root -> Adrenal
# Sacral -> Gonads
# Solar Plexus -> Pancreas
# Third Eye -> Pineal
# Throat -> Thyroid
# ------------------------------------------------------------
proposed_cols = [4, 0, 1, 2, 3, 5]

# ------------------------------------------------------------
# Observed proposed-pair mean
# ------------------------------------------------------------
observed = np.mean([M[i, proposed_cols[i]] for i in range(len(chakras))])

# ------------------------------------------------------------
# Off-diagonal mean
# ------------------------------------------------------------
off_diag_values = []
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        if j != proposed_cols[i]:
            off_diag_values.append(M[i, j])

off_diag_mean = np.mean(off_diag_values)

# ------------------------------------------------------------
# Exact permutation test over all 6! = 720 assignments
# ------------------------------------------------------------
perm_means = []
for perm in permutations(range(len(glands))):
    perm_mean = np.mean([M[i, perm[i]] for i in range(len(chakras))])
    perm_means.append(perm_mean)

perm_means = np.array(perm_means)
p_value = np.mean(perm_means >= observed)

# ------------------------------------------------------------
# Print results
# ------------------------------------------------------------
print("Observed proposed-pair mean:", round(float(observed), 4))
print("Off-diagonal mean:", round(float(off_diag_mean), 4))
print("Diagonal advantage:", round(float(observed - off_diag_mean), 4))
print("Permutation-test p-value:", round(float(p_value), 4))

# ------------------------------------------------------------
# Rank of proposed gland within each chakra row
# ------------------------------------------------------------
print("\nRank of proposed gland within each chakra row:")
for i, chakra in enumerate(chakras):
    row = M[i]
    sorted_indices = np.argsort(row)[::-1]  # descending
    rank = list(sorted_indices).index(proposed_cols[i]) + 1
    print(f"{chakra}: rank {rank} of {len(glands)}")
    