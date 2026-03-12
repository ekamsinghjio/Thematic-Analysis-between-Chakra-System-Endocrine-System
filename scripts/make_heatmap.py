import matplotlib.pyplot as plt
import numpy as np

# Your final cosine similarity matrix
data = np.array([
    [0.0247, 0.0323, 0.0136, 0.0658, 0.0468, 0.0358],  # Heart
    [0.0263, 0.0314, 0.0199, 0.0511, 0.0420, 0.0281],  # Root
    [0.0477, 0.0603, 0.0350, 0.0992, 0.0783, 0.0685],  # Sacral
    [0.0230, 0.0189, 0.0087, 0.0125, 0.0137, 0.0129],  # Solar Plexus
    [0.0523, 0.0667, 0.0353, 0.0664, 0.0901, 0.0619],  # Third Eye
    [0.0532, 0.0619, 0.0272, 0.0953, 0.0913, 0.0688],  # Throat
])

row_labels = ["Heart", "Root", "Sacral", "Solar Plexus", "Third Eye", "Throat"]
col_labels = ["Adrenal", "Gonads", "Pancreas", "Pineal", "Thymus", "Thyroid"]

fig, ax = plt.subplots(figsize=(10, 6))

# Create heatmap
im = ax.imshow(data, aspect="auto")

# Axis labels
ax.set_xticks(np.arange(len(col_labels)))
ax.set_yticks(np.arange(len(row_labels)))
ax.set_xticklabels(col_labels)
ax.set_yticklabels(row_labels)

# Rotate x labels for readability
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# Add value labels in each cell
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        ax.text(j, i, f"{data[i, j]:.3f}", ha="center", va="center")

# Title and colorbar
ax.set_title("Cosine Similarity Between Chakra Corpora and Endocrine Gland Corpora")
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Cosine Similarity")

fig.tight_layout()

# Save figure
plt.savefig("../results/chakra_gland_heatmap.png", dpi=300, bbox_inches="tight")
plt.show()
