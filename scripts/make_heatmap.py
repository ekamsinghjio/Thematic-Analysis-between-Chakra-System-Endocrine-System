import os
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
results_folder = "../results"
matrix_file = os.path.join(results_folder, "all_chakra_all_gland_cosine_matrix.csv")
output_file = os.path.join(results_folder, "updated_chakra_gland_heatmap.png")

# ------------------------------------------------------------
# Load the UPDATED cosine similarity matrix
# ------------------------------------------------------------
df = pd.read_csv(matrix_file, index_col=0)

# Optional: enforce consistent row/column order
chakra_order = ["Heart", "Root", "Sacral", "Solar Plexus", "Third Eye", "Throat"]
gland_order = ["Adrenal", "Gonads", "Pancreas", "Pineal", "Thymus", "Thyroid"]

df = df.loc[chakra_order, gland_order]

# ------------------------------------------------------------
# Create heatmap
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 6))

im = ax.imshow(df.values, aspect="auto")

# Axis labels
ax.set_xticks(range(len(df.columns)))
ax.set_xticklabels(df.columns, rotation=45, ha="right")
ax.set_yticks(range(len(df.index)))
ax.set_yticklabels(df.index)

# Title
ax.set_title("Updated Chakra–Gland Cosine Similarity Heatmap")

# Add cell values
for i in range(df.shape[0]):
    for j in range(df.shape[1]):
        ax.text(
            j,
            i,
            f"{df.iloc[i, j]:.4f}",
            ha="center",
            va="center",
            fontsize=8
        )

# Colorbar
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Cosine Similarity")

plt.tight_layout()

# ------------------------------------------------------------
# Save figure
# ------------------------------------------------------------
plt.savefig(output_file, dpi=300, bbox_inches="tight")
plt.show()

print(f"\nHeatmap saved to: {output_file}")
