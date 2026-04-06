import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from doc_labels import classify_system

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
results_folder = "../results"
input_file = os.path.join(results_folder, "document_topic_distribution.csv")

# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------
df = pd.read_csv(input_file)

# Assign system labels consistently
df["System"] = df["Document"].apply(classify_system)

# ------------------------------------------------------------
# Topic columns
# ------------------------------------------------------------
topic_cols = [col for col in df.columns if col.startswith("Topic_")]

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------
def normalize_vector(vec):
    vec = np.array(vec, dtype=float)
    total = vec.sum()
    if total == 0:
        return vec
    return vec / total

def overlap_score(vec1, vec2):
    """
    Shared probability mass across two topic distributions.
    Range: 0 to 1
    - 1 means complete overlap
    - 0 means no overlap
    """
    v1 = normalize_vector(vec1)
    v2 = normalize_vector(vec2)
    return np.sum(np.minimum(v1, v2))

def js_distance(vec1, vec2):
    """
    Jensen-Shannon distance between two probability distributions.
    Range: 0 to 1
    - 0 means identical
    - larger means more different
    """
    v1 = normalize_vector(vec1)
    v2 = normalize_vector(vec2)
    return jensenshannon(v1, v2)

# ------------------------------------------------------------
# A. SYSTEM-LEVEL COMPARISON
# ------------------------------------------------------------
chakra_df = df[df["System"] == "Chakra"]
endocrine_df = df[df["System"] == "Endocrine"]

chakra_avg = chakra_df[topic_cols].mean().values
endocrine_avg = endocrine_df[topic_cols].mean().values

system_js = js_distance(chakra_avg, endocrine_avg)
system_overlap = overlap_score(chakra_avg, endocrine_avg)

print("\n=== A. SYSTEM-LEVEL THEMATIC SIMILARITY ===")
print("Chakra average topic distribution:")
print(pd.Series(chakra_avg, index=topic_cols))
print("\nEndocrine average topic distribution:")
print(pd.Series(endocrine_avg, index=topic_cols))
print(f"\nJensen-Shannon distance (Chakra vs Endocrine): {system_js:.4f}")
print(f"Overlap score (Chakra vs Endocrine): {system_overlap:.4f}")

system_summary = pd.DataFrame({
    "Comparison": ["Chakra_vs_Endocrine"],
    "JS_Distance": [system_js],
    "Overlap_Score": [system_overlap]
})

# Save system-level summary
system_summary_file = os.path.join(results_folder, "system_topic_similarity.csv")
system_summary.to_csv(system_summary_file, index=False)

# ------------------------------------------------------------
# B. PER-GLAND COMPARISON
# Compare each endocrine paper to the overall chakra average
# ------------------------------------------------------------
print("\n=== B. PER-GLAND THEMATIC SIMILARITY ===")

gland_rows = []

for _, row in endocrine_df.iterrows():
    gland_name = row["Document"]
    gland_vec = row[topic_cols].values

    gland_js = js_distance(chakra_avg, gland_vec)
    gland_overlap = overlap_score(chakra_avg, gland_vec)

    gland_rows.append({
        "Gland_Document": gland_name,
        "JS_Distance_to_ChakraAvg": gland_js,
        "Overlap_Score_with_ChakraAvg": gland_overlap
    })

gland_summary = pd.DataFrame(gland_rows)

# Sort for easier interpretation:
# lower JS = more similar
gland_summary = gland_summary.sort_values(by="JS_Distance_to_ChakraAvg", ascending=True)

print(gland_summary)

gland_summary_file = os.path.join(results_folder, "gland_topic_similarity.csv")
gland_summary.to_csv(gland_summary_file, index=False)

print("\nSaved files:")
print(f" - {system_summary_file}")
print(f" - {gland_summary_file}")
