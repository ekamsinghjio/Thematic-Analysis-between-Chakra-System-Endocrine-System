import pandas as pd
import os
from doc_labels import classify_system

# Load data
file_path = "../results/document_topic_distribution.csv"
df = pd.read_csv(file_path)

# Assign system labels consistently
df["System"] = df["Document"].apply(classify_system)

# Split by system
chakra_df = df[df["System"] == "Chakra"]
endocrine_df = df[df["System"] == "Endocrine"]

# Drop non-topic columns for averaging
topic_cols = [col for col in df.columns if col.startswith("Topic_")]

chakra_topics = chakra_df[topic_cols]
endocrine_topics = endocrine_df[topic_cols]

# Compute averages
chakra_avg = chakra_topics.mean()
endocrine_avg = endocrine_topics.mean()

# Combine into one clean table
summary = pd.DataFrame({
    "Chakra": chakra_avg,
    "Endocrine": endocrine_avg
}).T

print("\n=== CLEAN SUMMARY TABLE ===")
print(summary)

# Save to CSV
summary.to_csv("../results/topic_summary.csv")
print("\nSaved to results/topic_summary.csv")
