import pandas as pd
import os
from doc_labels import classify_system

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
results_folder = "../results"
input_file = os.path.join(results_folder, "document_topic_distribution.csv")
output_file = os.path.join(results_folder, "top_topic_summary.csv")

# ------------------------------------------------------------
# Load document-topic distribution
# ------------------------------------------------------------
df = pd.read_csv(input_file)

# Topic columns only
topic_cols = [col for col in df.columns if col.startswith("Topic_")]

# ------------------------------------------------------------
# For each document, find top 2 topics
# ------------------------------------------------------------
summary_rows = []

for _, row in df.iterrows():
    document_name = row["Document"]

    # Get topic scores as a Series
    topic_scores = row[topic_cols].astype(float)

    # Sort descending
    sorted_topics = topic_scores.sort_values(ascending=False)

    top_topic = sorted_topics.index[0]
    top_score = sorted_topics.iloc[0]

    second_topic = sorted_topics.index[1]
    second_score = sorted_topics.iloc[1]

    summary_rows.append({
        "Document": document_name,
        "System": classify_system(document_name),
        "Top_Topic": top_topic,
        "Top_Score": top_score,
        "Second_Topic": second_topic,
        "Second_Score": second_score
    })

summary_df = pd.DataFrame(summary_rows)

# Reorder columns for readability
summary_df = summary_df[
    ["Document", "System", "Top_Topic", "Top_Score", "Second_Topic", "Second_Score"]
]

# ------------------------------------------------------------
# Print to terminal
# ------------------------------------------------------------
print("\n=== TOP TOPIC SUMMARY ===")
print(summary_df)

# ------------------------------------------------------------
# Save to CSV
# ------------------------------------------------------------
summary_df.to_csv(output_file, index=False)

print(f"\nSaved to: {output_file}")
