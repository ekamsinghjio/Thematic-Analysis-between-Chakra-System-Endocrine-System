import os
from pdfminer.high_level import extract_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

project_root = ".."

chakra_root = os.path.join(project_root, "chakra_split")
endocrine_root = os.path.join(project_root, "endocrine_papers")
results_root = os.path.join(project_root, "results")

os.makedirs(results_root, exist_ok=True)

def load_chakra_text(folder):
    text = ""
    for file in os.listdir(folder):
        if file.endswith(".txt"):
            with open(os.path.join(folder, file), "r", encoding="utf-8") as f:
                text += f.read() + " "
    return text

def load_endocrine_text(files):
    text = ""
    for file in files:
        path = os.path.join(endocrine_root, file)
        text += extract_text(path) + " "
    return text

# ---- UPDATE THESE FILENAMES TO MATCH YOUR ACTUAL FILES ----
glands = {
    "Adrenal": [
        "adrenal gland 1.pdf"
    ],
    "Gonads": [
        "(Gonad) estrogen2. pdf.pdf",
        "testosterone new.pdf"
    ],
    "Pancreas": [
        "(Pancreas) insulin.pdf"
    ],
    "Thymus": [
        "thymus 1.pdf"
    ],
    "Thyroid": [
        "thyroid 1.pdf",
        "thyroid2.pdf"
    ],
    "Pineal": [
        "pineal.pdf"
    ]
}

chakras = {
    "Root": "root",
    "Sacral": "sacral",
    "Solar Plexus": "solar_plexus",
    "Heart": "heart",
    "Throat": "throat",
    "Third Eye": "third_eye"
}

rows = []

for chakra_label, chakra_folder in chakras.items():
    chakra_text = load_chakra_text(os.path.join(chakra_root, chakra_folder))

    for gland_label, gland_files in glands.items():
        endocrine_text = load_endocrine_text(gland_files)

        documents = [chakra_text, endocrine_text]

        vectorizer = TfidfVectorizer(stop_words="english")
        X = vectorizer.fit_transform(documents)

        similarity = cosine_similarity(X[0], X[1])[0][0]

        rows.append({
            "Chakra": chakra_label,
            "Gland": gland_label,
            "CosineSimilarity": round(similarity, 4)
        })

df = pd.DataFrame(rows)

# Save long-form table
long_path = os.path.join(results_root, "all_chakra_all_gland_cosine_long.csv")
df.to_csv(long_path, index=False)

# Save matrix table
matrix_df = df.pivot(index="Chakra", columns="Gland", values="CosineSimilarity")
matrix_path = os.path.join(results_root, "all_chakra_all_gland_cosine_matrix.csv")
matrix_df.to_csv(matrix_path)

print("\nLong-form results:\n")
print(df)

print("\nMatrix results:\n")
print(matrix_df)

print(f"\nSaved to:\n{long_path}\n{matrix_path}")
