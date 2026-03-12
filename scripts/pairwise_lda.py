from pdfminer.high_level import extract_text
import os
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

project_root = ".."

# Folders
chakra_split_root = os.path.join(project_root, "chakra_split")
endocrine_root = os.path.join(project_root, "endocrine_papers")
results_root = os.path.join(project_root, "results", "pairwise_lda")

os.makedirs(results_root, exist_ok=True)

# ---- IMPORTANT ----
# Update these filenames to match the actual PDFs in your endocrine_papers folder
pairings = [
    {
        "name": "root_vs_adrenal",
        "chakra_folder": os.path.join(chakra_split_root, "root"),
        "endocrine_files": [
            "adrenal gland 1.pdf",
        ],
        "n_topics": 3,
    },
    {
        "name": "sacral_vs_gonads",
        "chakra_folder": os.path.join(chakra_split_root, "sacral"),
        "endocrine_files": [
            "(Gonad) estrogen2. pdf.pdf",
            "testosterone new.pdf",
        ],
        "n_topics": 3,
    },
    {
        "name": "solar_plexus_vs_pancreas",
        "chakra_folder": os.path.join(chakra_split_root, "solar_plexus"),
        "endocrine_files": [
            "(Pancreas) insulin.pdf",
        ],
        "n_topics": 3,
    },
    {
        "name": "heart_vs_thymus",
        "chakra_folder": os.path.join(chakra_split_root, "heart"),
        "endocrine_files": [
            "thymus 1.pdf",
        ],
        "n_topics": 3,
    },
    {
        "name": "throat_vs_thyroid",
        "chakra_folder": os.path.join(chakra_split_root, "throat"),
        "endocrine_files": [
            "thyroid 1.pdf",
            "thyroid2.pdf",
        ],
        "n_topics": 3,
    },
    {
        "name": "third_eye_vs_pineal",
        "chakra_folder": os.path.join(chakra_split_root, "third_eye"),
        "endocrine_files": [
            "pineal.pdf",
        ],
        "n_topics": 3,
    },
    
]

def clean_text(text):
    text = text.lower()
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\[[^\]]*\]", " ", text)
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    words = text.split()
    words = [w for w in words if w not in stop_words and len(w) > 2]

    return " ".join(words)

def load_txt_documents(folder, label_prefix):
    docs = []
    names = []

    for file in os.listdir(folder):
        if file.endswith(".txt"):
            path = os.path.join(folder, file)
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            text = clean_text(text)
            if len(text.strip()) > 0:
                docs.append(text)
                names.append(f"{label_prefix}:{file}")

    return docs, names

def load_pdf_documents(folder, filenames, label_prefix):
    docs = []
    names = []

    for file in filenames:
        path = os.path.join(folder, file)

        if not os.path.exists(path):
            print(f"WARNING: File not found -> {path}")
            continue

        text = extract_text(path)
        text = clean_text(text)

        if len(text.strip()) > 0:
            docs.append(text)
            names.append(f"{label_prefix}:{file}")

    return docs, names

summary_lines = []

for pairing in pairings:
    print(f"\nRunning {pairing['name']} ...")

    chakra_docs, chakra_names = load_txt_documents(pairing["chakra_folder"], "chakra")
    endocrine_docs, endocrine_names = load_pdf_documents(
        endocrine_root,
        pairing["endocrine_files"],
        "endocrine"
    )

    documents = chakra_docs + endocrine_docs
    document_names = chakra_names + endocrine_names

    if len(documents) < 3:
        print(f"Skipping {pairing['name']} - not enough documents.")
        summary_lines.append(f"{pairing['name']}: skipped (not enough documents)")
        continue

    vectorizer = CountVectorizer(max_df=0.9, min_df=1)
    X = vectorizer.fit_transform(documents)

    lda = LatentDirichletAllocation(
        n_components=pairing["n_topics"],
        random_state=42
    )
    lda.fit(X)

    words = vectorizer.get_feature_names_out()

    output_file = os.path.join(results_root, f"{pairing['name']}_topics.txt")
    docs_file = os.path.join(results_root, f"{pairing['name']}_documents.txt")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"Pairing: {pairing['name']}\n")
        f.write("=" * 60 + "\n\n")

        for i, topic in enumerate(lda.components_):
            topic_words = [words[j] for j in topic.argsort()[-12:]][::-1]
            line = f"Topic {i+1}: " + ", ".join(topic_words) + "\n"
            print(line.strip())
            f.write(line)

    with open(docs_file, "w", encoding="utf-8") as f:
        for name in document_names:
            f.write(name + "\n")

    summary_lines.append(
        f"{pairing['name']}: {len(chakra_docs)} chakra docs + {len(endocrine_docs)} endocrine docs"
    )

summary_file = os.path.join(results_root, "summary.txt")
with open(summary_file, "w", encoding="utf-8") as f:
    for line in summary_lines:
        f.write(line + "\n")

print("\nDone.")
print(f"Results saved in: {results_root}")
