import os
from pdfminer.high_level import extract_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

project_root = ".."

chakra_root = os.path.join(project_root, "chakra_split")
endocrine_root = os.path.join(project_root, "endocrine_papers")

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


pairs = [
    ("root", "root", ["adrenal gland 1.pdf"]),
    ("sacral", "sacral", ["(Gonad) estrogen2. pdf.pdf", "testosterone new.pdf"]),
    ("solar_plexus", "solar_plexus", ["(Pancreas) insulin.pdf"]),
    ("heart", "heart", ["thymus 1.pdf"]),
    ("throat", "throat", ["thyroid 1.pdf", "thyroid2.pdf"]),
    ("third_eye_pineal", "third_eye", ["pineal.pdf"]),
]

print("\nCosine Similarity Results\n")

for label, chakra_folder, endocrine_files in pairs:

    chakra_text = load_chakra_text(os.path.join(chakra_root, chakra_folder))
    endocrine_text = load_endocrine_text(endocrine_files)

    documents = [chakra_text, endocrine_text]

    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(documents)

    similarity = cosine_similarity(X[0], X[1])[0][0]

    print(f"{label} similarity: {similarity:.4f}")
