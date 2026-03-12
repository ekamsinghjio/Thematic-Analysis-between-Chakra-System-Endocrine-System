from pdfminer.high_level import extract_text
import os
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Download stopwords once
nltk.download("stopwords")

stop_words = set(stopwords.words("english"))

# Project folders
project_root = ".."
chakra_folder = os.path.join(project_root, "chakra_papers")
endocrine_folder = os.path.join(project_root, "endocrine_papers")
results_folder = os.path.join(project_root, "results")

os.makedirs(results_folder, exist_ok=True)

documents = []
doc_names = []

def clean_text(text):
    text = text.lower()
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\[[^\]]*\]", " ", text)
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)

    words = text.split()
    words = [w for w in words if w not in stop_words and len(w) > 2]

    return " ".join(words)

def load_pdfs(folder):
    for file in os.listdir(folder):
        if file.endswith(".pdf"):
            path = os.path.join(folder, file)
            print(f"Reading: {file}")
            text = extract_text(path)
            text = clean_text(text)

            if len(text.strip()) > 0:
                documents.append(text)
                doc_names.append(file)
            else:
                print(f"Warning: no text extracted from {file}")

# Load both corpora
load_pdfs(chakra_folder)
load_pdfs(endocrine_folder)

print(f"\nTotal documents loaded: {len(documents)}")

if len(documents) < 2:
    raise ValueError("Not enough documents were loaded. Check that your PDFs are in the correct folders.")

# Vectorize text
vectorizer = CountVectorizer(max_df=0.9, min_df=2)
X = vectorizer.fit_transform(documents)

# Fit LDA model
lda = LatentDirichletAllocation(n_components=6, random_state=42)
lda.fit(X)

words = vectorizer.get_feature_names_out()

# Save topics
topics_file = os.path.join(results_folder, "lda_topics.txt")
with open(topics_file, "w", encoding="utf-8") as f:
    for i, topic in enumerate(lda.components_):
        topic_words = [words[j] for j in topic.argsort()[-10:]]
        line = f"Topic {i+1}: " + ", ".join(topic_words) + "\n"
        print(line)
        f.write(line)

# Save document names
docs_file = os.path.join(results_folder, "documents_used.txt")
with open(docs_file, "w", encoding="utf-8") as f:
    for name in doc_names:
        f.write(name + "\n")

print("\nAnalysis complete.")
print("Results saved in:")
print(f" - {topics_file}")
print(f" - {docs_file}")
