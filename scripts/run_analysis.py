from pdfminer.high_level import extract_text
import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Import the shared preprocessing pipeline
from preprocessing import preprocess_text

# Project folders
project_root = ".."
chakra_folder = os.path.join(project_root, "chakra_papers")
endocrine_folder = os.path.join(project_root, "endocrine_papers")
results_folder = os.path.join(project_root, "results")

os.makedirs(results_folder, exist_ok=True)

documents = []
doc_names = []

def load_pdfs(folder):
    for file in os.listdir(folder):
        if file.endswith(".pdf"):
            path = os.path.join(folder, file)
            print(f"Reading: {file}")

            raw_text = extract_text(path)
            cleaned_text = preprocess_text(raw_text)

            if len(cleaned_text.strip()) > 0:
                documents.append(cleaned_text)
                doc_names.append(file)
            else:
                print(f"Warning: no usable psychophysiology text remained after preprocessing for {file}")

# Load both corpora
load_pdfs(chakra_folder)
load_pdfs(endocrine_folder)

print(f"\nTotal documents loaded: {len(documents)}")

if len(documents) < 2:
    raise ValueError("Not enough documents were loaded. Check that your PDFs are in the correct folders.")

# Vectorize text for topic modeling
vectorizer = CountVectorizer(
    max_df=0.85,
    min_df=2,
    ngram_range=(1, 2)
)
X = vectorizer.fit_transform(documents)

# Fit LDA model
n_topics = 6
lda = LatentDirichletAllocation(
    n_components=n_topics,
    random_state=42
)
lda.fit(X)

words = vectorizer.get_feature_names_out()

# ------------------------------------------------------------
# 1. Save topics
# ------------------------------------------------------------
topics_file = os.path.join(results_folder, "lda_topics.txt")
with open(topics_file, "w", encoding="utf-8") as f:
    for i, topic in enumerate(lda.components_):
        topic_words = [words[j] for j in topic.argsort()[-10:]]
        line = f"Topic {i+1}: " + ", ".join(topic_words) + "\n"
        print(line)
        f.write(line)

# ------------------------------------------------------------
# 2. Save document names
# ------------------------------------------------------------
docs_file = os.path.join(results_folder, "documents_used.txt")
with open(docs_file, "w", encoding="utf-8") as f:
    for name in doc_names:
        f.write(name + "\n")

# ------------------------------------------------------------
# 3. Compute and save document-topic distributions
# ------------------------------------------------------------
doc_topic_matrix = lda.transform(X)

topic_columns = [f"Topic_{i+1}" for i in range(n_topics)]

doc_topic_df = pd.DataFrame(doc_topic_matrix, columns=topic_columns)
doc_topic_df.insert(0, "Document", doc_names)

doc_topic_file = os.path.join(results_folder, "document_topic_distribution.csv")
doc_topic_df.to_csv(doc_topic_file, index=False)

print("\nDocument-topic distribution preview:")
print(doc_topic_df.head())

print("\nAnalysis complete.")
print("Results saved in:")
print(f" - {topics_file}")
print(f" - {docs_file}")
print(f" - {doc_topic_file}")

