# scripts/preprocessing.py

import re
from typing import List, Iterable

import nltk
from nltk.corpus import stopwords

# Download resources once
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

ENGLISH_STOPWORDS = set(stopwords.words("english"))

# -------------------------------------------------------------------
# 1. Broad seed terms for psychophysiology
#    This is a broad starting point and can be refined later.
# -------------------------------------------------------------------
PSYCHOPHYSIOLOGY_TERMS = {
    # mental health / affect
    "mental", "psychological", "psychiatric", "mood", "emotion", "emotional",
    "affect", "affective", "anxiety", "depression", "depressive", "distress",
    "stress", "stressor", "trauma", "wellbeing", "well-being",

    # cognition / behavior
    "behavior", "behaviour", "behavioral", "behavioural", "cognition",
    "cognitive", "memory", "attention", "learning", "motivation",
    "reward", "executive", "social", "bonding", "attachment",

    # regulation / physiology tied to psychophysiology
    "regulation", "reactivity", "arousal", "circadian", "sleep",
    "fatigue", "homeostasis", "neuroendocrine", "autonomic",
    "hpa", "cortisol", "melatonin", "insulin", "thyroid", "testosterone",
    "estradiol", "immune", "inflammation",

    # symptom / outcome style words often used in reviews
    "symptom", "symptoms", "impairment", "dysregulation", "quality",
    "adaptation", "coping", "resilience", "fatigue"
}

# -------------------------------------------------------------------
# 2. Words that are very common in papers but not useful for topics
# -------------------------------------------------------------------
DOMAIN_STOPWORDS = {
    "study", "studies", "review", "reviews", "article", "articles",
    "author", "authors", "paper", "papers", "result", "results", "method", "methods",
    "introduction", "discussion", "conclusion", "limitations",
    "table", "tables", "figure", "fig", "figures",
    "copyright", "license", "published", "journal",
    "elsevier", "springer", "pmid", "doi",

    # vague academic filler
    "may", "might", "could", "would", "also", "however", "therefore",
    "based", "using", "used", "use", "show", "shown", "found",

    # clinical / generic words that often dominate without helping your themes
    "patient", "patients", "disease", "diseases", "disorder", "disorders",
    "function", "functions", "level", "levels", "acute", "chronic",
    "placebo", "faculty", "node", "physical", "self", "outcomes", "controlled",

    # likely noisy for topic modeling if they dominate
    "chakra", "chakras", "gland", "glands", "energy"
}

ALL_STOPWORDS = ENGLISH_STOPWORDS.union(DOMAIN_STOPWORDS)

# -------------------------------------------------------------------
# 3. Helper: remove obvious PDF / citation / formatting junk
# -------------------------------------------------------------------
def remove_pdf_junk(text: str) -> str:
    if not text:
        return ""

    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", " ", text)

    # Remove DOI-like strings
    text = re.sub(r"\bdoi\s*:\s*\S+\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", " ", text, flags=re.IGNORECASE)

    # Remove bracket citations like [1], [12], [3, 4]
    text = re.sub(r"\[[0-9,\-\s]+\]", " ", text)

    # Remove author-year style artifacts
    text = re.sub(r"\bet al\.?\b", " ", text, flags=re.IGNORECASE)

    # Remove common section headers
    text = re.sub(
        r"\b(abstract|introduction|methods|materials and methods|results|discussion|conclusion|references)\b",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    # Remove figure/table labels
    text = re.sub(r"\b(fig(?:ure)?|table)\s*\d+[a-z]?\b", " ", text, flags=re.IGNORECASE)

    # Remove long runs of numbers
    text = re.sub(r"\b\d+\b", " ", text)

    # Normalize whitespace and line breaks
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)

    return text.strip()

# -------------------------------------------------------------------
# 4. Helper: split into sentences
# -------------------------------------------------------------------
def split_into_sentences(text: str) -> List[str]:
    return nltk.sent_tokenize(text)

# -------------------------------------------------------------------
# 5. Decide whether a sentence looks psychophysiology-relevant
# -------------------------------------------------------------------
def sentence_is_relevant(sentence: str, terms: Iterable[str] = PSYCHOPHYSIOLOGY_TERMS) -> bool:
    sentence_lower = sentence.lower()
    return any(term in sentence_lower for term in terms)

# -------------------------------------------------------------------
# 6. Keep matching sentences + neighboring context
# -------------------------------------------------------------------
def filter_relevant_sentences(
    text: str,
    terms: Iterable[str] = PSYCHOPHYSIOLOGY_TERMS,
    keep_neighbor_sentences: bool = True
) -> str:
    sentences = split_into_sentences(text)
    if not sentences:
        return ""

    keep_indices = set()

    for i, sentence in enumerate(sentences):
        if sentence_is_relevant(sentence, terms):
            keep_indices.add(i)
            if keep_neighbor_sentences:
                if i - 1 >= 0:
                    keep_indices.add(i - 1)
                if i + 1 < len(sentences):
                    keep_indices.add(i + 1)

    filtered = [sentences[i] for i in sorted(keep_indices)]
    return " ".join(filtered)

# -------------------------------------------------------------------
# 7. Final token-level cleanup for modeling
# -------------------------------------------------------------------
def normalize_for_modeling(text: str) -> str:
    text = text.lower()

    # Remove anything not letters or spaces
    text = re.sub(r"[^a-z\s]", " ", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    words = text.split()

    # Remove stopwords and very short tokens
    words = [w for w in words if w not in ALL_STOPWORDS and len(w) > 2]

    # Remove repeated junk-like one-offs if needed later; for now keep simple
    return " ".join(words)

# -------------------------------------------------------------------
# 8. Full preprocessing pipeline
# -------------------------------------------------------------------
def preprocess_text(
    text: str,
    psych_terms: Iterable[str] = PSYCHOPHYSIOLOGY_TERMS,
    keep_neighbor_sentences: bool = True,
) -> str:
    text = remove_pdf_junk(text)
    text = filter_relevant_sentences(
        text,
        terms=psych_terms,
        keep_neighbor_sentences=keep_neighbor_sentences
    )
    text = normalize_for_modeling(text)
    return text
