# scripts/doc_labels.py

CHAKRA_DOCS = [
    "An analytical overview of the 7 chakras.pdf",
    "Theoretical underpinngs.pdf",
    "An overview of shatchakra.pdf",
    "The geometry of emotions.pdf",
    "The Chakra System as a Framework for Holistic Educational Develop.pdf",
    "MAIN The Foundations of the Eastern Chakra System.pdf",
    "CONCEPTUAL STUDY ON SHAD CHAKRAS.pdf"
]

def classify_system(document_name: str) -> str:
    return "Chakra" if document_name in CHAKRA_DOCS else "Endocrine"
