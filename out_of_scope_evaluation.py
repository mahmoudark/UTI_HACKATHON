import re
import chromadb
from sentence_transformers import SentenceTransformer


# ============================================================
# SETTINGS
# ============================================================

MODEL_NAME = "all-mpnet-base-v2"

DATABASE_PATH = "./chroma_db_mpnet"
COLLECTION_NAME = "uti_guideline_mpnet"

LEXICAL_WEIGHT = 0.10


# ============================================================
# OUT-OF-SCOPE QUESTIONS
# ============================================================

test_questions = [

    "What is diabetes mellitus?",

    "What are the symptoms of pneumonia?",

    "How is hypertension treated?",

    "What are the symptoms of asthma?",

    "What is appendicitis?",

    "What antibiotics are used for pneumonia?",

    "How is diabetes diagnosed?",

    "What is the treatment for migraine?",

    "What causes high blood pressure?",

    "What are the symptoms of a heart attack?",

    "How is influenza treated?",

    "What is chronic kidney disease?",

    "What are the symptoms of meningitis?",

    "How is asthma diagnosed?",

    "What is the treatment for appendicitis?"
]


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading MPNet...")

model = SentenceTransformer(
    MODEL_NAME
)

print("MPNet loaded")


# ============================================================
# LOAD DATABASE
# ============================================================

client = chromadb.PersistentClient(
    path=DATABASE_PATH
)

collection = client.get_collection(
    name=COLLECTION_NAME
)

print(
    "Documents:",
    collection.count()
)


# ============================================================
# TOKENIZATION
# ============================================================

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "for", "with",
    "what", "which", "how", "should", "be", "is", "are", "was",
    "were", "do", "does", "did", "when", "where", "who", "that",
    "this", "these", "those", "in", "on", "at", "from", "by",
    "about", "all", "people", "person", "patients", "patient"
}


def tokenize(text):

    return {
        word
        for word in re.findall(
            r"[a-zA-Z0-9]+",
            text.lower()
        )
        if len(word) >= 3
        and word not in STOPWORDS
    }


def lexical_overlap(query, document):

    q = tokenize(query)
    d = tokenize(document)

    if not q or not d:
        return 0.0

    overlap = len(q.intersection(d))

    precision = overlap / len(q)
    recall = overlap / len(d)

    if precision + recall == 0:
        return 0.0

    return (
        2 * precision * recall
        / (precision + recall)
    )


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve(question):

    embedding = model.encode(
        [question],
        normalize_embeddings=True
    )[0].tolist()

    results = collection.query(
        query_embeddings=[embedding],
        n_results=collection.count()
    )

    candidates = []

    for document, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):

        similarity = 1 - distance

        lexical = lexical_overlap(
            question,
            document
        )

        hybrid = (
            similarity
            + LEXICAL_WEIGHT * lexical
        )

        candidates.append({
            "source_id": metadata.get("source_id"),
            "title": metadata.get("title"),
            "similarity": similarity,
            "lexical": lexical,
            "hybrid": hybrid
        })

    candidates.sort(
        key=lambda x: x["hybrid"],
        reverse=True
    )

    return candidates


# ============================================================
# TEST
# ============================================================

print()
print("=" * 75)
print("OUT-OF-SCOPE EVALUATION")
print("=" * 75)

print()

for index, question in enumerate(
    test_questions,
    start=1
):

    results = retrieve(question)

    best = results[0]

    print()
    print(
        f"[{index}/{len(test_questions)}]"
    )

    print(
        "Question:",
        question
    )

    print(
        "Top Source:",
        best["source_id"]
    )

    print(
        "Similarity:",
        f"{best['similarity']:.4f}"
    )

    print(
        "Lexical:",
        f"{best['lexical']:.4f}"
    )

    print(
        "Hybrid:",
        f"{best['hybrid']:.4f}"
    )

    print(
        "Top 3:",
        " | ".join(
            x["source_id"]
            for x in results[:3]
        )
    )


# ============================================================
# THRESHOLD SUMMARY
# ============================================================

print()
print("=" * 75)
print("THRESHOLD SUMMARY")
print("=" * 75)

for threshold in [
    0.20,
    0.22,
    0.24,
    0.25,
    0.28,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50
]:

    rejected = 0

    for question in test_questions:

        results = retrieve(question)

        best = results[0]

        if best["similarity"] < threshold:
            rejected += 1

    refusal_rate = (
        rejected /
        len(test_questions)
    )

    print(
        f"Threshold {threshold:.2f} | "
        f"Refusal rate: {refusal_rate:.4f}"
    )


print()
print("=" * 75)
print("EVALUATION COMPLETED")
print("=" * 75)