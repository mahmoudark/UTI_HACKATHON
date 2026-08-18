import chromadb
from sentence_transformers import SentenceTransformer


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding model loaded")


# ============================================================
# LOAD DATABASE
# ============================================================

client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_collection(
    name="uti_guideline"
)

print("Vector database loaded")
print("Documents:", collection.count())


# ============================================================
# TEST DATA
# ============================================================

test_cases = [
    ("What is a lower urinary tract infection?",
     "recommendation_1.1.1"),

    ("What advice should be given to all people with lower UTI about managing symptoms?",
     "recommendation_1.1.2"),

    ("What should be done if symptoms do not start to improve within 48 hours or worsen at any time for women with lower UTI who are not pregnant?",
     "recommendation_1.1.3"),

    ("What should be done when microbiological results are available after a urine sample was sent for culture and susceptibility testing?",
     "recommendation_1.1.4"),

    ("What should be done for pregnant women and men with lower UTI?",
     "recommendation_1.1.5"),

    ("What should people with lower UTI be advised to use for pain?",
     "recommendation_1.3.1"),

    ("What should be considered when prescribing antibiotics for lower UTI?",
     "recommendation_1.4.1"),

    ("What antibiotics are recommended for non-pregnant women aged 16 years and over?",
     "TABLE_1"),

    ("What antibiotics are recommended for pregnant women aged 12 years and over?",
     "TABLE_2"),

    ("What antibiotics are recommended for men aged 16 years and over?",
     "TABLE_3"),

    ("What antibiotics are recommended for children and young people under 16 years?",
     "TABLE_4"),
]


# ============================================================
# RETRIEVE
# ============================================================

def retrieve(question, k):

    embedding = model.encode(
        [question],
        normalize_embeddings=True
    )[0].tolist()

    results = collection.query(
        query_embeddings=[embedding],
        n_results=k
    )

    return results["ids"][0]


# ============================================================
# EVALUATION
# ============================================================

def evaluate(k):

    recall_hits = 0
    precision_sum = 0
    reciprocal_sum = 0

    for question, expected in test_cases:

        retrieved = retrieve(question, k)

        # Recall
        if expected in retrieved:
            recall_hits += 1

            rank = retrieved.index(expected) + 1
            reciprocal_sum += 1 / rank

        # Precision
        relevant = 0

        for doc_id in retrieved:
            if doc_id == expected:
                relevant += 1

        precision_sum += relevant / k

    total = len(test_cases)

    recall = recall_hits / total
    precision = precision_sum / total
    mrr = reciprocal_sum / total

    return recall, precision, mrr


# ============================================================
# RUN EVALUATION
# ============================================================

print()
print("=" * 75)
print("FINAL RETRIEVAL EVALUATION")
print("=" * 75)

print()
print("Recall + Precision + MRR")
print()

k_values = [1, 2, 3, 4, 5, 7, 10]

best_k = None
best_score = -1

for k in k_values:

    recall, precision, mrr = evaluate(k)

    # Balanced score
    balanced_score = (
        recall + precision + mrr
    ) / 3

    print(
        f"K={k:2d} | "
        f"Recall={recall:.4f} | "
        f"Precision={precision:.4f} | "
        f"MRR={mrr:.4f} | "
        f"Balanced={balanced_score:.4f}"
    )

    if balanced_score > best_score:
        best_score = balanced_score
        best_k = k


# ============================================================
# FINAL RECOMMENDATION
# ============================================================

print()
print("=" * 75)
print("FINAL RECOMMENDATION")
print("=" * 75)

print(
    f"Recommended Top-K: {best_k}"
)

print(
    f"Balanced Score: {best_score:.4f}"
)

print()
print("Evaluation completed.")