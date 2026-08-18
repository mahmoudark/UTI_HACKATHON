import chromadb
from sentence_transformers import SentenceTransformer


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding model loaded")


# ============================================================
# LOAD CHROMA
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
# TEST QUESTIONS
# ============================================================

test_cases = [
    {
        "question": "What is a lower urinary tract infection?",
        "expected": "recommendation_1.1.1"
    },
    {
        "question": "What advice should be given to all people with lower UTI about managing symptoms?",
        "expected": "recommendation_1.1.2"
    },
    {
        "question": "What should be done if symptoms do not start to improve within 48 hours or worsen at any time for women with lower UTI who are not pregnant?",
        "expected": "recommendation_1.1.3"
    },
    {
        "question": "What should be done when microbiological results are available after a urine sample was sent for culture and susceptibility testing?",
        "expected": "recommendation_1.1.4"
    },
    {
        "question": "What should be done for pregnant women and men with lower UTI?",
        "expected": "recommendation_1.1.5"
    },
    {
        "question": "What should people with lower UTI be advised to use for pain?",
        "expected": "recommendation_1.3.1"
    },
    {
        "question": "What should be considered when prescribing antibiotics for lower UTI?",
        "expected": "recommendation_1.4.1"
    },
    {
        "question": "What antibiotics are recommended for non-pregnant women aged 16 years and over?",
        "expected": "TABLE_1"
    },
    {
        "question": "What antibiotics are recommended for pregnant women aged 12 years and over?",
        "expected": "TABLE_2"
    },
    {
        "question": "What antibiotics are recommended for men aged 16 years and over?",
        "expected": "TABLE_3"
    },
    {
        "question": "What antibiotics are recommended for children and young people under 16 years?",
        "expected": "TABLE_4"
    }
]


# ============================================================
# RETRIEVAL FUNCTION
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

    ids = results["ids"][0]
    distances = results["distances"][0]

    return ids, distances


# ============================================================
# EVALUATE TOP-K
# ============================================================

def evaluate_k(k):

    correct = 0
    reciprocal_sum = 0

    for case in test_cases:

        ids, distances = retrieve(
            case["question"],
            k
        )

        expected = case["expected"]

        if expected in ids:

            correct += 1

            rank = ids.index(expected) + 1

            reciprocal_sum += 1 / rank

    recall = correct / len(test_cases)

    mrr = reciprocal_sum / len(test_cases)

    return recall, mrr


# ============================================================
# MAIN
# ============================================================

print()
print("=" * 70)
print("TOP-K RETRIEVAL EVALUATION")
print("=" * 70)

k_values = [1, 2, 3, 4, 5, 7, 10]

best_k = None
best_recall = -1
best_mrr = -1

print()
print("Testing different K values...")
print()

for k in k_values:

    recall, mrr = evaluate_k(k)

    print(
        f"Top-K = {k:2d} | "
        f"Recall@{k} = {recall:.4f} | "
        f"MRR = {mrr:.4f}"
    )

    if recall > best_recall or (
        recall == best_recall and mrr > best_mrr
    ):
        best_k = k
        best_recall = recall
        best_mrr = mrr


# ============================================================
# FINAL RESULT
# ============================================================

print()
print("=" * 70)
print("BEST TOP-K")
print("=" * 70)

print("Best K:", best_k)
print("Recall:", round(best_recall, 4))
print("MRR:", round(best_mrr, 4))

print()
print("Recommendation:")
print(
    f"Use Top-K = {best_k} based on the evaluation results."
)

print()
print("Evaluation completed.")