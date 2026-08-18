import chromadb
from sentence_transformers import SentenceTransformer


# ============================================================
# 1) LOAD MODEL
# ============================================================

print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding model loaded")


# ============================================================
# 2) LOAD CHROMA
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
# 3) TEST QUESTIONS + GOLD DOCUMENT
# ============================================================

TEST_CASES = [

    {
        "question": "What is a lower urinary tract infection?",
        "gold": "recommendation_1.1.1"
    },

    {
        "question": "What advice should be given to all people with lower UTI about managing symptoms?",
        "gold": "recommendation_1.1.2"
    },

    {
        "question": "What should be done if symptoms do not start to improve within 48 hours or worsen at any time for women with lower UTI who are not pregnant?",
        "gold": "recommendation_1.1.3"
    },

    {
        "question": "What should be done when microbiological results are available after a urine sample was sent for culture and susceptibility testing?",
        "gold": "recommendation_1.1.4"
    },

    {
        "question": "What should be done for pregnant women and men with lower UTI?",
        "gold": "recommendation_1.1.5"
    },

    {
        "question": "What should people with lower UTI be advised to use for pain?",
        "gold": "recommendation_1.3.1"
    },

    {
        "question": "What should be considered when prescribing antibiotics for lower UTI?",
        "gold": "recommendation_1.4.1"
    },

    {
        "question": "What antibiotics are recommended for non-pregnant women aged 16 years and over?",
        "gold": "TABLE_1"
    },

    {
        "question": "What antibiotics are recommended for pregnant women aged 12 years and over?",
        "gold": "TABLE_2"
    },

    {
        "question": "What antibiotics are recommended for men aged 16 years and over?",
        "gold": "TABLE_3"
    },

    {
        "question": "What antibiotics are recommended for children and young people under 16 years?",
        "gold": "TABLE_4"
    }
]


# ============================================================
# 4) RETRIEVAL FUNCTION
# ============================================================

def search(query, k):

    embedding = model.encode(
        [query],
        normalize_embeddings=True
    )[0].tolist()

    results = collection.query(
        query_embeddings=[embedding],
        n_results=k
    )

    return results


# ============================================================
# 5) EVALUATE RECALL@K
# ============================================================

def evaluate_recall(k):

    correct = 0

    for case in TEST_CASES:

        results = search(
            case["question"],
            k
        )

        ids = results["ids"][0]

        if case["gold"] in ids:
            correct += 1

    return correct / len(TEST_CASES)


# ============================================================
# 6) EVALUATE MRR
# ============================================================

def evaluate_mrr(k):

    total = 0

    for case in TEST_CASES:

        results = search(
            case["question"],
            k
        )

        ids = results["ids"][0]

        if case["gold"] in ids:

            rank = ids.index(case["gold"]) + 1

            total += 1 / rank

    return total / len(TEST_CASES)


# ============================================================
# 7) TEST DIFFERENT K VALUES
# ============================================================

print("\n")
print("=" * 70)
print("RETRIEVAL TOP-K TUNING")
print("=" * 70)

print("\nTesting different K values...\n")

k_values = [1, 2, 3, 4, 5, 7, 10]

results = []

for k in k_values:

    recall = evaluate_recall(k)

    mrr = evaluate_mrr(k)

    results.append(
        {
            "k": k,
            "recall": recall,
            "mrr": mrr
        }
    )

    print(
        f"Top-K = {k:2d} | "
        f"Recall@{k} = {recall:.4f} | "
        f"MRR = {mrr:.4f}"
    )


# ============================================================
# 8) FIND BEST K
# ============================================================

best = max(
    results,
    key=lambda x: (x["recall"], x["mrr"])
)

print("\n")
print("=" * 70)
print("BEST RESULT")
print("=" * 70)

print(
    f"Best Top-K: {best['k']}"
)

print(
    f"Recall: {best['recall']:.4f}"
)

print(
    f"MRR: {best['mrr']:.4f}"
)

print("\nTuning completed.")