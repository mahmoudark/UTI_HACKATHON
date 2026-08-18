import chromadb
from sentence_transformers import SentenceTransformer


# ============================================================
# 1) LOAD EMBEDDING MODEL
# ============================================================

print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding model loaded")


# ============================================================
# 2) LOAD CHROMA DATABASE
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
# 3) TEST QUESTIONS
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
# 4) RETRIEVE TOP-K
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

    return ids


# ============================================================
# 5) PRECISION@K
# ============================================================

def precision_at_k(k):

    total_precision = 0

    for case in test_cases:

        question = case["question"]
        expected = case["expected"]

        retrieved_ids = retrieve(
            question,
            k
        )

        relevant_count = 0

        for document_id in retrieved_ids:

            if document_id == expected:
                relevant_count += 1

        precision = relevant_count / k

        total_precision += precision

    average_precision = (
        total_precision / len(test_cases)
    )

    return average_precision


# ============================================================
# 6) EVALUATE DIFFERENT K VALUES
# ============================================================

print()
print("=" * 70)
print("PRECISION@K EVALUATION")
print("=" * 70)

print()
print("Testing different K values...")
print()

k_values = [1, 2, 3, 4, 5, 7, 10]

best_k = None
best_precision = -1


for k in k_values:

    precision = precision_at_k(k)

    print(
        f"Top-K = {k:2d} | "
        f"Precision@{k} = {precision:.4f}"
    )

    if precision > best_precision:

        best_precision = precision
        best_k = k


# ============================================================
# 7) FINAL RESULT
# ============================================================

print()
print("=" * 70)
print("BEST PRECISION@K")
print("=" * 70)

print(
    "Best K:",
    best_k
)

print(
    "Precision:",
    round(best_precision, 4)
)

print()
print(
    f"Recommendation: Top-K = {best_k} "
    f"gave the highest Precision@K."
)

print()
print("Precision evaluation completed.")