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
# 3) TEST CASES
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
# 4) SEARCH
# ============================================================

def search(query, k=5):

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
# 5) CONVERT DISTANCE TO SIMILARITY
# ============================================================

def distance_to_similarity(distance):

    # Chroma returns cosine distance.
    # For this setup, similarity is approximated as:

    return 1 - distance


# ============================================================
# 6) TEST THRESHOLD
# ============================================================

def evaluate_threshold(threshold, k=5):

    found = 0
    total_retrieved = 0
    total_relevant = 0

    for case in TEST_CASES:

        results = search(
            case["question"],
            k
        )

        ids = results["ids"][0]
        distances = results["distances"][0]

        filtered_ids = []

        for doc_id, distance in zip(ids, distances):

            similarity = distance_to_similarity(
                distance
            )

            if similarity >= threshold:

                filtered_ids.append(
                    doc_id
                )

        total_retrieved += len(filtered_ids)

        if case["gold"] in filtered_ids:

            found += 1
            total_relevant += 1

    recall = found / len(TEST_CASES)

    avg_results = (
        total_retrieved / len(TEST_CASES)
    )

    return recall, avg_results


# ============================================================
# 7) RUN TESTS
# ============================================================

print("\n")
print("=" * 70)
print("SIMILARITY THRESHOLD TUNING")
print("=" * 70)

print("\nTop-K fixed at 5")
print("Testing different similarity thresholds...\n")


thresholds = [
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90
]


best = None


for threshold in thresholds:

    recall, avg_results = evaluate_threshold(
        threshold,
        k=5
    )

    print(
        f"Threshold = {threshold:.2f} | "
        f"Recall = {recall:.4f} | "
        f"Avg returned = {avg_results:.2f}"
    )

    # Prefer higher recall.
    # If recall is equal, prefer fewer returned chunks.

    current = (
        recall,
        -avg_results
    )

    if best is None or current > best["score"]:

        best = {
            "threshold": threshold,
            "recall": recall,
            "avg_results": avg_results,
            "score": current
        }


# ============================================================
# 8) BEST THRESHOLD
# ============================================================

print("\n")
print("=" * 70)
print("BEST THRESHOLD")
print("=" * 70)

print(
    f"Threshold: {best['threshold']:.2f}"
)

print(
    f"Recall: {best['recall']:.4f}"
)

print(
    f"Average returned chunks: "
    f"{best['avg_results']:.2f}"
)

print("\nThreshold tuning completed.")