import chromadb
from sentence_transformers import SentenceTransformer
import re


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
# 4) TITLE MATCHING
# ============================================================

def title_matches(query, metadata):

    title = metadata.get("title", "").lower()

    query_lower = query.lower()

    query_words = re.findall(
        r"[a-zA-Z]+",
        query_lower
    )

    matches = 0

    for word in query_words:

        if len(word) < 3:
            continue

        if word in title:
            matches += 1

    # Clinical table matching
    if "non-pregnant" in query_lower and "non-pregnant" in title:
        matches += 3

    if "pregnant" in query_lower and "pregnant" in title:
        matches += 3

    if "men" in query_lower and "men" in title:
        matches += 3

    if "children" in query_lower and "children" in title:
        matches += 3

    if "under 16" in query_lower and "under 16" in title:
        matches += 2

    if "16 years and over" in query_lower and "16 years and over" in title:
        matches += 2

    return matches


# ============================================================
# 5) HYBRID SEARCH
# ============================================================

def hybrid_search(query, k=5):

    embedding = model.encode(
        [query],
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

        semantic_score = 1 - distance

        matches = title_matches(
            query,
            metadata
        )

        hybrid_score = (
            semantic_score
            + (matches * 0.25)
        )

        candidates.append({
            "id": metadata.get("source_id"),
            "hybrid_score": hybrid_score,
            "matches": matches
        })

    candidates.sort(
        key=lambda x: x["hybrid_score"],
        reverse=True
    )

    return candidates[:k]


# ============================================================
# 6) THRESHOLD EVALUATION
# ============================================================

def evaluate_threshold(threshold, k=5):

    found = 0

    total_returned = 0

    for case in TEST_CASES:

        results = hybrid_search(
            case["question"],
            k
        )

        filtered = [
            result
            for result in results
            if result["hybrid_score"] >= threshold
        ]

        total_returned += len(filtered)

        ids = [
            result["id"]
            for result in filtered
        ]

        if case["gold"] in ids:

            found += 1

    recall = found / len(TEST_CASES)

    avg_returned = (
        total_returned / len(TEST_CASES)
    )

    return recall, avg_returned


# ============================================================
# 7) TEST DIFFERENT THRESHOLDS
# ============================================================

print("\n")
print("=" * 70)
print("HYBRID SCORE THRESHOLD TUNING")
print("=" * 70)

print("\nTop-K fixed at 5")
print("Testing hybrid score thresholds...\n")


thresholds = [
    0.5,
    0.75,
    1.0,
    1.25,
    1.5,
    1.75,
    2.0,
    2.25,
    2.5,
    2.75,
    3.0,
    3.25,
    3.5,
    4.0,
    4.5,
    5.0
]


best = None


for threshold in thresholds:

    recall, avg_returned = evaluate_threshold(
        threshold,
        k=5
    )

    print(
        f"Threshold = {threshold:4.2f} | "
        f"Recall = {recall:.4f} | "
        f"Avg returned = {avg_returned:.2f}"
    )

    # We want:
    # 1. Highest recall
    # 2. Fewer chunks if recall is equal

    current_score = (
        recall,
        -avg_returned
    )

    if best is None or current_score > best["score"]:

        best = {
            "threshold": threshold,
            "recall": recall,
            "avg_returned": avg_returned,
            "score": current_score
        }


# ============================================================
# 8) BEST RESULT
# ============================================================

print("\n")
print("=" * 70)
print("BEST HYBRID THRESHOLD")
print("=" * 70)

print(
    f"Threshold: {best['threshold']:.2f}"
)

print(
    f"Recall: {best['recall']:.4f}"
)

print(
    f"Average returned chunks: "
    f"{best['avg_returned']:.2f}"
)

print("\nHybrid threshold tuning completed.")