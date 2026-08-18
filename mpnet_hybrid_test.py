import chromadb
from sentence_transformers import SentenceTransformer
import re


# ============================================================
# SETTINGS
# ============================================================

TOP_K = 5
SIMILARITY_THRESHOLD = 0.25


# ============================================================
# LOAD MPNet
# ============================================================

print("Loading MPNet model...")

model = SentenceTransformer(
    "all-mpnet-base-v2"
)

print("MPNet model loaded")


# ============================================================
# LOAD MPNet DATABASE
# ============================================================

client = chromadb.PersistentClient(
    path="./chroma_db_mpnet"
)

collection = client.get_collection(
    name="uti_guideline_mpnet"
)

print("MPNet vector database loaded")
print("Documents:", collection.count())


# ============================================================
# TITLE MATCHING
# ============================================================

def title_matches(query, metadata):

    title = metadata.get("title", "").lower()
    query_lower = query.lower()

    matches = 0

    strong_phrases = [
        "non-pregnant women",
        "pregnant women",
        "men aged 16 years and over",
        "children and young people under 16 years",
        "16 years and over",
        "under 16 years",
        "all people with lower uti",
        "self-care",
        "choice of antibiotic",
        "microbiological results",
        "pregnant women and men"
    ]

    for phrase in strong_phrases:

        if phrase in query_lower and phrase in title:
            matches += 4

    return matches


# ============================================================
# HYBRID SEARCH
# ============================================================

def hybrid_search(query):

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

        similarity = 1 - distance

        # Threshold
        if similarity < SIMILARITY_THRESHOLD:
            continue

        matches = title_matches(
            query,
            metadata
        )

        hybrid_score = (
            similarity +
            matches * 0.25
        )

        candidates.append({
            "document": document,
            "metadata": metadata,
            "similarity": similarity,
            "matches": matches,
            "hybrid_score": hybrid_score
        })

    candidates.sort(
        key=lambda x: x["hybrid_score"],
        reverse=True
    )

    return candidates[:TOP_K]


# ============================================================
# TEST QUESTIONS
# ============================================================

test_cases = [

    (
        "What antibiotics are recommended for non-pregnant women aged 16 years and over?",
        "TABLE_1"
    ),

    (
        "What antibiotics are recommended for pregnant women aged 12 years and over?",
        "TABLE_2"
    ),

    (
        "What antibiotics are recommended for men aged 16 years and over?",
        "TABLE_3"
    ),

    (
        "What antibiotics are recommended for children and young people under 16 years?",
        "TABLE_4"
    ),

    (
        "What should people with lower UTI be advised to use for pain?",
        "1.3.1"
    )
]


# ============================================================
# RUN TESTS
# ============================================================

successful = 0


for question, expected in test_cases:

    results = hybrid_search(question)

    print()
    print("=" * 70)
    print("QUESTION")
    print("=" * 70)

    print(question)

    print()
    print("EXPECTED:", expected)

    print()
    print("RETRIEVED")
    print("-" * 70)

    found = False

    for rank, result in enumerate(
        results,
        start=1
    ):

        metadata = result["metadata"]

        source_id = metadata.get(
            "source_id"
        )

        print(
            f"{rank}. {source_id}"
        )

        print(
            f"   Similarity: "
            f"{result['similarity']:.4f}"
        )

        print(
            f"   Title matches: "
            f"{result['matches']}"
        )

        print(
            f"   Hybrid score: "
            f"{result['hybrid_score']:.4f}"
        )

        if source_id == expected:
            found = True

    print()

    if found:

        print("RESULT: PASS")
        successful += 1

    else:

        print("RESULT: FAIL")


# ============================================================
# FINAL
# ============================================================

print()
print()
print("=" * 70)
print("MPNET HYBRID FINAL TEST")
print("=" * 70)

print(
    f"Passed: {successful}/{len(test_cases)}"
)

print(
    f"Recall@5: "
    f"{successful / len(test_cases):.4f}"
)

print()
print("Test completed.")