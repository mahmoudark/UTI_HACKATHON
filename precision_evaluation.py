import chromadb
from sentence_transformers import SentenceTransformer
import re


# ============================================================
# LOAD MODEL + DATABASE
# ============================================================

print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding model loaded")

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_collection(
    name="uti_guideline"
)

print("Vector database loaded")
print("Documents:", collection.count())


# ============================================================
# TITLE MATCHING
# ============================================================

def title_matches(query, metadata):

    title = metadata.get("title", "").lower()
    query_lower = query.lower()

    matches = 0

    # --------------------------------------------------------
    # Only give title bonus for strong, specific phrases.
    # Do NOT reward common words such as:
    # what, should, women, lower, UTI, etc.
    # --------------------------------------------------------

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
        "pregnant women and men",
    ]

    for phrase in strong_phrases:

        if phrase in query_lower and phrase in title:
            matches += 4

    return matches


# ============================================================
# HYBRID SEARCH
# ============================================================

def hybrid_search(query, top_k=5):

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

        matches = title_matches(
            query,
            metadata
        )

        score = (
            similarity +
            matches * 0.25
        )

        candidates.append({
            "document": document,
            "metadata": metadata,
            "similarity": similarity,
            "matches": matches,
            "score": score
        })

    candidates.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return candidates[:top_k]


# ============================================================
# TEST DATA
# ============================================================

test_cases = [

    (
        "What is a lower urinary tract infection?",
        "recommendation_1.1.1"
    ),

    (
        "What advice should be given to all people with lower UTI about managing symptoms?",
        "recommendation_1.1.2"
    ),

    (
        "What should be done if symptoms do not start to improve within 48 hours or worsen at any time for women with lower UTI who are not pregnant?",
        "recommendation_1.1.3"
    ),

    (
        "What should be done when microbiological results are available after a urine sample was sent for culture and susceptibility testing?",
        "recommendation_1.1.4"
    ),

    (
        "What should be done for pregnant women and men with lower UTI?",
        "recommendation_1.1.5"
    ),

    (
        "What should people with lower UTI be advised to use for pain?",
        "recommendation_1.3.1"
    ),

    (
        "What should be considered when prescribing antibiotics for lower UTI?",
        "recommendation_1.4.1"
    ),

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
    )
]


# ============================================================
# PRECISION EVALUATION
# ============================================================

print()
print("=" * 75)
print("PRECISION EVALUATION")
print("=" * 75)

total_relevant = 0
total_retrieved = 0

precisions = []

for question, expected in test_cases:

    results = hybrid_search(
        question,
        top_k=5
    )

    relevant_count = 0

    print()
    print("=" * 75)
    print("QUESTION")
    print("=" * 75)

    print(question)

    print()
    print("EXPECTED:")
    print(expected)

    print()
    print("RETRIEVED CHUNKS")
    print("-" * 75)

    for rank, result in enumerate(
        results,
        start=1
    ):

        metadata = result["metadata"]

        source_id = metadata.get(
            "source_id"
        )

        score = result["score"]
        is_relevant = (
            source_id == expected
            or source_id == expected.replace(
                "recommendation_",
                ""
            )
        )

        if is_relevant:
            relevant_count += 1

        print(
            f"{rank}. "
            f"{source_id}"
        )

        print(
            f"   Similarity: {result['similarity']:.4f}"
        )

        print(
            f"   Title matches: {result['matches']}"
        )

        print(
            f"   Hybrid Score: {score:.4f}"
        )

        print(
            "   Relevant:",
            "YES" if is_relevant else "NO"
        )

    precision = (
        relevant_count / len(results)
        if results
        else 0
    )

    precisions.append(precision)

    total_relevant += relevant_count
    total_retrieved += len(results)

    print()
    print(
        f"Precision = {relevant_count}/{len(results)} "
        f"= {precision:.4f}"
    )


# ============================================================
# FINAL METRICS
# ============================================================

average_precision = (
    sum(precisions) / len(precisions)
)

overall_precision = (
    total_relevant / total_retrieved
)


print()
print()
print("=" * 75)
print("FINAL PRECISION RESULTS")
print("=" * 75)

print(
    "Total relevant retrieved:",
    total_relevant
)

print(
    "Total retrieved:",
    total_retrieved
)

print(
    "Overall Precision:",
    round(overall_precision, 4)
)

print(
    "Average Precision:",
    round(average_precision, 4)
)

print()
print("Precision evaluation completed.")