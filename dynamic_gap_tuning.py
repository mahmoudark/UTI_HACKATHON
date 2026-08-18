import chromadb
from sentence_transformers import SentenceTransformer
import re


print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding model loaded")


client = chromadb.PersistentClient(
    path="./chroma_db"
)

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

    return candidates


# ============================================================
# DYNAMIC SELECTION
# ============================================================

def dynamic_select(results, gap):

    if not results:
        return []

    selected = [results[0]]

    best_score = results[0]["hybrid_score"]

    for result in results[1:]:

        score_drop = best_score - result["hybrid_score"]

        if score_drop <= gap:
            selected.append(result)

        else:
            break

    return selected


# ============================================================
# EVALUATION QUESTIONS
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
# TEST DIFFERENT GAPS
# ============================================================

gaps = [
    0.25,
    0.50,
    0.75,
    1.00,
    1.25,
    1.50
]


print()
print("=" * 70)
print("DYNAMIC CHUNK GAP TUNING")
print("=" * 70)

print()
print("Testing different score gaps...")


best_gap = None
best_recall = -1
best_avg_chunks = 999


for gap in gaps:

    total = len(test_cases)

    found = 0
    total_chunks = 0
    reciprocal_sum = 0


    for question, expected in test_cases:

        results = hybrid_search(question)

        selected = dynamic_select(
            results,
            gap
        )

        total_chunks += len(selected)


        ranks = []

        for rank, result in enumerate(
            selected,
            start=1
        ):

            source_id = result["metadata"].get(
                "source_id"
            )

            if source_id == expected:
                ranks.append(rank)


        if ranks:

            found += 1

            reciprocal_sum += 1 / ranks[0]


    recall = found / total

    mrr = reciprocal_sum / total

    avg_chunks = total_chunks / total


    print(
        f"Gap = {gap:>4.2f} | "
        f"Recall = {recall:.4f} | "
        f"MRR = {mrr:.4f} | "
        f"Avg chunks = {avg_chunks:.2f}"
    )


    # Prefer higher recall.
    # If recall is equal, prefer fewer chunks.

    if (
        recall > best_recall
        or (
            recall == best_recall
            and avg_chunks < best_avg_chunks
        )
    ):

        best_gap = gap
        best_recall = recall
        best_avg_chunks = avg_chunks


# ============================================================
# FINAL RESULT
# ============================================================

print()
print("=" * 70)
print("BEST DYNAMIC GAP")
print("=" * 70)

print(
    "Best gap:",
    best_gap
)

print(
    "Recall:",
    round(best_recall, 4)
)

print(
    "Average chunks:",
    round(best_avg_chunks, 2)
)

print()
print("Dynamic gap tuning completed.")