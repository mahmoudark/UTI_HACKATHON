import chromadb
from sentence_transformers import SentenceTransformer


# ============================================================
# LOAD MPNet MODEL
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
# TEST QUESTIONS
# ============================================================

test_cases = [

    (
        "What is a lower urinary tract infection?",
        "1.1.1"
    ),

    (
        "What advice should be given to all people with lower UTI about managing symptoms?",
        "1.1.2"
    ),

    (
        "What should be done if symptoms do not start to improve within 48 hours or worsen at any time for women with lower UTI who are not pregnant?",
        "1.1.3"
    ),

    (
        "What should be done when microbiological results are available after a urine sample was sent for culture and susceptibility testing?",
        "1.1.4"
    ),

    (
        "What should be done for pregnant women and men with lower UTI?",
        "1.1.5"
    ),

    (
        "What should people with lower UTI be advised to use for pain?",
        "1.3.1"
    ),

    (
        "What should be considered when prescribing antibiotics for lower UTI?",
        "1.4.1"
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
# EVALUATION
# ============================================================

TOP_K = 5

total_relevant = 0
total_retrieved = 0
successful_queries = 0

reciprocal_ranks = []


for question, expected in test_cases:

    query_embedding = model.encode(
        [question],
        normalize_embeddings=True
    )[0].tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K
    )

    retrieved_ids = results["metadatas"][0]

    distances = results["distances"][0]

    print()
    print("=" * 70)
    print("QUESTION")
    print("=" * 70)

    print(question)

    print()
    print("Expected:", expected)

    print()
    print("TOP 5 RESULTS")
    print("-" * 70)

    found_rank = None

    for rank, (metadata, distance) in enumerate(
        zip(retrieved_ids, distances),
        start=1
    ):

        source_id = metadata["source_id"]

        similarity = 1 - distance

        is_relevant = (
            source_id == expected
        )

        if is_relevant and found_rank is None:
            found_rank = rank

        print(
            f"{rank}. "
            f"{source_id}"
            f" | Similarity: {similarity:.4f}"
            f" | Relevant: "
            f"{'YES' if is_relevant else 'NO'}"
        )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    total_retrieved += len(retrieved_ids)

    if found_rank is not None:

        total_relevant += 1
        successful_queries += 1

        reciprocal_ranks.append(
            1 / found_rank
        )

    else:

        reciprocal_ranks.append(0)


# ============================================================
# FINAL RESULTS
# ============================================================

precision_at_5 = (
    total_relevant /
    total_retrieved
)

recall_at_5 = (
    successful_queries /
    len(test_cases)
)

mrr = (
    sum(reciprocal_ranks) /
    len(reciprocal_ranks)
)


print()
print()
print("=" * 70)
print("MPNET FINAL RETRIEVAL EVALUATION")
print("=" * 70)

print()

print(
    "Top-K:",
    TOP_K
)

print(
    "Precision@5:",
    f"{precision_at_5:.4f}"
)

print(
    "Recall@5:",
    f"{recall_at_5:.4f}"
)

print(
    "MRR:",
    f"{mrr:.4f}"
)

print()
print("Evaluation completed.")