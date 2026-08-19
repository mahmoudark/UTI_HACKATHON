import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder


# ============================================================
# MODELS
# ============================================================

print("Loading MPNet...")

embedding_model = SentenceTransformer(
    "all-mpnet-base-v2"
)

print("MPNet loaded")

print("Loading reranker...")

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

print("Reranker loaded")


# ============================================================
# DATABASE
# ============================================================

client = chromadb.PersistentClient(
    path="./chroma_db_mpnet"
)

collection = client.get_collection(
    name="uti_guideline_mpnet"
)

print("Documents:", collection.count())


# ============================================================
# TEST SET
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


TOP_K_RETRIEVAL = 10
TOP_K_FINAL = 5


# ============================================================
# EVALUATION
# ============================================================

successful = 0
reciprocal_ranks = []

print()
print("=" * 70)
print("MPNET + RERANKER EVALUATION")
print("=" * 70)


for question, expected in test_cases:

    print()
    print("=" * 70)
    print("QUESTION")
    print("=" * 70)

    print(question)
    print("Expected:", expected)

    # --------------------------------------------------------
    # Step 1: MPNet retrieval
    # --------------------------------------------------------

    query_embedding = embedding_model.encode(
        [question],
        normalize_embeddings=True
    )[0].tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K_RETRIEVAL
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    # --------------------------------------------------------
    # Step 2: Cross-Encoder reranking
    # --------------------------------------------------------

    pairs = [
        (question, document)
        for document in documents
    ]

    rerank_scores = reranker.predict(pairs)

    candidates = []

    for document, metadata, score in zip(
        documents,
        metadatas,
        rerank_scores
    ):

        candidates.append({
            "document": document,
            "metadata": metadata,
            "score": float(score)
        })

    candidates.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    final_results = candidates[:TOP_K_FINAL]

    # --------------------------------------------------------
    # Show results
    # --------------------------------------------------------

    found_rank = None

    print()
    print("RERANKED TOP 5")
    print("-" * 70)

    for rank, result in enumerate(
        final_results,
        start=1
    ):

        source_id = result["metadata"]["source_id"]

        relevant = (
            source_id == expected
        )

        if relevant and found_rank is None:
            found_rank = rank

        print(
            f"{rank}. "
            f"{source_id}"
            f" | Score: {result['score']:.4f}"
            f" | Relevant: "
            f"{'YES' if relevant else 'NO'}"
        )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    if found_rank is not None:

        successful += 1

        reciprocal_ranks.append(
            1 / found_rank
        )

    else:

        reciprocal_ranks.append(0)


# ============================================================
# FINAL METRICS
# ============================================================

recall_at_5 = (
    successful /
    len(test_cases)
)

mrr = (
    sum(reciprocal_ranks) /
    len(reciprocal_ranks)
)

precision_at_5 = (
    successful /
    (len(test_cases) * TOP_K_FINAL)
)


print()
print()
print("=" * 70)
print("RERANKER FINAL RESULTS")
print("=" * 70)

print(
    "Recall@5:",
    f"{recall_at_5:.4f}"
)

print(
    "Precision@5:",
    f"{precision_at_5:.4f}"
)

print(
    "MRR:",
    f"{mrr:.4f}"
)

print()
print("Evaluation completed.")