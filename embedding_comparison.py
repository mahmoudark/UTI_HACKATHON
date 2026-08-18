import json
import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# EMBEDDING MODEL COMPARISON
# ============================================================

print("=" * 70)
print("EMBEDDING MODEL COMPARISON")
print("=" * 70)


# ============================================================
# 1) LOAD DATA
# ============================================================

with open("documents.json", "r", encoding="utf-8") as f:
    documents = json.load(f)

print("Documents:", len(documents))


# ============================================================
# 2) TEST QUESTIONS
# ============================================================

test_cases = [

    ("What is a lower urinary tract infection?", "1.1.1"),

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
# 3) MODELS
# ============================================================

models = [

    "all-MiniLM-L6-v2",

    "all-mpnet-base-v2"

]


# ============================================================
# 4) EVALUATE MODEL
# ============================================================

def evaluate_model(model_name):

    print()
    print("=" * 70)
    print("MODEL:", model_name)
    print("=" * 70)

    print("Loading model...")

    model = SentenceTransformer(model_name)

    print("Model loaded")

    # --------------------------------------------------------
    # Create document embeddings
    # --------------------------------------------------------

    texts = [
        document["text"]
        for document in documents
    ]

    print("Creating document embeddings...")

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    total_relevant = 0
    total_retrieved = 0

    reciprocal_ranks = []

    for question, expected in test_cases:

        query_embedding = model.encode(
            question,
            normalize_embeddings=True
        )

        scores = np.dot(
            embeddings,
            query_embedding
        )

        top_indices = np.argsort(
            scores
        )[::-1][:5]

        retrieved = []

        for index in top_indices:

            retrieved.append({
                "source_id": documents[index]["source_id"],
                "score": float(scores[index])
            })

        # ----------------------------------------------------
        # Check relevance
        # ----------------------------------------------------

        relevant_positions = []

        for rank, result in enumerate(
            retrieved,
            start=1
        ):

            if result["source_id"] == expected:
                relevant_positions.append(rank)

        # ----------------------------------------------------
        # Precision@5
        # ----------------------------------------------------

        relevant_count = len(
            relevant_positions
        )

        total_relevant += relevant_count
        total_retrieved += len(retrieved)

        # ----------------------------------------------------
        # MRR
        # ----------------------------------------------------

        if relevant_positions:

            reciprocal_ranks.append(
                1 / relevant_positions[0]
            )

        else:

            reciprocal_ranks.append(0)

        print()
        print("Question:", question)
        print("Expected:", expected)

        print("Top 5:")

        for rank, result in enumerate(
            retrieved,
            start=1
        ):

            print(
                f"  {rank}. "
                f"{result['source_id']} "
                f"| score: "
                f"{result['score']:.4f}"
            )

    # --------------------------------------------------------
    # Final metrics
    # --------------------------------------------------------

    precision_at_5 = (
        total_relevant /
        total_retrieved
    )

    recall_at_5 = (
        sum(
            1
            for x in reciprocal_ranks
            if x > 0
        )
        /
        len(test_cases)
    )

    mrr = (
        sum(reciprocal_ranks)
        /
        len(reciprocal_ranks)
    )

    print()
    print("-" * 70)
    print("MODEL RESULTS")
    print("-" * 70)

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

    return {
        "model": model_name,
        "precision": precision_at_5,
        "recall": recall_at_5,
        "mrr": mrr
    }


# ============================================================
# 5) RUN BOTH MODELS
# ============================================================

results = []

for model_name in models:

    results.append(
        evaluate_model(model_name)
    )


# ============================================================
# 6) FINAL COMPARISON
# ============================================================

print()
print()
print("=" * 70)
print("FINAL EMBEDDING MODEL COMPARISON")
print("=" * 70)

print()

print(
    "Model".ljust(30),
    "Precision@5".ljust(15),
    "Recall@5".ljust(15),
    "MRR"
)

print("-" * 70)

for result in results:

    print(
        result["model"].ljust(30),
        f"{result['precision']:.4f}".ljust(15),
        f"{result['recall']:.4f}".ljust(15),
        f"{result['mrr']:.4f}"
    )


# ============================================================
# 7) BEST MODEL
# ============================================================

best = max(
    results,
    key=lambda x: (
        x["recall"],
        x["precision"],
        x["mrr"]
    )
)

print()
print("=" * 70)
print("BEST EMBEDDING MODEL")
print("=" * 70)

print(
    "Model:",
    best["model"]
)

print(
    "Precision@5:",
    f"{best['precision']:.4f}"
)

print(
    "Recall@5:",
    f"{best['recall']:.4f}"
)

print(
    "MRR:",
    f"{best['mrr']:.4f}"
)

print()
print("Experiment completed.")