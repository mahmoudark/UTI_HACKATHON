import sys

from ask import (
    hybrid_search,
    TOP_K,
    SIMILARITY_THRESHOLD,
    LEXICAL_WEIGHT,
    MODEL_NAME,
    collection
)


# ============================================================
# FINAL PRODUCTION RETRIEVAL EVALUATION
# ============================================================

print()
print("=" * 75)
print("FINAL PRODUCTION RETRIEVAL EVALUATION")
print("=" * 75)

print()
print("Production Retrieval Configuration")
print("-----------------------------------")
print("Embedding Model:", MODEL_NAME)
print("Top-K:", TOP_K)
print("Similarity Threshold:", SIMILARITY_THRESHOLD)
print("Lexical Weight:", LEXICAL_WEIGHT)
print("Documents:", collection.count())


# ============================================================
# TEST DATA
# These IDs match the CURRENT MPNet database metadata
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
    ),
]


# ============================================================
# RETRIEVE USING THE ACTUAL PRODUCTION PIPELINE
# ============================================================

def retrieve(question, k):

    results = hybrid_search(
        question,
        top_k=k
    )

    retrieved_ids = []

    for result in results:

        metadata = result.get(
            "metadata",
            {}
        )

        source_id = metadata.get(
            "source_id"
        )

        if source_id is not None:

            retrieved_ids.append(
                str(source_id)
            )

    return retrieved_ids, results


# ============================================================
# EVALUATION
# ============================================================

def evaluate(k):

    recall_hits = 0
    precision_sum = 0.0
    reciprocal_sum = 0.0

    details = []

    for question, expected in test_cases:

        retrieved, results = retrieve(
            question,
            k
        )

        # ----------------------------------------------------
        # Recall@K
        # ----------------------------------------------------

        if expected in retrieved:

            recall_hits += 1

            rank = (
                retrieved.index(expected)
                + 1
            )

            reciprocal_sum += (
                1 / rank
            )

        # ----------------------------------------------------
        # Precision@K
        # ----------------------------------------------------

        relevant = 0

        for doc_id in retrieved:

            if doc_id == expected:

                relevant += 1

        # Important:
        # denominator is actual K requested,
        # matching the original evaluation design.

        precision_sum += (
            relevant / k
        )

        details.append({
            "question": question,
            "expected": expected,
            "retrieved": retrieved
        })

    total = len(test_cases)

    recall = (
        recall_hits / total
    )

    precision = (
        precision_sum / total
    )

    mrr = (
        reciprocal_sum / total
    )

    return (
        recall,
        precision,
        mrr,
        details
    )


# ============================================================
# RUN EVALUATION
# ============================================================

print()
print("=" * 75)
print("RETRIEVAL METRICS")
print("=" * 75)

print()
print(
    "K    Recall      Precision      MRR        Balanced"
)
print("-" * 75)


k_values = [
    1,
    2,
    3,
    4,
    5,
    7,
    10
]


best_k = None
best_score = -1

all_results = {}


for k in k_values:

    recall, precision, mrr, details = evaluate(k)

    balanced_score = (
        recall
        + precision
        + mrr
    ) / 3

    all_results[k] = {
        "recall": recall,
        "precision": precision,
        "mrr": mrr,
        "balanced": balanced_score,
        "details": details
    }

    print(
        f"{k:<4}"
        f"{recall:.4f}      "
        f"{precision:.4f}        "
        f"{mrr:.4f}      "
        f"{balanced_score:.4f}"
    )

    if balanced_score > best_score:

        best_score = balanced_score
        best_k = k


# ============================================================
# FINAL RECOMMENDATION
# ============================================================

print()
print("=" * 75)
print("FINAL RECOMMENDATION")
print("=" * 75)

print()
print(
    f"Recommended Top-K: {best_k}"
)

print(
    f"Balanced Score: {best_score:.4f}"
)


# ============================================================
# DETAILED RESULTS FOR RECOMMENDED K
# ============================================================

print()
print("=" * 75)
print(
    f"DETAILED RESULTS @ K={best_k}"
)
print("=" * 75)


for i, item in enumerate(
    all_results[best_k]["details"],
    start=1
):

    expected = item["expected"]
    retrieved = item["retrieved"]

    if expected in retrieved:

        rank = (
            retrieved.index(expected)
            + 1
        )

        status = "PASS"

    else:

        rank = "NOT FOUND"
        status = "FAIL"

    print()
    print(
        f"[{i}/{len(test_cases)}] {status}"
    )

    print(
        "Question:",
        item["question"]
    )

    print(
        "Expected:",
        expected
    )

    print(
        "Rank:",
        rank
    )

    print(
        "Retrieved:",
        " | ".join(retrieved)
    )


# ============================================================
# SUMMARY
# ============================================================

final_metrics = all_results[
    best_k
]

print()
print("=" * 75)
print("FINAL SUMMARY")
print("=" * 75)

print()
print(
    f"Top-K       : {best_k}"
)

print(
    f"Recall@{best_k:<4}: "
    f"{final_metrics['recall']:.4f}"
)

print(
    f"Precision@{best_k:<1}: "
    f"{final_metrics['precision']:.4f}"
)

print(
    f"MRR         : "
    f"{final_metrics['mrr']:.4f}"
)

print(
    f"Balanced    : "
    f"{final_metrics['balanced']:.4f}"
)

print()
print("Evaluation completed.")