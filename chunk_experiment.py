import json
import re
import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# CHUNK SIZE / OVERLAP EXPERIMENT
# ============================================================

print("=" * 70)
print("CHUNK SIZE / OVERLAP EXPERIMENT")
print("=" * 70)


# ============================================================
# 1) LOAD DATASET
# ============================================================

with open("documents.json", "r", encoding="utf-8") as f:
    documents = json.load(f)

print("Documents loaded:", len(documents))


# ============================================================
# 2) LOAD EMBEDDING MODEL
# ============================================================

print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding model loaded")


# ============================================================
# 3) TEST QUESTIONS
# ============================================================

test_questions = [

    {
        "question": "What is a lower urinary tract infection?",
        "expected": "1.1.1"
    },

    {
        "question": "What advice should be given to all people with lower UTI about managing symptoms?",
        "expected": "1.1.2"
    },

    {
        "question": "What should be done if symptoms do not start to improve within 48 hours or worsen at any time for women with lower UTI who are not pregnant?",
        "expected": "1.1.3"
    },

    {
        "question": "What should people with lower UTI be advised to use for pain?",
        "expected": "1.3.1"
    },

    {
        "question": "What antibiotics are recommended for non-pregnant women aged 16 years and over?",
        "expected": "TABLE_1"
    }
]


# ============================================================
# 4) CHUNKING FUNCTION
# ============================================================

def create_chunks(text, chunk_size, overlap):

    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    chunks = []

    start = 0

    while start < len(text):

        end = min(
            start + chunk_size,
            len(text)
        )

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - overlap

    return chunks


# ============================================================
# 5) BUILD CHUNKS FOR ONE CONFIGURATION
# ============================================================

def build_configuration(chunk_size, overlap):

    all_chunks = []

    for document in documents:

        source_id = document["source_id"]

        source_type = document["source_type"]

        title = document["title"]

        pages = document["pages"]

        text = document["text"]

        chunks = create_chunks(
            text,
            chunk_size,
            overlap
        )

        for index, chunk_text in enumerate(chunks):

            all_chunks.append({
                "chunk_id": f"{source_id}_chunk_{index + 1}",
                "source_id": source_id,
                "source_type": source_type,
                "title": title,
                "pages": pages,
                "text": chunk_text
            })

    return all_chunks


# ============================================================
# 6) RETRIEVE TOP-K
# ============================================================

def retrieve(
    query,
    chunks,
    embeddings,
    top_k=5
):

    query_embedding = model.encode(
        query,
        normalize_embeddings=True
    )

    scores = np.dot(
        embeddings,
        query_embedding
    )

    top_indices = np.argsort(
        scores
    )[::-1][:top_k]

    results = []

    for index in top_indices:

        results.append({
            "source_id": chunks[index]["source_id"],
            "score": float(scores[index])
        })

    return results


# ============================================================
# 7) EVALUATE ONE CONFIGURATION
# ============================================================

def evaluate_configuration(
    chunk_size,
    overlap
):

    print()
    print("-" * 70)
    print(
        f"CONFIGURATION: Chunk Size = {chunk_size}, "
        f"Overlap = {overlap}"
    )
    print("-" * 70)

    # --------------------------------------------------------
    # Create chunks
    # --------------------------------------------------------

    chunks = build_configuration(
        chunk_size,
        overlap
    )

    print("Total chunks:", len(chunks))

    # --------------------------------------------------------
    # Create embeddings
    # --------------------------------------------------------

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    # --------------------------------------------------------
    # Evaluate questions
    # --------------------------------------------------------

    total_relevant = 0

    total_retrieved = 0

    successful_queries = 0

    for item in test_questions:

        query = item["question"]

        expected = item["expected"]

        results = retrieve(
            query,
            chunks,
            embeddings,
            top_k=5
        )

        retrieved_sources = [
            result["source_id"]
            for result in results
        ]

        relevant = expected in retrieved_sources

        if relevant:
            total_relevant += 1
            successful_queries += 1

        total_retrieved += 5

        print()
        print("Question:", query)
        print("Expected:", expected)

        print("Retrieved:")

        for rank, result in enumerate(
            results,
            start=1
        ):

            print(
                f"  {rank}. "
                f"{result['source_id']} "
                f"| score: "
                f"{result['score']:.4f}"
            )

        print(
            "Relevant:",
            "YES" if relevant else "NO"
        )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    precision_at_5 = (
        total_relevant /
        total_retrieved
    )

    recall_at_5 = (
        successful_queries /
        len(test_questions)
    )

    print()
    print("RESULTS")
    print("-" * 70)

    print(
        f"Precision@5 = "
        f"{precision_at_5:.4f}"
    )

    print(
        f"Recall@5    = "
        f"{recall_at_5:.4f}"
    )

    return {
        "chunk_size": chunk_size,
        "overlap": overlap,
        "num_chunks": len(chunks),
        "precision_at_5": precision_at_5,
        "recall_at_5": recall_at_5
    }


# ============================================================
# 8) RUN THREE CONFIGURATIONS
# ============================================================

configurations = [

    (200, 0),

    (400, 50),

    (600, 100)

]


results = []


for chunk_size, overlap in configurations:

    result = evaluate_configuration(
        chunk_size,
        overlap
    )

    results.append(result)


# ============================================================
# 9) FINAL COMPARISON
# ============================================================

print()
print()
print("=" * 70)
print("FINAL CHUNKING COMPARISON")
print("=" * 70)

print()

print(
    "Configuration".ljust(20),
    "Chunks".ljust(10),
    "Precision@5".ljust(15),
    "Recall@5"
)

print("-" * 70)

for result in results:

    configuration = (
        f"{result['chunk_size']} / "
        f"{result['overlap']}"
    )

    print(
        configuration.ljust(20),
        str(result["num_chunks"]).ljust(10),
        f"{result['precision_at_5']:.4f}".ljust(15),
        f"{result['recall_at_5']:.4f}"
    )


# ============================================================
# 10) BEST CONFIGURATION
# ============================================================

best = max(
    results,
    key=lambda x: (
        x["recall_at_5"],
        x["precision_at_5"]
    )
)

print()
print("=" * 70)
print("BEST CHUNKING CONFIGURATION")
print("=" * 70)

print(
    "Chunk Size:",
    best["chunk_size"]
)

print(
    "Overlap:",
    best["overlap"]
)

print(
    "Precision@5:",
    f"{best['precision_at_5']:.4f}"
)

print(
    "Recall@5:",
    f"{best['recall_at_5']:.4f}"
)

print()
print("Experiment completed.")