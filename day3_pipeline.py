import chromadb
from sentence_transformers import SentenceTransformer


# ============================================================
# DAY 3 - GROUNDED GENERATION & CITATION
# ============================================================

MODEL_NAME = "all-mpnet-base-v2"
DATABASE_PATH = "./chroma_db_mpnet"
COLLECTION_NAME = "uti_guideline_mpnet"

TOP_K = 5

# Confidence threshold for refusal
MIN_SIMILARITY = 0.25


# ============================================================
# 1) GROUNDING SYSTEM PROMPT
# ============================================================

GROUNDING_PROMPT = """
You are a clinical evidence assistant for the UTI guideline.

STRICT GROUNDING RULES:

1. Answer ONLY from the retrieved evidence provided to you.
2. Do NOT use outside medical knowledge.
3. Do NOT invent facts, dosages, thresholds, durations, or recommendations.
4. Do NOT guess when the evidence is insufficient.
5. You may paraphrase the retrieved evidence for clarity.
6. If the evidence does not answer the question, refuse to answer.

Every answer must contain exactly these sections:

RECOMMENDATION:
Give a short direct answer supported by the evidence.

EXCERPT:
Provide the relevant retrieved evidence.

CITATION:
Include the document name, section/source ID, and page number.

If the evidence is insufficient, say:

"I couldn't find enough information in the indexed UTI
guideline to answer this confidently. Please rephrase the
question or consult a clinician directly."
"""


# ============================================================
# 2) LOAD MODEL
# ============================================================

print("Loading embedding model...")

model = SentenceTransformer(
    MODEL_NAME
)

print("Embedding model loaded")


# ============================================================
# 3) LOAD VECTOR DATABASE
# ============================================================

client = chromadb.PersistentClient(
    path=DATABASE_PATH
)

collection = client.get_collection(
    name=COLLECTION_NAME
)

print("Vector database loaded")
print("Documents:", collection.count())


# ============================================================
# 4) RETRIEVAL
# ============================================================

def retrieve(query):

    embedding = model.encode(
        [query],
        normalize_embeddings=True
    )[0].tolist()

    results = collection.query(
        query_embeddings=[embedding],
        n_results=TOP_K
    )

    retrieved = []

    for document, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):

        similarity = 1 - distance

        retrieved.append({
            "text": document,
            "metadata": metadata,
            "similarity": similarity
        })

    return retrieved


# ============================================================
# 5) REFUSAL LOGIC
# ============================================================

def should_refuse(results):

    if not results:
        return True

    best_similarity = results[0]["similarity"]

    if best_similarity < MIN_SIMILARITY:
        return True

    return False


# ============================================================
# 6) GROUNDED RESPONSE
# ============================================================

def generate_grounded_response(
    query,
    results
):

    if should_refuse(results):

        return {
            "refused": True,
            "response": (
                "I couldn't find enough information in the "
                "indexed UTI guideline to answer this confidently. "
                "Please rephrase the question or consult a "
                "clinician directly."
            )
        }


    best = results[0]

    text = best["text"]

    metadata = best["metadata"]

    source_id = metadata.get(
        "source_id",
        "Unknown"
    )

    title = metadata.get(
        "title",
        "UTI guideline"
    )

    pages = metadata.get(
        "pages",
        "Unknown"
    )

    # --------------------------------------------------------
    # Structured answer
    # --------------------------------------------------------

    response = f"""
RECOMMENDATION:
According to the retrieved UTI guideline evidence, the relevant
recommendation is provided in source {source_id}.

EXCERPT:
{text}

CITATION:
[UTI guideline, Section {source_id}, Page {pages}]
"""

    return {
        "refused": False,
        "response": response,
        "source_id": source_id,
        "title": title,
        "pages": pages,
        "similarity": best["similarity"]
    }


# ============================================================
# 7) FULL PIPELINE
# ============================================================

def run_pipeline(query):

    print()
    print("=" * 70)
    print("QUERY")
    print("=" * 70)

    print(query)

    print()
    print("Retrieving evidence...")

    results = retrieve(query)

    print(
        "Retrieved:",
        len(results),
        "chunks"
    )

    if results:

        print(
            "Best similarity:",
            round(
                results[0]["similarity"],
                4
            )
        )

    # --------------------------------------------------------
    # Refusal check
    # --------------------------------------------------------

    if should_refuse(results):

        print()
        print("=" * 70)
        print("REFUSAL")
        print("=" * 70)

        print(
            "I couldn't find enough information in the "
            "indexed UTI guideline to answer this confidently. "
            "Please rephrase the question or consult a "
            "clinician directly."
        )

        return

    # --------------------------------------------------------
    # Grounded generation
    # --------------------------------------------------------

    result = generate_grounded_response(
        query,
        results
    )

    print()
    print("=" * 70)
    print("GROUNDED RESPONSE")
    print("=" * 70)

    print(
        result["response"]
    )


# ============================================================
# 8) ADVERSARIAL / REFUSAL TEST
# ============================================================

REFUSAL_TEST = (
    "What is the recommended treatment for migraine?"
)


# ============================================================
# 9) MAIN
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("DAY 3 - GROUNDED GENERATION & CITATION")
    print("=" * 70)

    print()
    print("Grounding prompt loaded.")
    print("Refusal logic loaded.")
    print("Citation structure loaded.")

    print()
    print("Type 'exit' to quit.")
    print("Type 'refusal_test' to run the saved refusal test.")

    while True:

        question = input(
            "\nQuestion: "
        ).strip()

        if question.lower() == "exit":

            print("Goodbye.")
            break

        if question.lower() == "refusal_test":

            run_pipeline(
                REFUSAL_TEST
            )

            continue

        if not question:

            continue

        run_pipeline(
            question
        )