import chromadb
import torch

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# ============================================================
# SETTINGS
# ============================================================

EMBEDDING_MODEL = "all-mpnet-base-v2"
LLM_MODEL = "google/flan-t5-small"

DATABASE_PATH = "./chroma_db_mpnet"
COLLECTION_NAME = "uti_guideline_mpnet"

TOP_K = 5
MIN_SIMILARITY = 0.25


# ============================================================
# 1) LOAD EMBEDDING MODEL
# ============================================================

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)

print("Embedding model loaded")


# ============================================================
# 2) LOAD LOCAL LLM
# ============================================================

print("Loading local LLM...")

tokenizer = AutoTokenizer.from_pretrained(
    LLM_MODEL
)

llm = AutoModelForSeq2SeqLM.from_pretrained(
    LLM_MODEL
)

print("Local LLM loaded")


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
# 4) RETRIEVE EVIDENCE
# ============================================================

def retrieve(query):

    embedding = embedding_model.encode(
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

    return results[0]["similarity"] < MIN_SIMILARITY


# ============================================================
# 6) LLM GROUNDED GENERATION
# ============================================================

def generate_with_llm(query, evidence):

    context_parts = []

    for item in evidence[:3]:

        metadata = item["metadata"]

        source_id = metadata.get(
            "source_id",
            "Unknown"
        )

        pages = metadata.get(
            "pages",
            "Unknown"
        )

        text = item["text"]

        context_parts.append(
            f"SOURCE: {source_id}\n"
            f"PAGES: {pages}\n"
            f"EVIDENCE:\n{text}"
        )

    context = "\n\n".join(
        context_parts
    )

    prompt = f"""
You are a clinical evidence assistant.

Answer the user's question ONLY using the evidence below.

Do not use outside medical knowledge.
Do not invent information.
Do not invent dosage, duration, thresholds, or recommendations.
If the evidence does not answer the question, say:
INSUFFICIENT EVIDENCE

User question:
{query}

Evidence:
{context}

Preserve every important recommendation, antibiotic name,
condition, dosage, duration, threshold, and exception stated
in the evidence.

Do not omit important items from the evidence.

If multiple treatment choices are listed, include all relevant
choices.

Answer using only the evidence provided.
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    )

    with torch.no_grad():

        output = llm.generate(
            **inputs,
            max_new_tokens=180,
            do_sample=False
        )

    answer = tokenizer.decode(
        output[0],
        skip_special_tokens=True
    )

    return answer.strip()


# ============================================================
# 7) FULL PIPELINE
# ============================================================

def run_pipeline(query):

    print()
    print("=" * 70)
    print("QUERY")
    print("=" * 70)

    print(query)

    # --------------------------------------------------------
    # RETRIEVE
    # --------------------------------------------------------

    print()
    print("Retrieving evidence...")

    results = retrieve(query)

    print(
        "Retrieved:",
        len(results)
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
    # REFUSAL
    # --------------------------------------------------------

    if should_refuse(results):

        print()
        print("=" * 70)
        print("REFUSAL")
        print("=" * 70)

        print(
            "I couldn't find enough information in the indexed "
            "UTI guideline to answer this confidently. "
            "Please rephrase the question or consult a "
            "clinician directly."
        )

        return

    # --------------------------------------------------------
    # GENERATE
    # --------------------------------------------------------

    answer = generate_with_llm(
        query,
        results
    )

    # --------------------------------------------------------
    # CHECK LLM REFUSAL
    # --------------------------------------------------------

    if "INSUFFICIENT EVIDENCE" in answer.upper():

        print()
        print("=" * 70)
        print("REFUSAL")
        print("=" * 70)

        print(
            "I couldn't find enough information in the indexed "
            "UTI guideline to answer this confidently."
        )

        return

    # --------------------------------------------------------
    # BEST SOURCE
    # --------------------------------------------------------

    best = results[0]

    metadata = best["metadata"]

    source_id = metadata.get(
        "source_id",
        "Unknown"
    )

    pages = metadata.get(
        "pages",
        "Unknown"
    )

    title = metadata.get(
        "title",
        "UTI guideline"
    )

    # --------------------------------------------------------
    # FINAL STRUCTURED RESPONSE
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)

    print(answer)

    print()
    print("=" * 70)
    print("EXCERPT")
    print("=" * 70)

    print(
        best["text"]
    )

    print()
    print("=" * 70)
    print("CITATION")
    print("=" * 70)

    print(
        f"[{title}, Section {source_id}, Page {pages}]"
    )

    print()
    print("=" * 70)
    print("PIPELINE")
    print("=" * 70)

    print(
        "Query"
        " -> Retrieve"
        " -> Grounded Prompt"
        " -> Local LLM"
        " -> Citation"
    )


# ============================================================
# 8) MAIN
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("DAY 3 - GROUNDED LLM PIPELINE")
    print("=" * 70)

    print(
        "Embedding:",
        EMBEDDING_MODEL
    )

    print(
        "LLM:",
        LLM_MODEL
    )

    print(
        "Top-K:",
        TOP_K
    )

    print(
        "Threshold:",
        MIN_SIMILARITY
    )

    print()
    print("Type 'exit' to quit.")

    while True:

        question = input(
            "\nQuestion: "
        ).strip()

        if question.lower() == "exit":

            print("Goodbye.")
            break

        if not question:

            continue

        run_pipeline(
            question
        )