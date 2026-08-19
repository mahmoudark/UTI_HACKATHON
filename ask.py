import chromadb
from sentence_transformers import SentenceTransformer
import re
from intent_classifier import detect_intent, intent_bonus

# ============================================================
# FINAL SETTINGS
# ============================================================

TOP_K = 4
SIMILARITY_THRESHOLD = 0.20
LEXICAL_WEIGHT = 0.10

MODEL_NAME = "all-mpnet-base-v2"

DATABASE_PATH = "./chroma_db_mpnet"
COLLECTION_NAME = "uti_guideline_mpnet"


# ============================================================
# 1) LOAD EMBEDDING MODEL
# ============================================================

print("Loading embedding model...")

model = SentenceTransformer(
    MODEL_NAME
)

print("Embedding model loaded")


# ============================================================
# 2) LOAD FINAL VECTOR DATABASE
# ============================================================

client = chromadb.PersistentClient(
    path=DATABASE_PATH
)

collection = client.get_collection(
    name=COLLECTION_NAME
)

print("Final vector database loaded")
print("Documents:", collection.count())


# ============================================================
# 3) TITLE / KEYWORD MATCHING
# ============================================================

def title_matches(query, metadata):

    title = metadata.get(
        "title",
        ""
    ).lower()

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

        if (
            phrase in query_lower
            and phrase in title
        ):

            matches += 4

    return matches


# ============================================================
# 4) CONSERVATIVE LEXICAL OVERLAP
# ============================================================

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "for", "with",
    "what", "which", "how", "should", "be", "is", "are", "was",
    "were", "do", "does", "did", "when", "where", "who", "that",
    "this", "these", "those", "in", "on", "at", "from", "by",
    "about", "all", "people", "person", "patients", "patient"
}


def tokenize(text):

    words = re.findall(
        r"[a-zA-Z0-9]+",
        text.lower()
    )

    return {
        word
        for word in words
        if len(word) >= 3
        and word not in STOPWORDS
    }


def lexical_overlap(query, document):

    query_tokens = tokenize(query)
    document_tokens = tokenize(document)

    if not query_tokens or not document_tokens:
        return 0.0

    overlap = len(
        query_tokens.intersection(document_tokens)
    )

    precision = overlap / len(query_tokens)
    recall = overlap / len(document_tokens)

    if precision + recall == 0:
        return 0.0

    return (
        2 * precision * recall
        / (precision + recall)
    )


# ============================================================
# 5) HYBRID SEARCH
# ============================================================

def hybrid_search(
    query,
    top_k=TOP_K
):

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )[0].tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=collection.count()
    )

    candidates = []

    for (
        document,
        metadata,
        distance
    ) in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):

        similarity = 1 - distance

        if similarity < SIMILARITY_THRESHOLD:
            continue

        matches = title_matches(
            query,
            metadata
        )

        lexical_score = lexical_overlap(
            query,
            document
        )

        # Tested and tuned on the 55-question benchmark:
        # semantic similarity + 0.10 * lexical overlap
        hybrid_score = (
    similarity
    + LEXICAL_WEIGHT * lexical_score
)

        bonus = intent_bonus(
            query,
            metadata,
            document
        )

        hybrid_score += bonus

        candidates.append({

            "document": document,

            "metadata": metadata,

            "similarity": similarity,

            "matches": matches,

            "lexical_score": lexical_score,

            "hybrid_score": hybrid_score

        })

    candidates.sort(
        key=lambda x: x["hybrid_score"],
        reverse=True
    )

    return candidates[:top_k]


# ============================================================
# 6) GENERATE ANSWER
# ============================================================

def generate_answer(
    query,
    best
):

    metadata = best["metadata"]

    text = best["document"]

    source_type = metadata.get(
        "source_type",
        ""
    )

    source_id = metadata.get(
        "source_id",
        ""
    )

    title = metadata.get(
        "title",
        ""
    )

    # --------------------------------------------------------
    # TABLE ANSWERS
    # --------------------------------------------------------

    if source_type == "table":

        if source_id == "TABLE_1":

            return f"""
According to {title}:

First choices:
- Nitrofurantoin
- Trimethoprim

Second choices, if the first choice is not suitable or
there is no improvement after at least 48 hours:
- Nitrofurantoin
- Pivmecillinam
- Fosfomycin

The exact dosage and duration are given in the source table.
"""

        if source_id == "TABLE_2":

            return f"""
According to {title}:

First choice:
- Nitrofurantoin

Second choices:
- Amoxicillin, only if culture results are available and susceptible
- Cefalexin

Alternative second choices should be based on culture and
susceptibility results.

For asymptomatic bacteriuria, choose from nitrofurantoin,
amoxicillin or cefalexin based on recent culture and
susceptibility results.
"""

        if source_id == "TABLE_3":

            return f"""
According to {title}:

First choices:
- Trimethoprim
- Nitrofurantoin, when eGFR is 45 ml/minute or more

Nitrofurantoin is not recommended for men with suspected
prostate involvement because it is unlikely to reach
therapeutic levels in the prostate.

If there is no improvement after at least 48 hours, or the
first choice is not suitable, consider alternative diagnoses
and base antibiotic choice on recent culture and
susceptibility results.
"""

        if source_id == "TABLE_4":

            return f"""
According to {title}:

Children under 3 months:
- Refer to a paediatric specialist and treat with intravenous
  antibiotics according to the relevant NICE guideline.

For children aged 3 months and over, first choices include:
- Trimethoprim, when there is a low risk of resistance
- Nitrofurantoin, when eGFR is 45 ml/minute or more

Second choices include:
- Nitrofurantoin
- Amoxicillin, when culture results are available and susceptible
- Cefalexin

The exact dosage and duration depend on age and are given
in the source table.
"""

    # --------------------------------------------------------
    # RECOMMENDATION ANSWER
    # --------------------------------------------------------

    return f"""
According to recommendation {source_id}:

{text}
"""


# ============================================================
# 7) PRINT ANSWER
# ============================================================

def answer_question(query):

    results = hybrid_search(
        query,
        top_k=TOP_K
    )

    if not results:

        print()
        print("No evidence found.")
        return

    best = results[0]

    metadata = best["metadata"]

    print()
    print("=" * 60)
    print("QUESTION")
    print("=" * 60)

    print(query)

    print()
    print("=" * 60)
    print("ANSWER")
    print("=" * 60)

    answer = generate_answer(
        query,
        best
    )

    print(answer)

    print("=" * 60)
    print("RETRIEVAL SETTINGS")
    print("=" * 60)

    print(
        "Embedding Model:",
        MODEL_NAME
    )

    print(
        "Top-K:",
        TOP_K
    )

    print(
        "Cosine Similarity Threshold:",
        SIMILARITY_THRESHOLD
    )

    print(
        "Lexical Weight:",
        LEXICAL_WEIGHT
    )

    print(
        "Selected chunks:",
        len(results)
    )

    print()
    print("=" * 60)
    print("SOURCE")
    print("=" * 60)

    print(
        "Source ID:",
        metadata.get("source_id")
    )

    print(
        "Source type:",
        metadata.get("source_type")
    )

    print(
        "Title:",
        metadata.get("title")
    )

    print(
        "Page(s):",
        metadata.get("pages")
    )

    print(
        "Similarity:",
        round(
            best["similarity"],
            4
        )
    )

    print(
        "Lexical overlap:",
        round(
            best["lexical_score"],
            4
        )
    )

    print(
        "Hybrid score:",
        round(
            best["hybrid_score"],
            4
        )
    )

    print()
    print("=" * 60)
    print("EVIDENCE")
    print("=" * 60)

    print(
        best["document"]
    )

    print()
    print("=" * 60)
    print("OTHER RETRIEVED SOURCES")
    print("=" * 60)

    for i, result in enumerate(
        results[1:],
        start=2
    ):

        meta = result["metadata"]

        print(
            f"{i}. "
            f"{meta.get('source_id')}"
            f" | similarity: "
            f"{result['similarity']:.4f}"
            f" | lexical: "
            f"{result['lexical_score']:.4f}"
            f" | hybrid: "
            f"{result['hybrid_score']:.4f}"
        )


# ============================================================
# 8) MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("UTI CLINICAL DECISION SUPPORT - HYBRID V2")
    print("=" * 60)

    print(
        "Embedding:",
        MODEL_NAME
    )

    print(
        "Top-K:",
        TOP_K
    )

    print(
        "Similarity Threshold:",
        SIMILARITY_THRESHOLD
    )

    print(
        "Lexical Weight:",
        LEXICAL_WEIGHT
    )

    print(
        "Type 'exit' to quit."
    )

    print()

    while True:

        question = input(
            "Question: "
        ).strip()

        if question.lower() == "exit":

            print(
                "Goodbye."
            )

            break

        if not question:

            continue

        answer_question(
            question
        )
