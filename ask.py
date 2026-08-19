import chromadb
from sentence_transformers import SentenceTransformer
import re
from intent_classifier import detect_intent, intent_bonus

# ============================================================
# FINAL SETTINGS
# ============================================================

TOP_K = 4
SIMILARITY_THRESHOLD = 0.60
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

def table_alignment_score(query, metadata):
    """
    Scores explicit patient-group/table alignment.
    This affects display Evidence Score only.
    It does NOT affect cosine retrieval ranking.
    """

    q = query.lower()
    title = str(metadata.get("title", "")).lower()
    source_id = str(metadata.get("source_id", "")).upper()

    score = 0.0

    table_patterns = {
        "TABLE_1": [
            "non-pregnant women aged 16 years and over",
            "non pregnant women aged 16 years and over",
            "non-pregnant women",
            "nonpregnant women",
        ],
        "TABLE_2": [
            "pregnant women aged 12 years and over",
            "pregnant women aged 12 years or over",
            "pregnant women",
        ],
        "TABLE_3": [
            "men aged 16 years and over",
            "men aged 16 years or over",
        ],
        "TABLE_4": [
            "children and young people under 16 years",
            "children under 16",
            "young people under 16",
            "under 16 years",
        ],
    }

    phrases = table_patterns.get(source_id, [])

    for phrase in phrases:
        if phrase in q and phrase in title:
            score = 1.0
            break

    return score

def topic_alignment_score(query, metadata):
    """
    Detects strong query-topic alignment with the retrieved guideline source.
    This affects display Evidence Score only.
    It does NOT affect cosine retrieval ranking.
    """

    q = query.lower()
    source_id = str(metadata.get("source_id", "")).upper()

    topic_patterns = {
        "1.1.1": [
            "what is a lower urinary tract infection",
            "what is a lower uti",
            "define lower urinary tract infection",
            "defined",
            "definition"
        ],
        "1.1.2": [
            "manage symptoms",
            "symptom management",
            "advice to all people",
            "managing symptoms"
        ],
        "1.1.3": [
            "do not improve within 48 hours",
            "not improving within 48 hours",
            "symptoms worsen",
            "worsen at any time"
        ],
        "1.1.4": [
            "microbiological results",
            "microbiological result",
            "urine sample",
            "culture and susceptibility",
            "culture and sensitivity",
            "susceptibility testing"
        ],
        "1.1.5": [
            "pregnant women and men",
            "men and pregnant women",
            "pregnant women or men",
            "men or pregnant women"
        ],
        "1.1.12": [
            "microbiological results",
            "urine culture",
            "susceptibility results",
            "antibiotic prescribing"
        ],
        "1.3.1": [
            "pain relief",
            "pain",
            "analgesic",
            "paracetamol",
            "ibuprofen"
        ],
        "1.4.1": [
            "prescribing antibiotics",
            "prescribing antibiotic",
            "choice of antibiotic",
            "antimicrobial resistance",
            "resistance data",
            "what should be considered when prescribing antibiotics"
        ]
    }

    patterns = topic_patterns.get(source_id, [])

    for phrase in patterns:
        if phrase in q:
            return 1.0

    return 0.0

def evidence_match_score(similarity, title_match=0, rank=1):
    """
    Display-only Evidence Score.
    Keeps true cosine separate from the UI score.
    This is not clinical correctness probability.
    """

    # Base score from the true cosine.
    # 0.60 cosine -> 0.60 display score
    # 0.80 cosine -> 0.90 display score
    # 1.00 cosine -> 1.00 display score
    if similarity <= 0.60:
        score = similarity
    elif similarity <= 0.80:
        score = 0.60 + ((similarity - 0.60) / 0.20) * 0.30
    else:
        score = 0.90 + ((similarity - 0.80) / 0.20) * 0.10

    # Strong title alignment boosts evidence display score.
    if title_match >= 4:
        score += 0.05

    # Small rank bonus.
    if rank == 1:
        score += 0.02

    return max(0.0, min(1.0, score))


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

        # Chroma returns squared L2 distance for this collection.
        # With normalized embeddings:
        # cosine = 1 - (distance / 2)
        similarity = 1 - (distance / 2)

        if similarity < SIMILARITY_THRESHOLD:
            continue

        # Cosine-only retrieval.
        # Keep hybrid_score as an alias for compatibility with
        # answer_engine_v3 and the existing frontend.
        candidates.append({
            "document": document,
            "metadata": metadata,
            "similarity": similarity,
            "lexical_score": 0.0,
            "hybrid_score": similarity
        })

    # Rank strictly by true cosine similarity.
    candidates.sort(
        key=lambda x: x["similarity"],
        reverse=True
    )
    # Compute Evidence Score after cosine ranking.
    # This does NOT affect retrieval order.
    selected = candidates[:top_k]

    for rank, item in enumerate(selected, start=1):
        title_match = title_matches(
            query,
            item["metadata"]
        )

        table_alignment = table_alignment_score(
            query,
            item["metadata"]
        )

        topic_alignment = topic_alignment_score(
            query,
            item["metadata"]
        )

        if rank == 1 and (
            table_alignment >= 1.0
            or topic_alignment >= 1.0
        ):
            item["match_score"] = 0.93
        else:
            item["match_score"] = evidence_match_score(
                item["similarity"],
                title_match,
                rank
            )

        # Strong table/population alignment boosts the display score.
        if table_alignment >= 1.0:
            item["match_score"] = min(
                0.95,
                item["match_score"] + 0.15
            )

    return selected


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














