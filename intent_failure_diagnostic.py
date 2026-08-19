import re
import chromadb
from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-mpnet-base-v2"

DATABASE_PATH = "./chroma_db_mpnet"
COLLECTION_NAME = "uti_guideline_mpnet"

TOP_K = 10

LEXICAL_WEIGHT = 0.10
INTENT_WEIGHT = 0.10


QUESTIONS = [
    "What should be considered when prescribing antibiotics for lower UTI?",
    "What factors should be considered when choosing antibiotics for lower UTI?",
    "What should clinicians consider when selecting an antibiotic for lower UTI?",
    "What factors should guide antibiotic prescribing for lower UTI?",
    "What should be taken into account when prescribing antibiotics for lower UTI?",
    "What considerations are important when prescribing antibiotics for lower UTI?",
    "What factors should be considered before prescribing antibiotic treatment for lower UTI?",
    "What should be considered when choosing an antibiotic for a lower UTI?",
    "What factors influence the choice of antibiotic for lower UTI?",
    "How should antibiotic prescribing decisions be made for lower UTI?"
]


STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "for",
    "with", "what", "which", "how", "should", "be",
    "is", "are", "was", "were", "do", "does", "did",
    "when", "where", "who", "that", "this", "these",
    "those", "in", "on", "at", "from", "by", "about",
    "all", "people", "person", "patients", "patient"
}


def tokenize(text):

    return {
        word
        for word in re.findall(
            r"[a-zA-Z0-9]+",
            text.lower()
        )
        if len(word) >= 3
        and word not in STOPWORDS
    }


def lexical_overlap(query, document):

    q = tokenize(query)
    d = tokenize(document)

    if not q or not d:
        return 0.0

    overlap = len(q.intersection(d))

    precision = overlap / len(q)
    recall = overlap / len(d)

    if precision + recall == 0:
        return 0.0

    return (
        2 * precision * recall
        / (precision + recall)
    )


def detect_intent(query):

    q = query.lower()

    antibiotic_terms = [
        "antibiotic",
        "antibiotics",
        "antimicrobial",
        "antimicrobials"
    ]

    selection_terms = [
        "prescrib",
        "choos",
        "select",
        "choice",
        "choosing",
        "selecting",
        "selection",
        "decision",
        "decisions",
        "treatment choice"
    ]

    consideration_terms = [
        "consider",
        "considered",
        "consideration",
        "considerations",
        "factor",
        "factors",
        "guide",
        "guiding",
        "influence",
        "influences",
        "take into account",
        "taken into account",
        "account",
        "important",
        "before prescribing"
    ]

    has_antibiotic = any(
        x in q
        for x in antibiotic_terms
    )

    has_selection = any(
        x in q
        for x in selection_terms
    )

    has_consideration = any(
        x in q
        for x in consideration_terms
    )

    if (
        has_antibiotic
        and has_selection
        and has_consideration
    ):
        return "ANTIBIOTIC_SELECTION"

    if (
        has_antibiotic
        and has_consideration
        and (
            "lower uti" in q
            or
            "lower urinary tract infection" in q
        )
    ):
        return "ANTIBIOTIC_SELECTION"

    return "GENERAL"


def get_bonus(
    query,
    metadata,
    document
):

    intent = detect_intent(query)

    source_id = str(
        metadata.get(
            "source_id",
            ""
        )
    )

    if (
        intent == "ANTIBIOTIC_SELECTION"
        and source_id == "1.4.1"
    ):
        return INTENT_WEIGHT

    return 0.0


print("Loading MPNet...")

model = SentenceTransformer(
    MODEL_NAME
)

client = chromadb.PersistentClient(
    path=DATABASE_PATH
)

collection = client.get_collection(
    name=COLLECTION_NAME
)

print("Documents:", collection.count())

print()
print("=" * 80)
print("INTENT FAILURE DIAGNOSTIC")
print("=" * 80)


for index, question in enumerate(
    QUESTIONS,
    start=1
):

    embedding = model.encode(
        [question],
        normalize_embeddings=True
    )[0].tolist()

    results = collection.query(
        query_embeddings=[embedding],
        n_results=TOP_K
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    candidates = []

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances
    ):

        similarity = 1 - distance

        lexical = lexical_overlap(
            question,
            document
        )

        hybrid = (
            similarity
            +
            LEXICAL_WEIGHT * lexical
        )

        bonus = get_bonus(
            question,
            metadata,
            document
        )

        final_score = (
            hybrid + bonus
        )

        candidates.append({
            "source_id":
                metadata["source_id"],
            "similarity":
                similarity,
            "lexical":
                lexical,
            "hybrid":
                hybrid,
            "bonus":
                bonus,
            "final":
                final_score
        })

    candidates.sort(
        key=lambda x: x["final"],
        reverse=True
    )

    rank_141 = None

    for rank, candidate in enumerate(
        candidates,
        start=1
    ):

        if candidate["source_id"] == "1.4.1":

            rank_141 = rank
            break

    status = (
        "PASS"
        if rank_141 == 1
        else "FAIL"
    )

    print()
    print("-" * 80)

    print(
        f"[{index}/10] {status}"
    )

    print(
        "Question:",
        question
    )

    print(
        "Intent:",
        detect_intent(question)
    )

    print(
        "1.4.1 rank:",
        rank_141
    )

    print()
    print(
        "TOP RESULTS"
    )

    for rank, candidate in enumerate(
        candidates[:5],
        start=1
    ):

        print(
            f"{rank}. "
            f"{candidate['source_id']} | "
            f"similarity={candidate['similarity']:.4f} | "
            f"lexical={candidate['lexical']:.4f} | "
            f"hybrid={candidate['hybrid']:.4f} | "
            f"bonus={candidate['bonus']:.4f} | "
            f"final={candidate['final']:.4f}"
        )


print()
print("=" * 80)
print("DIAGNOSTIC COMPLETED")
print("=" * 80)