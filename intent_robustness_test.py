from intent_classifier import (
    detect_intent,
    intent_bonus,
)
import chromadb
from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-mpnet-base-v2"

client = chromadb.PersistentClient(
    path="./chroma_db_mpnet"
)

collection = client.get_collection(
    name="uti_guideline_mpnet"
)

model = SentenceTransformer(
    MODEL_NAME
)


TEST_CASES = [
    "What should be considered when prescribing antibiotics for lower UTI?",

    "What factors should be considered when choosing antibiotics for lower UTI?",

    "What should clinicians consider when selecting an antibiotic for lower UTI?",

    "What factors should guide antibiotic prescribing for lower UTI?",

    "What should be taken into account when prescribing antibiotics for lower UTI?",

    "What considerations are important when prescribing antibiotics for lower UTI?",

    "What factors should be considered before prescribing antibiotic treatment for lower UTI?",

    "What should be considered when choosing an antibiotic for a lower UTI?",

    "What factors influence the choice of antibiotic for lower UTI?",

    "How should antibiotic prescribing decisions be made for lower UTI?",
]


print()
print("=" * 75)
print("INTENT ROBUSTNESS TEST")
print("=" * 75)

print()
print("Target source: 1.4.1")
print()

passed = 0

for index, question in enumerate(
    TEST_CASES,
    start=1
):

    intent = detect_intent(question)

    embedding = model.encode(
        [question],
        normalize_embeddings=True
    )[0].tolist()

    results = collection.query(
        query_embeddings=[embedding],
        n_results=10
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

        bonus = intent_bonus(
            question,
            metadata,
            document
        )

        score = similarity + bonus

        candidates.append(
            (
                score,
                metadata["source_id"],
                similarity,
                bonus
            )
        )

    candidates.sort(
        reverse=True
    )

    top5 = candidates[:5]

    target_rank = None

    for rank, candidate in enumerate(
        top5,
        start=1
    ):

        if candidate[1] == "1.4.1":
            target_rank = rank
            break

    success = (
        intent == "ANTIBIOTIC_SELECTION"
        and target_rank == 1
    )

    if success:
        passed += 1

    print(
        f"[{index}/{len(TEST_CASES)}] "
        f"{'PASS' if success else 'FAIL'}"
    )

    print(
        "Question:",
        question
    )

    print(
        "Intent:",
        intent
    )

    print(
        "1.4.1 rank:",
        target_rank
    )

    print(
        "Top-5:",
        " | ".join(
            candidate[1]
            for candidate in top5
        )
    )

    print()


print("=" * 75)
print("ROBUSTNESS RESULTS")
print("=" * 75)

print()

print(
    f"Passed: {passed}/{len(TEST_CASES)}"
)

print(
    f"Success rate: "
    f"{passed / len(TEST_CASES):.2%}"
)

print()

if passed == len(TEST_CASES):

    print(
        "FINAL ROBUSTNESS: PASS"
    )

else:

    print(
        "FINAL ROBUSTNESS: FAIL"
    )

print()
print("Testing completed.")
