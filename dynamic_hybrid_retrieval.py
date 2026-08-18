import chromadb
from sentence_transformers import SentenceTransformer
import re


# ============================================================
# 1) LOAD MODEL
# ============================================================

print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding model loaded")


# ============================================================
# 2) LOAD CHROMA
# ============================================================

client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_collection(
    name="uti_guideline"
)

print("Vector database loaded")
print("Documents:", collection.count())


# ============================================================
# 3) TITLE MATCHING
# ============================================================

def title_matches(query, metadata):

    title = metadata.get("title", "").lower()

    query_lower = query.lower()

    query_words = re.findall(
        r"[a-zA-Z]+",
        query_lower
    )

    matches = 0

    for word in query_words:

        if len(word) < 3:
            continue

        if word in title:
            matches += 1


    # Strong matching for patient groups
    if "non-pregnant" in query_lower:
        if "non-pregnant" in title:
            matches += 3

    if "pregnant" in query_lower:
        if "pregnant" in title:
            matches += 3

    if "men" in query_lower:
        if "men" in title:
            matches += 3

    if "children" in query_lower:
        if "children" in title:
            matches += 3

    if "under 16" in query_lower:
        if "under 16" in title:
            matches += 2

    if "16 years and over" in query_lower:
        if "16 years and over" in title:
            matches += 2

    return matches


# ============================================================
# 4) HYBRID RETRIEVAL
# ============================================================

def hybrid_search(query, max_k=10):

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )[0].tolist()


    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=collection.count()
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

        matches = title_matches(
            query,
            metadata
        )


        # Hybrid score
        hybrid_score = (
            similarity +
            (matches * 0.25)
        )


        candidates.append({
            "document": document,
            "metadata": metadata,
            "distance": distance,
            "similarity": similarity,
            "matches": matches,
            "hybrid_score": hybrid_score
        })


    # Sort by hybrid score
    candidates.sort(
        key=lambda x: x["hybrid_score"],
        reverse=True
    )


    return candidates[:max_k]


# ============================================================
# 5) DYNAMIC CHUNK SELECTION
# ============================================================

def dynamic_select(results):

    if not results:
        return []


    selected = []


    for i, result in enumerate(results):

        # Always keep the best result
        if i == 0:

            selected.append(result)

            continue


        previous = results[i - 1]


        current_score = result["hybrid_score"]
        previous_score = previous["hybrid_score"]


        gap = previous_score - current_score


        # Keep results with strong title matching
        if result["matches"] > 0:

            selected.append(result)

            continue


        # Stop when there is a large score drop
        if gap > 0.50:

            break


        selected.append(result)


    return selected


# ============================================================
# 6) TEST
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("DYNAMIC HYBRID RETRIEVAL")
    print("=" * 70)


    while True:

        query = input("\nQuestion: ").strip()


        if query.lower() == "exit":

            print("Goodbye.")

            break


        if not query:

            continue


        results = hybrid_search(
            query,
            max_k=10
        )


        selected = dynamic_select(
            results
        )


        # ====================================================
        # SHOW RANKING
        # ====================================================

        print()
        print("=" * 70)
        print("HYBRID RANKING")
        print("=" * 70)


        for i, result in enumerate(
            results,
            1
        ):

            metadata = result["metadata"]


            print(
                f"{i}. "
                f"{metadata.get('source_id')} | "
                f"Similarity: {result['similarity']:.4f} | "
                f"Title matches: {result['matches']} | "
                f"Hybrid: {result['hybrid_score']:.4f}"
            )


        # ====================================================
        # SHOW SELECTED
        # ====================================================

        print()
        print("=" * 70)
        print("DYNAMICALLY SELECTED CHUNKS")
        print("=" * 70)


        print(
            "Number of selected chunks:",
            len(selected)
        )


        for i, result in enumerate(
            selected,
            1
        ):

            metadata = result["metadata"]


            print()
            print(
                f"--- CHUNK {i} ---"
            )


            print(
                "ID:",
                metadata.get("source_id")
            )


            print(
                "Title:",
                metadata.get("title")
            )


            print(
                "Similarity:",
                round(
                    result["similarity"],
                    4
                )
            )


            print(
                "Title matches:",
                result["matches"]
            )


            print(
                "Hybrid score:",
                round(
                    result["hybrid_score"],
                    4
                )
            )