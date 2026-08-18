import chromadb
from sentence_transformers import SentenceTransformer


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
# 3) DYNAMIC RETRIEVAL
# ============================================================

def dynamic_search(query, max_k=5, min_similarity=0.20):

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )[0].tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=max_k
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

        # Chroma cosine distance
        similarity = 1 - distance

        candidates.append({
            "document": document,
            "metadata": metadata,
            "distance": distance,
            "similarity": similarity
        })


    # ========================================================
    # DYNAMIC SELECTION
    # ========================================================

    selected = []

    for i, candidate in enumerate(candidates):

        similarity = candidate["similarity"]

        # Always keep the best result
        if i == 0:
            selected.append(candidate)
            continue

        # Remove clearly weak results
        if similarity < min_similarity:
            continue

        # Compare each result with the previous one
        previous_similarity = candidates[i - 1]["similarity"]

        gap = previous_similarity - similarity

        # If there is a large drop, stop retrieving
        if gap > 0.15:
            break

        selected.append(candidate)


    return selected, candidates


# ============================================================
# 4) TEST
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("DYNAMIC RETRIEVAL TEST")
    print("=" * 70)

    while True:

        query = input("\nQuestion: ").strip()

        if query.lower() == "exit":
            print("Goodbye.")
            break

        if not query:
            continue

        selected, all_results = dynamic_search(query)

        print()
        print("=" * 70)
        print("SIMILARITY RESULTS")
        print("=" * 70)

        for i, result in enumerate(all_results, 1):

            metadata = result["metadata"]

            print(
                f"{i}. "
                f"{metadata.get('source_id')} | "
                f"Similarity: {result['similarity']:.4f}"
            )


        print()
        print("=" * 70)
        print("SELECTED CHUNKS")
        print("=" * 70)

        print(
            "Number of selected chunks:",
            len(selected)
        )

        for i, result in enumerate(selected, 1):

            metadata = result["metadata"]

            print()
            print(f"--- CHUNK {i} ---")

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
                round(result["similarity"], 4)
            )