import chromadb
import answer_engine_v3 as e
import ask

client = chromadb.PersistentClient(path="./chroma_db_mpnet")
collection = client.get_collection("uti_guideline_mpnet")

def cosine_search(query, top_k=4):
    emb = ask.model.encode(
        [query],
        normalize_embeddings=True
    )[0].tolist()

    r = collection.query(
        query_embeddings=[emb],
        n_results=collection.count()
    )

    candidates = []

    for document, metadata, distance in zip(
        r["documents"][0],
        r["metadatas"][0],
        r["distances"][0]
    ):
        similarity = 1 - (distance / 2)

        if similarity < e.SIMILARITY_THRESHOLD:
            continue

        candidates.append({
            "document": document,
            "metadata": metadata,
            "similarity": similarity,
            "lexical_score": 0.0,
            "hybrid_score": similarity,
        })

    candidates.sort(
        key=lambda x: x["similarity"],
        reverse=True
    )

    return candidates[:top_k]

e.hybrid_search = cosine_search
e.SIMILARITY_THRESHOLD = 0.60
e.TOP_K = 4

questions = [
    "What antibiotics are recommended for non-pregnant women aged 16 years and over?",
    "What antibiotics are recommended for pregnant women aged 12 years and over?",
    "What antibiotics are recommended for men aged 16 years and over?",
    "What is diabetes mellitus?",
]

for q in questions:
    r = e.answer_with_guard(q)

    print()
    print("QUESTION:", q)
    print("STATUS:", r.get("status"))
    print("CONFIDENCE:", r.get("confidence"))
    print("SOURCE:", (r.get("source") or {}).get("source_id"))
    print("RANK:", r.get("rank"))
    print("REASON:", r.get("reason"))

