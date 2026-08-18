import json
import chromadb

# ============================================================
# LOAD MPNet EMBEDDINGS
# ============================================================

with open(
    "documents_embeddings_mpnet.json",
    "r",
    encoding="utf-8"
) as f:
    documents = json.load(f)

print("Documents:", len(documents))


# ============================================================
# CREATE SEPARATE MPNet DATABASE
# ============================================================

client = chromadb.PersistentClient(
    path="./chroma_db_mpnet"
)

collection = client.get_or_create_collection(
    name="uti_guideline_mpnet"
)

print("MPNet collection ready")


# ============================================================
# PREPARE DATA
# ============================================================

ids = []
embeddings = []
documents_text = []
metadatas = []

for item in documents:

    ids.append(
        f"{item['source_type']}_{item['source_id']}"
    )

    embeddings.append(
        item["embedding"]
    )

    documents_text.append(
        item["text"]
    )

    metadatas.append({
        "source_type": item["source_type"],
        "source_id": item["source_id"],
        "title": item["title"],
        "pages": ",".join(
            map(str, item["pages"])
        )
    })


# ============================================================
# INSERT
# ============================================================

collection.upsert(
    ids=ids,
    embeddings=embeddings,
    documents=documents_text,
    metadatas=metadatas
)


# ============================================================
# VERIFY
# ============================================================

print()
print("MPNet vector database created")
print(
    "Documents in collection:",
    collection.count()
)