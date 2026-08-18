import json
from sentence_transformers import SentenceTransformer

# ============================================================
# LOAD DATASET
# ============================================================

with open("documents.json", "r", encoding="utf-8") as f:
    documents = json.load(f)

print("Documents:", len(documents))


# ============================================================
# LOAD MPNet MODEL
# ============================================================

print("Loading MPNet embedding model...")

model = SentenceTransformer(
    "all-mpnet-base-v2"
)

print("MPNet model loaded")


# ============================================================
# CREATE EMBEDDINGS
# ============================================================

texts = [
    document["text"]
    for document in documents
]

print("Creating MPNet embeddings...")

embeddings = model.encode(
    texts,
    show_progress_bar=True,
    normalize_embeddings=True
)

print("Embeddings created")


# ============================================================
# ADD EMBEDDINGS
# ============================================================

for document, embedding in zip(
    documents,
    embeddings
):

    document["embedding"] = embedding.tolist()


# ============================================================
# SAVE
# ============================================================

with open(
    "documents_embeddings_mpnet.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        documents,
        f,
        ensure_ascii=False,
        indent=2
    )


print()
print("Created: documents_embeddings_mpnet.json")
print("Documents:", len(documents))
print("Embedding dimension:", len(embeddings[0]))