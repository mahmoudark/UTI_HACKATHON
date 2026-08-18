import chromadb
from sentence_transformers import SentenceTransformer


print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding model loaded")


client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_collection(
    name="uti_guideline"
)

print("Vector database loaded")
print("Documents:", collection.count())


# ============================================================
# TEST QUESTIONS
# ============================================================

test_cases = [
    ("What is a lower urinary tract infection?",
     "recommendation_1.1.1"),

    ("What advice should be given to all people with lower UTI about managing symptoms?",
     "recommendation_1.1.2"),

    ("What should be done if symptoms do not start to improve within 48 hours or worsen at any time for women with lower UTI who are not pregnant?",
     "recommendation_1.1.3"),

    ("What should be done when microbiological results are available after a urine sample was sent for culture and susceptibility testing?",
     "recommendation_1.1.4"),

    ("What should be done for pregnant women and men with lower UTI?",
     "recommendation_1.1.5"),

    ("What should people with lower UTI be advised to use for pain?",
     "recommendation_1.3.1"),

    ("What should be considered when prescribing antibiotics for lower UTI?",
     "recommendation_1.4.1"),

    ("What antibiotics are recommended for non-pregnant women aged 16 years and over?",
     "TABLE_1"),

    ("What antibiotics are recommended for pregnant women aged 12 years and over?",
     "TABLE_2"),

    ("What antibiotics are recommended for men aged 16 years and over?",
     "TABLE_3"),

    ("What antibiotics are recommended for children and young people under 16 years?",
     "TABLE_4"),
]


# ============================================================
# GET TOP 5 WITH COSINE SIMILARITY
# ============================================================

def retrieve(question):

    embedding = model.encode(
        [question],
        normalize_embeddings=True
    )[0].tolist()

    results = collection.query(
        query_embeddings=[embedding],
        n_results=5
    )

    ids = results["ids"][0]
    distances = results["distances"][0]

    # Chroma distance is converted to similarity
    similarities = [
        1 - distance
        for distance in distances
    ]

    return ids, similarities


# ============================================================
# EVALUATE THRESHOLD
# ============================================================

def evaluate_threshold(threshold):

    correct_questions = 0
    total_returned = 0
    total_relevant = 0

    for question, expected in test_cases:

        ids, similarities = retrieve(question)

        selected = []

        for doc_id, similarity in zip(ids, similarities):

            if similarity >= threshold:
                selected.append(doc_id)

        total_returned += len(selected)

        if expected in selected:
            correct_questions += 1
            total_relevant += 1

    recall = correct_questions / len(test_cases)

    if total_returned > 0:
        precision = total_relevant / total_returned
    else:
        precision = 0

    avg_returned = total_returned / len(test_cases)

    return recall, precision, avg_returned


# ============================================================
# TEST THRESHOLDS
# ============================================================

print()
print("=" * 75)
print("COSINE SIMILARITY THRESHOLD EVALUATION")
print("=" * 75)

print()
print("Top-K fixed at 5")
print("Testing different similarity thresholds...")
print()

thresholds = [
    0.00,
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50
]

best_threshold = None
best_score = -1

for threshold in thresholds:

    recall, precision, avg_returned = evaluate_threshold(
        threshold
    )

    # Balanced score
    balanced_score = (
        recall + precision
    ) / 2

    print(
        f"Threshold = {threshold:.2f} | "
        f"Recall = {recall:.4f} | "
        f"Precision = {precision:.4f} | "
        f"Avg returned = {avg_returned:.2f} | "
        f"Balanced = {balanced_score:.4f}"
    )

    if balanced_score > best_score:

        best_score = balanced_score
        best_threshold = threshold


# ============================================================
# FINAL RESULT
# ============================================================

print()
print("=" * 75)
print("BEST COSINE SIMILARITY THRESHOLD")
print("=" * 75)

print(
    "Best Threshold:",
    best_threshold
)

print(
    "Balanced Score:",
    round(best_score, 4)
)

print()
print("Threshold evaluation completed.")