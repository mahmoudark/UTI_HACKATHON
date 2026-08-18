from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from answer_engine_v3 import (
    answer_with_guard,
    population_compatible,
)

from ask import (
    MODEL_NAME,
    TOP_K,
    SIMILARITY_THRESHOLD,
    LEXICAL_WEIGHT,
    collection,
)


app = FastAPI(
    title="UTI Clinical Decision Support API",
    version="1.0.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# REQUEST MODEL
# ============================================================

class AnswerRequest(BaseModel):
    query: str


# ============================================================
# FILTER DISPLAYED EVIDENCE
# ============================================================

def filter_display_results(query, results):
    """
    Keep only evidence compatible with the requested population
    before sending retrieval details to the frontend.

    This does NOT change the clinical backend logic.
    It only controls which retrieved evidence is displayed.
    """

    if not results:
        return []

    filtered = []

    for item in results:
        if population_compatible(query, item):
            filtered.append(item)

    return filtered


# ============================================================
# HEALTH
# ============================================================

@app.get("/api/health")
def health():

    return {
        "status": "ok",
        "model": MODEL_NAME,
        "top_k": TOP_K,
        "similarity_threshold": SIMILARITY_THRESHOLD,
        "lexical_weight": LEXICAL_WEIGHT,
        "documents": collection.count(),
    }


# ============================================================
# ANSWER
# ============================================================

@app.post("/api/answer")
def answer(request: AnswerRequest):

    query = request.query.strip()

    if not query:

        return {
            "status": "REFUSED",
            "reason": "Question is empty.",
            "answer": None,
            "source": None,
            "rank": None,
            "confidence": "INSUFFICIENT",
            "grounding": None,
            "results": [],
        }

    # --------------------------------------------------------
    # Existing clinical engine
    # --------------------------------------------------------

    result = answer_with_guard(query)

    # --------------------------------------------------------
    # Filter ONLY the evidence displayed to the frontend
    # --------------------------------------------------------

    result["results"] = filter_display_results(
        query,
        result.get("results", [])
    )

    return result