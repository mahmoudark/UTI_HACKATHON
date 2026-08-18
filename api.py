import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from answer_engine_v3 import answer_with_guard
from ask import (
    MODEL_NAME,
    TOP_K,
    SIMILARITY_THRESHOLD,
    LEXICAL_WEIGHT,
    collection,
)


app = FastAPI(
    title="UTI Clinical Decision Support API",
    description="Thin API bridge over the existing Python retrieval and guard engine.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QuestionRequest(BaseModel):
    query: str


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "top_k": TOP_K,
        "similarity_threshold": SIMILARITY_THRESHOLD,
        "lexical_weight": LEXICAL_WEIGHT,
        "indexed_documents": collection.count(),
    }


@app.post("/api/answer")
def answer(req: QuestionRequest):
    result = answer_with_guard(req.query.strip())
    return result


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
