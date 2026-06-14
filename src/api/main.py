import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import time
import numpy as np

app = FastAPI(title="Medical QA API", version="1.0")

USE_GROQ = os.environ.get("USE_GROQ", "false").lower() == "true"

# Load models at startup
print("Loading models...")
if USE_GROQ:
    print("Using Groq for inference. Loading retriever only...")
    from src.rag.retriever import load_retriever, load_reranker, retrieve_and_rerank
    from src.inference.groq_client import answer_with_groq
    embedder, collection = load_retriever()
    reranker = load_reranker()
    model, tokenizer = None, None
else:
    print("Using local model for inference. Loading all models...")
    from src.rag.pipeline import load_all, answer_question
    model, tokenizer, embedder, collection, reranker = load_all()
print("Ready!")

class QuestionRequest(BaseModel):
    question: str

class QuestionResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: str
    latency_seconds: float

def estimate_confidence(question, sources, embedder):
    """Estimate confidence based on how well the sources match the question.
    Uses cosine similarity between the question and retrieved chunks."""
    if not sources:
        return "low"
    query_emb  = embedder.encode([question])
    source_emb = embedder.encode(sources)
    similarities = np.dot(source_emb, query_emb.T).flatten()
    avg_similarity = float(np.mean(similarities))
    if avg_similarity > 0.5:
        return "high"
    elif avg_similarity > 0.3:
        return "medium"
    return "low"

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/ask", response_model=QuestionResponse)
def ask(request: QuestionRequest):
    if len(request.question.strip()) < 5:
        raise HTTPException(status_code=400, detail="Question too short")

    start  = time.time()
    
    if USE_GROQ:
        chunks = retrieve_and_rerank(request.question, collection, embedder, reranker)
        answer = answer_with_groq(request.question, chunks)
        sources = chunks
    else:
        result = answer_question(request.question, model, tokenizer, embedder, collection, reranker)
        answer = result['answer']
        sources = result['sources']
        
    latency = round(time.time() - start, 2)

    confidence = estimate_confidence(request.question, sources, embedder)

    return QuestionResponse(
        answer=answer,
        sources=sources,
        confidence=confidence,
        latency_seconds=latency
    )
