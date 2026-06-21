import os
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import time
import numpy as np
import json
from sqlalchemy.orm import Session
from src.database.config import get_db
from src.database.models import User, Conversation, Message
from src.auth.dependencies import get_current_user

from src.auth.router import router as auth_router
from src.conversations.router import router as conversations_router

app = FastAPI(title="Medical QA API", version="1.0")


# Allow frontend to talk to backend (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(conversations_router)


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
    conversation_id: int | None = None

class QuestionResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: str
    latency_seconds: float
    conversation_id: int

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
def ask(
    request: QuestionRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if len(request.question.strip()) < 5:
        raise HTTPException(status_code=400, detail="Question too short")

    # Create a new conversation if none provided
    if request.conversation_id:
        conv = db.query(Conversation).filter(
            Conversation.id == request.conversation_id,
            Conversation.user_id == current_user.id
        ).first()
        if not conv:
            raise HTTPException(status_code=404, detail="Conversation not found")
    else:
        # Auto-title using the first few words of the question
        title = request.question[:50] + ("..." if len(request.question) > 50 else "")
        conv = Conversation(user_id=current_user.id, title=title)
        db.add(conv)
        db.commit()
        db.refresh(conv)

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

    # Save user message
    user_msg = Message(
        conversation_id=conv.id,
        role="user",
        content=request.question
    )
    db.add(user_msg)

    # Save assistant message
    assistant_msg = Message(
        conversation_id=conv.id,
        role="assistant",
        content=answer,
        sources=json.dumps(sources),
        confidence=confidence,
        latency_seconds=latency
    )
    db.add(assistant_msg)
    db.commit()

    return QuestionResponse(
        answer=answer,
        sources=sources,
        confidence=confidence,
        latency_seconds=latency,
        conversation_id=conv.id
    )
