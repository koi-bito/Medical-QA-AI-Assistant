import os
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator
import time
import numpy as np
import json
from sqlalchemy.orm import Session
from src.database.config import get_db
from src.database.models import User, Conversation, Message
from src.auth.dependencies import get_current_user
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from src.auth.router import router as auth_router
from src.conversations.router import router as conversations_router

# Rate limiter — key by the requester's IP address
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(title="Medical QA API", version="1.0")
app.state.limiter = limiter

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded. Max 10 questions per hour per IP. Please try again later."}
    )


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

    @field_validator('question')
    @classmethod
    def validate_question(cls, v):
        v = v.strip()
        if len(v) < 5:
            raise ValueError("Question must be at least 5 characters")
        if len(v) > 1000:
            raise ValueError("Question must be under 1000 characters")
        return v

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
@limiter.limit("10/hour")
def ask(
    request: Request,
    body: QuestionRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Validation is now handled by the Pydantic field_validator above

    # Create a new conversation if none provided
    if body.conversation_id:
        conv = db.query(Conversation).filter(
            Conversation.id == body.conversation_id,
            Conversation.user_id == current_user.id
        ).first()
        if not conv:
            raise HTTPException(status_code=404, detail="Conversation not found")
    else:
        # Auto-title using the first few words of the question
        title = body.question[:50] + ("..." if len(body.question) > 50 else "")
        conv = Conversation(user_id=current_user.id, title=title)
        db.add(conv)
        db.commit()
        db.refresh(conv)

    start  = time.time()
    
    if USE_GROQ:
        chunks = retrieve_and_rerank(body.question, collection, embedder, reranker)
        answer = answer_with_groq(body.question, chunks)
        sources = chunks
    else:
        result = answer_question(body.question, model, tokenizer, embedder, collection, reranker)
        answer = result['answer']
        sources = result['sources']
        
    latency = round(time.time() - start, 2)

    confidence = estimate_confidence(body.question, sources, embedder)

    # Save user message
    user_msg = Message(
        conversation_id=conv.id,
        role="user",
        content=body.question
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
