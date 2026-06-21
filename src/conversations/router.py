from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from src.database.config import get_db
from src.database.models import User, Conversation, Message
from src.auth.dependencies import get_current_user
from src.conversations.schemas import (
    ConversationCreate, ConversationResponse, ConversationDetailResponse
)

router = APIRouter(prefix="/conversations", tags=["Conversations"])

@router.post("/", response_model=ConversationResponse, status_code=201)
def create_conversation(
    request: ConversationCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Start a new conversation thread."""
    conv = Conversation(user_id=current_user.id, title=request.title)
    db.add(conv)
    db.commit()
    db.refresh(conv)
    return conv

@router.get("/", response_model=list[ConversationResponse])
def list_conversations(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """List all conversations for the logged-in user, newest first."""
    return (
        db.query(Conversation)
        .filter(Conversation.user_id == current_user.id)
        .order_by(Conversation.created_at.desc())
        .all()
    )

@router.get("/{conversation_id}", response_model=ConversationDetailResponse)
def get_conversation(
    conversation_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get a conversation with all its messages."""
    conv = (
        db.query(Conversation)
        .filter(Conversation.id == conversation_id, Conversation.user_id == current_user.id)
        .first()
    )
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conv

@router.delete("/{conversation_id}", status_code=204)
def delete_conversation(
    conversation_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete a conversation and all its messages."""
    conv = (
        db.query(Conversation)
        .filter(Conversation.id == conversation_id, Conversation.user_id == current_user.id)
        .first()
    )
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    db.delete(conv)
    db.commit()
