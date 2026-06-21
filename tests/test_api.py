import sys
import numpy as np
from unittest.mock import MagicMock
import os

# Set a fallback DATABASE_URL for CI environments BEFORE importing anything from src
os.environ["DATABASE_URL"] = "sqlite:///./test.db"

# Mock the heavy ML pipeline so tests can run without PyTorch or actual models
mock_pipeline = MagicMock()
mock_embedder = MagicMock()
# Mock encode to return a dummy 2D array for the cosine similarity math
mock_embedder.encode.return_value = np.array([[1.0, 0.0, 0.0]])

mock_pipeline.load_all.return_value = (None, None, mock_embedder, None, None)
mock_pipeline.answer_question.return_value = {
    "answer": "This is a mocked test answer that is long enough.",
    "sources": ["Mock Source 1"]
}

sys.modules['src.rag.pipeline'] = mock_pipeline

# Also mock Groq dependencies just in case USE_GROQ is true in the test environment
mock_retriever = MagicMock()
mock_retriever.load_retriever.return_value = (mock_embedder, None)
mock_retriever.load_reranker.return_value = None
mock_retriever.retrieve_and_rerank.return_value = ["Mock Source 1"]
sys.modules['src.rag.retriever'] = mock_retriever

mock_groq = MagicMock()
mock_groq.answer_with_groq.return_value = "This is a mocked Groq test answer."
sys.modules['src.inference.groq_client'] = mock_groq


import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.api.main import app
from src.database.config import Base, get_db

# Set up a dedicated SQLite database for testing
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
test_engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

# Create tables in the test database
Base.metadata.create_all(bind=test_engine)

def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db

client = TestClient(app)

# Helper: create a test user and get a token
def get_auth_token():
    """Register a test user and return a valid JWT token."""
    import uuid
    # Use a random string so we don't conflict on subsequent test runs
    rand_id = uuid.uuid4().hex[:8]
    rand_email = f"user_{rand_id}@test.com"
    rand_user = f"testuser_{rand_id}"
    client.post("/auth/register", json={
        "email": rand_email,
        "username": rand_user,
        "password": "testpass123"
    })
    # Login uses form data (OAuth2PasswordRequestForm)
    response = client.post("/auth/login", data={
        "username": rand_email,
        "password": "testpass123"
    })
    return response.json()["access_token"]

def auth_headers():
    token = get_auth_token()
    return {"Authorization": f"Bearer {token}"}

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_register_new_user():
    import uuid
    rand_id = uuid.uuid4().hex[:8]
    rand_email = f"newuser_{rand_id}@test.com"
    rand_user = f"newuser_{rand_id}"
    response = client.post("/auth/register", json={
        "email": rand_email,
        "username": rand_user,
        "password": "password123"
    })
    assert response.status_code == 201

def test_ask_without_auth():
    """Verify that /ask rejects unauthenticated requests."""
    response = client.post("/ask", json={"question": "What are symptoms of diabetes?"})
    assert response.status_code == 401

def test_ask_with_auth():
    response = client.post(
        "/ask",
        json={"question": "What are symptoms of diabetes?"},
        headers=auth_headers()
    )
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "conversation_id" in data

def test_ask_empty_question():
    response = client.post(
        "/ask", 
        json={"question": ""},
        headers=auth_headers()
    )
    assert response.status_code == 400

def test_conversation_history():
    headers = auth_headers()
    # Ask a question (creates a conversation)
    ask_response = client.post(
        "/ask",
        json={"question": "What causes high blood pressure?"},
        headers=headers
    )
    assert ask_response.status_code == 200
    conv_id = ask_response.json()["conversation_id"]

    # Retrieve the conversation
    conv_response = client.get(f"/conversations/{conv_id}", headers=headers)
    assert conv_response.status_code == 200
