import sys
import numpy as np
from unittest.mock import MagicMock

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

import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_ask_valid_question():
    response = client.post("/ask", json={"question": "What are symptoms of diabetes?"})
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert len(data["answer"]) > 10

def test_ask_empty_question():
    response = client.post("/ask", json={"question": ""})
    assert response.status_code == 400

def test_ask_returns_sources():
    response = client.post("/ask", json={"question": "What causes high blood pressure?"})
    data = response.json()
    assert "sources" in data
    assert len(data["sources"]) > 0
