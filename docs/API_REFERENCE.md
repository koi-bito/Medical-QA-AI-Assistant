# Medical QA Assistant API Reference

Base URL: `https://medical-qa-ai-assistant.onrender.com`

This backend is built with FastAPI. All endpoints support standard HTTP protocols and CORS is configured for the frontend. 

## Interactive Documentation
FastAPI provides automatic interactive API documentation:
- **Swagger UI:** `/docs`
- **ReDoc:** `/redoc`

---

## 1. Authentication

### `POST /auth/register`
Creates a new user account in the PostgreSQL database.

**Request Body (JSON):**
```json
{
  "email": "user@example.com",
  "username": "johndoe",
  "password": "securepassword123"
}
```

**Response (200 OK):**
```json
{
  "message": "User registered successfully"
}
```

### `POST /auth/login`
Authenticates a user and returns a JWT access token. 
*Note: Uses `x-www-form-urlencoded` format (OAuth2 standard).*

**Request Body (Form Data):**
- `username` (string)
- `password` (string)

**Response (200 OK):**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

---

## 2. Conversations

*All endpoints require a valid JWT token in the `Authorization: Bearer <token>` header.*

### `GET /conversations`
Retrieves a list of all conversations for the authenticated user.

**Response (200 OK):**
```json
[
  {
    "id": 1,
    "title": "What are the symptoms of...",
    "created_at": "2024-05-10T12:00:00Z"
  }
]
```

### `GET /conversations/{conversation_id}`
Retrieves a specific conversation and all its associated messages (both user and assistant).

**Response (200 OK):**
```json
{
  "id": 1,
  "title": "What are the symptoms of...",
  "messages": [
    {
      "id": 10,
      "role": "user",
      "content": "What is ibuprofen used for?",
      "sources": null,
      "confidence": null
    },
    {
      "id": 11,
      "role": "assistant",
      "content": "Ibuprofen is a nonsteroidal anti-inflammatory drug...",
      "sources": "[{\"chunk_id\": \"123\", \"text\": \"...\"}]",
      "confidence": "high"
    }
  ]
}
```

### `DELETE /conversations/{conversation_id}`
Deletes a specific conversation and cascades to delete all associated messages.

**Response (200 OK):**
```json
{
  "status": "deleted"
}
```

---

## 3. Q&A and Inference

*All endpoints require a valid JWT token in the `Authorization: Bearer <token>` header.*

### `POST /ask`
Submits a question to the RAG pipeline. If no `conversation_id` is provided, a new conversation is automatically created.
*Rate limited to 10 requests per hour per IP.*

**Request Body (JSON):**
```json
{
  "question": "What is ibuprofen used for?",
  "conversation_id": 123  // Optional
}
```

**Response (200 OK):**
```json
{
  "answer": "Ibuprofen is a nonsteroidal anti-inflammatory drug (NSAID) used for treating pain, fever, and inflammation.",
  "sources": [
    {
      "chunk_id": "doc_142",
      "topic": "NSAIDs",
      "text": "Ibuprofen is used to reduce fever and treat pain...",
      "distance": 0.23
    }
  ],
  "confidence": "high",
  "latency_seconds": 1.45,
  "conversation_id": 123,
  "message_id": 456
}
```

### `POST /feedback`
Allows users to rate assistant responses (Thumbs up/down).

**Request Body (JSON):**
```json
{
  "message_id": 456,
  "rating": "up"  // "up" or "down"
}
```

**Response (200 OK):**
```json
{
  "status": "ok"
}
```

---

## 4. System

### `GET /health`
A simple health check endpoint used by deployment services (like Render) to verify the API is running.

**Response (200 OK):**
```json
{
  "status": "ok"
}
```
