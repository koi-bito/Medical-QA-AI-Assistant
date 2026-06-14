# Week 5 — MLOps + API (Days 34–40)

## Overview
This week focused on transforming the local RAG pipeline and fine-tuned model into a robust, production-ready backend service. We implemented a FastAPI backend, established automated testing and CI/CD, managed our datasets with DVC, created a reproducible evaluation script, and integrated Groq for high-speed cloud inference.

## Key Accomplishments

### 1. FastAPI Backend (Day 34)
- Built a RESTful API using **FastAPI** to serve the Q&A system.
- Designed endpoints for health checks (`/health`) and inference (`/ask`), utilizing Pydantic models for strict input/output validation.
- Implemented global model loading at startup to minimize per-request latency.

### 2. Automated Testing (Day 35)
- Set up **pytest** for automated unit and integration testing.
- Utilized FastAPI's `TestClient` to programmatically test the endpoints, ensuring the API correctly handles valid requests, missing data, and returns expected sources.

### 3. Continuous Integration (Day 36)
- Configured **GitHub Actions** to automate the testing pipeline.
- Workflows were set up to run tests automatically on every push and pull request to the `main` branch, preventing broken code from being merged.

### 4. Data Version Control (Day 37)
- Integrated **DVC (Data Version Control)** to track large data files (like processed datasets and chunks) separately from Git.
- This ensures reproducibility without bloating the GitHub repository with massive files.

### 5. Systematic Evaluation (Day 38)
- Developed an automated evaluation script in `src/evaluation/evaluate.py`.
- Evaluated the model using a keyword and synonym-matching heuristic to verify the clinical accuracy of the generated answers across a set of diverse medical topics.

### 6. Cloud Inference with Groq (Day 39)
- Integrated the **Groq API** to perform lightning-fast cloud inference.
- This provides an alternative to local GPU inference, which is crucial for handling multiple concurrent requests in the planned public HuggingFace Spaces demo.

### 7. Integration & Documentation (Day 40)
- Refactored the API to conditionally use either the Groq client or the local fine-tuned model via an environment variable (`USE_GROQ`).
- Ensured seamless integration of the retriever pipeline with the dynamic backend.
- Updated the main `README.md` with comprehensive API documentation and cURL examples.

## Conclusion
Week 5 successfully moved the project from local experimentation into a structured, production-ready MLOps framework. We now have a reliable, fully tested API backed by continuous integration and scalable inference options, paving the way for our final deployment.
