# Week 4 — RAG Pipeline (Days 27–33)

## Overview
This week focused on building the Retrieval-Augmented Generation (RAG) pipeline to give our fine-tuned LLM access to reliable, external medical knowledge. This is critical for preventing hallucinations and grounding the model's responses in factual data.

## Key Accomplishments

### 1. Data Collection (Day 27)
- **Source:** Fetched public domain health topic summaries from the **MedlinePlus API**.
- **Topics:** Covered 20 common medical conditions (e.g., diabetes, hypertension, asthma, migraine).
- **Result:** Saved raw article content as JSON for processing.

### 2. Cleaning and Chunking (Day 28)
- **Cleaning:** Stripped unnecessary HTML/XML tags from the MedlinePlus data.
- **Chunking:** Split the articles into 400-word chunks with a 50-word overlap. This overlap ensures context is not abruptly cut off between chunks, making the text digestible for the embedding model and the LLM context window.

### 3. Vector Database Setup (Day 29)
- **Embeddings:** Used `pritamdeka/S-PubMedBert-MS-MARCO`, a domain-specific BERT model fine-tuned on medical text, to embed all chunks.
- **Storage:** Stored the vectors in **ChromaDB**, a lightweight local vector database, allowing for rapid semantic similarity search.

### 4. Retrieval & Re-ranking (Days 30–31)
- **Bi-Encoder Search:** Implemented initial semantic search against ChromaDB to quickly retrieve the top 10 most relevant document chunks based on the user's query.
- **Cross-Encoder Re-ranking:** Added a second pass using `cross-encoder/ms-marco-MiniLM-L-6-v2`. This model evaluates the question and candidate chunks together, re-scoring them to select the absolute most relevant top 3 chunks, significantly boosting accuracy.

### 5. Wiring the Pipeline Together (Day 32)
- Built `src/rag/pipeline.py` to seamlessly combine retrieval, re-ranking, and text generation.
- **Prompt Engineering:** Engineered a system prompt that injects the retrieved chunks directly into the model's context before the user's question.
- **Bug Fix:** Adjusted `max_new_tokens` and output parsing logic (slicing token IDs instead of string splitting) to ensure robust and complete generation outputs.

### 6. Interactive Demo (Day 33)
- Created `src/rag/demo.py` using **Gradio**.
- Wrapped the entire multi-model pipeline into an easy-to-use web interface where users can ask questions and see both the generated answer and the source documents used to ground the response.

## Conclusion
We successfully transitioned from a standalone, fine-tuned LLM to an integrated RAG system. The model now reliably provides answers that are backed by verified medical literature and transparently displays its sources. Next up is packaging this system into a robust API endpoint.
