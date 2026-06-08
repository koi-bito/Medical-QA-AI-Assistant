# 🧠 Pre-Week: Foundations

This document summarizes the foundational machine learning concepts established during Days 1–5 of the project.

---

## 1. Key Concept Definitions

### Neural Network Weights

- **What is a weight?** A weight is a numerical value representing the importance or strength of a connection between nodes (neurons) in a neural network. For example, during image recognition, weights dictate how strongly a specific set of pixels triggers the detection of a visual pattern.

### Training

- **What does training mean?** Training is the iterative process where a model refines its predictions. The model makes a prediction, computes the error using a **Loss Function**, calculates the **Gradients** (direction and rate of change of the error), and updates its internal **Weights** accordingly to reduce future error.

### Embeddings

- **What are embeddings?** An embedding is a low-dimensional, continuous vector representation (a list of numbers) of discrete variables like words, sentences, or documents.
- **"Similar meaning = similar vector":** By mapping words into a multi-dimensional semantic space, words with related meanings end up closer to each other. For instance, the cosine similarity between the vectors of `"happy"` and `"joyful"` will be very high (nearly pointing in the same direction), whereas `"happy"` and `"car"` will be far apart.

---

## 2. Architectural Choices

### Base vs. Fine-tuned Models

- **Base Model:** Pre-trained on massive datasets (e.g., internet text) to learn general language structures, grammar, and basic reasoning (e.g., standard `microsoft/Phi-3-mini-4k-instruct`).
- **Fine-tuned Model:** Takes a base model and adapts its weights on a smaller, highly focused, domain-specific dataset (like medical dialogs) to learn a particular response tone, format, and clinical context.

### Why QLoRA?

- Full-parameter fine-tuning of a 3.8-billion-parameter model like Phi-3 would require massive VRAM (typically 24GB+).
- **QLoRA (Quantized Low-Rank Adaptation)** solves this hardware limitation by:
  1. Quantizing the base model weights to 4-bit representation (dramatically reducing the VRAM footprint).
  2. Injecting small, trainable **Low-Rank Adapters** (LoRA matrices) into attention modules.
  3. Updating only these adapter weights (~2M parameters instead of 3.8B), allowing fine-tuning to run on local consumer GPUs (6GB VRAM).

### RAG (Retrieval-Augmented Generation) vs. Fine-tuning

- **Fine-tuning** changes the internal weights of the LLM to learn the **how** (tone, style, guidelines, and format).
- **RAG** solves the **knowledge-access** problem. It queries a separate vector database at runtime to fetch fresh, factual source documents and inserts them into the prompt, giving the model external context to read before writing.
- Combining **both** ensures the model replies in a professional, cautious medical persona (fine-tuning) while grounding its medical statements in trusted facts (RAG) to prevent hallucination.
