# 🏥 Medical QA Assistant

![Tests](https://github.com/koi-bito/Medical-QA-AI-Assistant/actions/workflows/test.yml/badge.svg) | [Live Demo](https://huggingface.co/spaces/koi-bito/medical-qa) | [Vector DB on HuggingFace](https://huggingface.co/datasets/koi-bito/medical-qa-vectorstore)

> A medical Q&A system built with fine-tuned Phi-3 Mini + RAG pipeline

## 🧠 Journey & Learnings

Want to see how this was built? Read my **[Day-by-Day Key Learnings](LEARNINGS.md)** to see the development process, challenges faced, and what was learned along the way.

## Live Demo

Try the live public demo deployed on Hugging Face Spaces:  
👉 **[Medical QA Assistant - Live Demo](https://huggingface.co/spaces/koi-bito/medical-qa)**

## Architecture

_(Architecture diagram placeholder — to be completed on Day 45)_

## What It Does

- **Fine-tunes Phi-3 Mini (3.8B)** using QLoRA on 100k+ real doctor-patient conversations.
- **Retrieves relevant medical context** using domain-specific BioMedical embeddings (`PubMedBERT`).
- **Re-ranks retrieved documents** with a cross-encoder (`ms-marco-MiniLM`) to dramatically improve accuracy.
- **Provides High-Speed Inference** using Groq's cloud APIs.
- **Serves answers via a FastAPI backend** equipped with unit tests and GitHub Actions CI/CD pipelines.
- **Deployed publicly via a Gradio UI** securely hosted on Hugging Face Spaces.

## Setup

To run this project locally on your machine:

```bash
# 1. Clone the repository
git clone https://github.com/koi-bito/Medical-QA-AI-Assistant.git
cd Medical-QA-AI-Assistant

# 2. Create and activate a Python virtual environment (Python 3.13)
py -3.13 -m venv medqa_env
medqa_env\Scripts\activate  # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

### Running the System

```bash
# Start the local FastAPI server
uvicorn src.api.main:app --reload

# In a separate terminal, launch the Gradio UI
python app.py
```

## How It Works

The Medical QA Assistant operates on a **Retrieval-Augmented Generation (RAG)** pipeline. Instead of relying solely on an AI model's static memory, the system fetches real, clinically rooted information to ground its answers in facts.

When a user asks a medical question, the input is converted into numerical vectors (embeddings) using a specialized medical model (`PubMedBERT`). The system searches a ChromaDB vector database—built from over 100,000 real doctor-patient interactions—for the most relevant historical conversations and document chunks.

To ensure accuracy, a Cross-Encoder heavily analyzes and re-ranks the top retrieved results, filtering out low-quality matches. Finally, the highest-scoring medical contexts are passed alongside the user's question to a high-speed cloud LLM (via Groq), which synthesizes a clear, accurate, and transparent response that cites the retrieved sources directly.

## Progress

| Week                                                   | Topic                                                | Status         |
| ------------------------------------------------------ | ---------------------------------------------------- | -------------- |
| [Pre-Week (Days 1–5)](docs/pre_week_foundations.md)    | Foundations — NN, Embeddings, RAG, HuggingFace       | ✅ Done        |
| [Week 1 (Days 6–12)](docs/week_1_environment_setup.md) | Environment Setup — Python, CUDA, Libraries, GitHub  | ✅ Done        |
| [Week 2 (Days 13–19)](docs/week_2_data_preparation.md) | Data Preparation — Clean, Format, Baseline Inference | ✅ Done        |
| [Week 3 (Days 20–26)](docs/week_3_fine_tuning.md)      | Fine-tuning — QLoRA training on Phi-3 Mini           | ✅ Done        |
| [Week 4 (Days 27–33)](docs/week_4_rag_pipeline.md)     | RAG Pipeline — ChromaDB, PubMedBERT embeddings       | ✅ Done        |
| [Week 5 (Days 34–40)](docs/week_5_mlops_api.md)        | MLOps + API — FastAPI, Tests, CI/CD, Evaluation      | ✅ Done        |
| Week 6 (Days 41–47)                                    | Deployment & Polish — Gradio UI                      | ✅ In Progress |
| Week 7 (Days 48–54)                                    | Deployment — HuggingFace Spaces                      | ⏳ Upcoming    |

## Limitations

- **Not a substitute for real medical advice.** Always consult a qualified healthcare provider.
- **Model may hallucinate.** While grounded in RAG, the model can still generate inaccurate information.
- **Trained on a limited dataset.** Rare conditions or highly specialized medical contexts may not be covered effectively.
