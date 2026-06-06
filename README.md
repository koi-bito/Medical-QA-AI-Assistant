# 🏥 Medical QA Assistant

> A medical Q&A chatbot built with a fine-tuned LLM + RAG pipeline, deployed publicly on HuggingFace Spaces.

**Status:** Week 1–2 — Environment Setup & Data Preparation

---

## Tech Stack

| Component | Technology |
|---|---|
| Base Model | Phi-3 Mini (3.8B) |
| Fine-tuning | QLoRA via PEFT + TRL |
| Embeddings | PubMedBERT (domain-specific) |
| Vector DB | ChromaDB |
| Re-ranking | Cross-Encoder (ms-marco) |
| API | FastAPI |
| UI | Gradio |
| Cloud Inference | Groq API |
| Experiment Tracking | MLflow |
| Data Versioning | DVC |
| CI/CD | GitHub Actions |

## Hardware

- GPU: NVIDIA GeForce RTX 4050 Laptop GPU (6GB VRAM)
- Training: QLoRA 4-bit quantization

## Dataset

Training data: [`lavita/ChatDoctor-HealthCareMagic-100k`](https://huggingface.co/datasets/lavita/ChatDoctor-HealthCareMagic-100k) — 100k+ real doctor-patient conversations.

Each example has:
- `instruction` — the doctor's role prompt
- `input` — the patient's question/description
- `output` — the doctor's response

## Project Structure

```
medical-qa-assistant/
├── data/
│   ├── raw/                 # Downloaded raw data (DVC tracked)
│   └── processed/           # Cleaned and formatted data (DVC tracked)
├── models/                  # Fine-tuned adapter weights (gitignored)
├── notebooks/               # Exploration and evaluation notebooks
├── src/
│   ├── training/            # Data prep and training scripts
│   ├── rag/                 # RAG pipeline (collection, chunking, retrieval)
│   ├── api/                 # FastAPI backend
│   ├── inference/           # Groq cloud inference client
│   └── evaluation/          # Evaluation script
├── tests/                   # API tests
├── app.py                   # Gradio entry point (HuggingFace Spaces)
├── requirements.txt
└── .gitignore
```

## Setup

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/medical-qa-assistant.git
cd medical-qa-assistant

# Create virtual environment
py -3.13 -m venv medqa_env
medqa_env\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Limitations

- This is an educational project — **NOT** a substitute for real medical advice.
- The model may hallucinate — always verify with a qualified healthcare professional.
- Trained on a limited dataset; rare conditions may not be well-covered.
