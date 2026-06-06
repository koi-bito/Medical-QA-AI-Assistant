# 🏥 Medical QA Assistant

> A medical Q&A chatbot built with a fine-tuned LLM + RAG pipeline, deployed publicly on HuggingFace Spaces.

**Status:** Week 1–2 Complete — Environment Setup & Data Preparation ✅ | Next: Week 3 — Fine-tuning

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
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_analysis.ipynb
│   ├── 03_model_exploration.ipynb
│   └── 04_qlora_exploration.ipynb
├── src/
│   ├── training/
│   │   ├── data_prep.py     # Clean + format the dataset
│   │   └── train.py         # QLoRA fine-tuning script
│   ├── rag/                 # RAG pipeline (Week 4)
│   ├── api/                 # FastAPI backend (Week 5)
│   ├── inference/           # Groq cloud inference client (Week 5)
│   └── evaluation/          # Evaluation script (Week 6)
├── tests/                   # pytest API tests (Week 6)
├── app.py                   # Gradio entry point — HuggingFace Spaces (Week 7)
├── requirements.txt
└── .gitignore
```

## Setup

```bash
# Clone the repo
git clone https://github.com/koi-bito/Medical-QA-AI-Assistant.git
cd Medical-QA-AI-Assistant

# Create virtual environment
py -3.13 -m venv medqa_env
medqa_env\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Running the Data Pipeline

```bash
# Step 1: Download, clean, and format the training data (~5 min)
python src/training/data_prep.py

# Step 2: Run fine-tuning (~3–5 hrs on RTX 4050)
python src/training/train.py

# Step 3: View training metrics
mlflow ui   # then open http://localhost:5000
```

## Progress

| Week | Topic | Status |
|------|-------|--------|
| Pre-Week (Days 1–5) | Foundations — NN, Embeddings, RAG, HuggingFace | ✅ Done |
| Week 1 (Days 6–12) | Environment Setup — Python, CUDA, Libraries, GitHub | ✅ Done |
| Week 2 (Days 13–19) | Data Preparation — Clean, Format, Baseline Inference | ✅ Done |
| Week 3 (Days 20–26) | Fine-tuning — QLoRA training on Phi-3 Mini | 🔄 Next |
| Week 4 (Days 27–33) | RAG Pipeline — ChromaDB, PubMedBERT embeddings | ⏳ Upcoming |
| Week 5 (Days 34–40) | API — FastAPI backend + Gradio UI | ⏳ Upcoming |
| Week 6 (Days 41–47) | Evaluation + CI/CD | ⏳ Upcoming |
| Week 7 (Days 48–54) | Deployment — HuggingFace Spaces | ⏳ Upcoming |

## Limitations

- This is an educational project — **NOT** a substitute for real medical advice.
- The model may hallucinate — always verify with a qualified healthcare professional.
- Trained on a limited dataset; rare conditions may not be well-covered.
