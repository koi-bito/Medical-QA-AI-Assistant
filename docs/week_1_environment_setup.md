# 🛠️ Week 1: Environment Setup & Tooling

This document outlines the system configuration, dependency stack, and engineering tools set up during Days 6–12.

---

## 1. Virtual Environment

* **Tool:** Python `venv`
* **Target Environment:** `medqa_env/`
* **Version Choice:** Python **3.13**
  * *Rationale:* Python 3.13 is fully supported across the Hugging Face and PyTorch ecosystems. Using the absolute newest prereleases (like 3.14) was avoided because they lack precompiled packages (wheels) for major deep learning libraries (like PyTorch and tokenizers), which would cause compiling failures during setup.

---

## 2. GPU Hardware & CUDA Acceleration

* **GPU Used:** NVIDIA GeForce RTX 4050 Laptop GPU (6GB VRAM)
* **API Framework:** CUDA (Compute Unified Device Architecture)
* **Configuration Learning:** 
  * The graphic drivers and CUDA toolkit must be aligned with PyTorch. 
  * Getting `torch.cuda.is_available()` to return `True` required downloading the exact compiled binary corresponding to the local system drivers (e.g., standard Windows installation using the specific stable wheels targeting CUDA 12.x). 
  * Any driver-to-library mismatch would leave the GPU completely invisible to PyTorch.

---

## 3. Tech Stack Dependencies

A pinned set of dependencies was established in `requirements.txt`:
* **Model Fine-Tuning & Quantization:**
  * `transformers` (model loaders & tokenizers)
  * `peft` (Parameter-Efficient Fine-Tuning/LoRA configurations)
  * `trl` (Transformer Reinforcement Learning - high-level `SFTTrainer` wrapper)
  * `bitsandbytes` (4-bit/8-bit quantization handlers)
  * `accelerate` (Hardware abstraction for mixed-precision/multi-GPU training)
* **RAG Pipeline & Embeddings:**
  * `chromadb` (Lightweight local vector database)
  * `sentence-transformers` (Domain-specific embedding generation)
* **Backend, UI, & Operations:**
  * `mlflow` (Experiment tracking)
  * `fastapi` & `uvicorn` (Backend REST API endpoints)
  * `gradio` (Interactive UI demo)
  * `dvc` (Data Version Control for large data stores)
  * `pytest` (API unit testing)

---

## 4. Experiment Tracking Configuration

* **Tool:** MLflow
* **Backend Storage:** Local sqlite database (`mlflow.db`)
* **Tracking UI:** Local host on `http://localhost:5000` via command `mlflow ui`
* **Use Case:** Every training execution automatically logs parameters (learning rate, rank, batch size), metrics (training loss steps, validation loss), and models, creating an organized, searchable timeline of experiments.
