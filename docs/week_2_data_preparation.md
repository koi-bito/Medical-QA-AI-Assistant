# 🧹 Week 2: Data Preparation & Cleaning

This document outlines the dataset exploration, custom quality filtering, template formatting, and baseline model evaluations executed during Days 13–19.

---

## 1. Raw Dataset Exploration

* **Dataset:** [`lavita/ChatDoctor-HealthCareMagic-100k`](https://huggingface.co/datasets/lavita/ChatDoctor-HealthCareMagic-100k) (112,165 doctor-patient interactions).
* **Fields:**
  * `instruction` — A standard instruction string defining the doctor's persona.
  * `input` — The question or description of symptoms submitted by the patient.
  * `output` — The response, diagnostic recommendations, or treatment instructions from the doctor.

---

## 2. Quality Filtering Criteria

A set of rule-based data cleaners was established in [data_prep.py](file:///c:/GITHUB%20Repos/Medical-QA-AI-Assistant/src/training/data_prep.py) to remove low-quality rows:

* **Minimum Output Length (100 characters):** Excludes short, generic replies such as *"Please consult a doctor"* or *"I cannot diagnose you."* This ensures the model learns real medical context and tone rather than evasiveness.
* **Maximum Output Length (2,000 characters):** Ensures that patient dialogues, when combined with prompts, do not exceed the model's target sequence limit. This avoids silent truncation during training.
* **Minimum Input Length (10 characters):** Filters out empty questions, typos, or single punctuation inputs.
* **Outcome:** Cleaned the raw 112,165 dataset down to **110,139 high-quality examples** (saved in `cleaned_chatdoctor.csv`).

---

## 3. Template Formatting for Phi-3 Mini

The cleaned rows were wrapped in the special tokens required by Phi-3 Mini's chat templates to distinguish between speaker turns:

```text
<|system|>
You are a knowledgeable medical assistant.
Provide clear, accurate information based on the patient's question.
Always recommend consulting a qualified healthcare professional for diagnosis and treatment.
If you are uncertain about something, say so clearly rather than guessing.<|end|>
<|user|>
{patient_question}<|end|>
<|assistant|>
{doctor_response}<|end|>
```

The output was structured and saved to `data/processed/formatted_training_data.csv` (~150.9 MB).

---

## 4. Baseline Model Exploration

Before training, the base model (`microsoft/Phi-3-mini-4k-instruct`) was evaluated on clinical test queries:
* **Observations:** The base model had general medical knowledge but generated answers that were overly formal, detached, or lacked the conversational tone of a medical assistant. It also occasionally hallucinated details or failed to provide standard disclaimer advice automatically.
* **Conclusion:** Fine-tuning was necessary to shift the model's persona to sound like a conversational medical assistant while maintaining safety constraints.
