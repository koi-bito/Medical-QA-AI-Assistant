# Automation Breakdown — Medical QA AI Assistant

What I (your AI assistant) can build for you vs. what only you can do.

---

## Summary

| Category | Days | % of Plan |
|---|---|---|
| 🟢 **I can fully automate** | ~18 days of coding work | ~40% |
| 🟡 **Hybrid** (I write, you run/verify) | ~15 days | ~35% |
| 🔴 **Only you** (learning, GPU, accounts, judgment) | ~12 days | ~25% |

> [!IMPORTANT]
> Even the "automatable" days still require you to **understand** what the code does. If you skip the learning, you won't be able to debug problems, answer interview questions, or extend the project.

---

## PRE-WEEK — Foundations (Days 1–5)

| Day | Task | Who? | Why |
|---|---|---|---|
| 1 | Watch 3Blue1Brown, read HF blog | 🔴 Only you | Learning — no code to write |
| 2 | Run embeddings in Colab | 🟡 Hybrid | I can write the code, but you need to run it in Colab and observe the results |
| 3 | Watch LoRA videos, read blogs | 🔴 Only you | Conceptual learning |
| 4 | Watch RAG video, draw pipeline on paper | 🔴 Only you | Physical drawing + learning |
| 5 | Create HF account, explore dataset | 🔴 Only you | Account creation + exploring data manually |

**Pre-week verdict:** Almost entirely manual. This is intentional — you're building the mental model. **I can't learn for you.**

---

## WEEK 1 — Environment Setup (Days 6–12)

| Day | Task | Who? | Why |
|---|---|---|---|
| 6 | Python + venv setup | 🔴 Only you | Depends on your machine, OS, existing installs |
| 7 | PyTorch + CUDA setup | 🔴 Only you | GPU driver issues are machine-specific, requires `nvidia-smi` verification |
| 8 | Install all libraries | 🟡 Hybrid | I can give you the exact commands, but install errors are machine-specific |
| 9 | GitHub repo + folder structure | 🟢 **I can do this** | I can create every file, folder, `.gitignore`, and `requirements.txt` |
| 10 | Download + explore dataset | 🟡 Hybrid | I can write the notebook, you need to run it and read through examples yourself |
| 11 | MLflow setup | 🟡 Hybrid | I can write the code, you need to run it and verify the UI |
| 12 | Week review + README | 🟢 **I can do this** | I can write the README |

**Week 1 verdict:** Mostly manual due to environment/hardware setup. Once your environment works, I can start doing more.

---

## WEEK 2 — Data Preparation (Days 13–19)

| Day | Task | Who? | Why |
|---|---|---|---|
| 13 | Data quality analysis notebook | 🟢 **I can do this** | I can write the full analysis notebook |
| 14 | Data cleaning script | 🟢 **I can do this** | I can write `data_prep.py` completely |
| 15 | Format data for fine-tuning | 🟢 **I can do this** | I can write the formatting function |
| 16 | Run base Phi-3 (inference) | 🟡 Hybrid | I can write the notebook, but running it requires your GPU |
| 17 | QLoRA config setup | 🟢 **I can do this** | I can write the LoRA config code |
| 18 | Training script skeleton | 🟢 **I can do this** | I can write the full `train.py` skeleton |
| 19 | Week review | 🟡 Hybrid | I can update README, you spot-check data |

**Week 2 verdict:** Heavy automation potential. I can write all 4 scripts/notebooks. You just need to run them and verify outputs look right.

---

## WEEK 3 — Fine-tuning (Days 20–26)

| Day | Task | Who? | Why |
|---|---|---|---|
| 20 | Complete model + data loading functions | 🟢 **I can do this** | I can write the full functions |
| 21 | LoRA + training loop | 🟢 **I can do this** | I can write the complete `train()` function |
| 22 | First training run (small test) | 🔴 Only you | Requires your GPU. Debugging CUDA errors is machine-specific |
| 23 | Full training run | 🔴 Only you | 3-5 hours on your GPU. You need to monitor it |
| 24 | Evaluate fine-tuned model | 🟡 Hybrid | I can write the eval notebook, you need to run it and judge the quality |
| 25 | Push to HuggingFace Hub | 🔴 Only you | Requires your HF account + credentials |
| 26 | Week review | 🟡 Hybrid | I can update README, you write LEARNINGS |

**Week 3 verdict:** I can write all the code, but the actual training runs are 100% you — they need your GPU, your time watching it, and your judgment on quality.

---

## WEEK 4 — RAG Pipeline (Days 27–33)

| Day | Task | Who? | Why |
|---|---|---|---|
| 27 | Download medical documents | 🟢 **I can do this** | I can write the full `data_collection.py` |
| 28 | Clean + chunk documents | 🟢 **I can do this** | I can write `chunking.py` completely |
| 29 | Build vector database | 🟢 **I can do this** | I can write `vector_store.py` completely |
| 30 | Build retrieval function | 🟢 **I can do this** | I can write `retriever.py` completely |
| 31 | Cross-encoder re-ranking | 🟢 **I can do this** | I can write the re-ranking function |
| 32 | Full RAG pipeline | 🟢 **I can do this** | I can write `pipeline.py` completely |
| 33 | Local Gradio demo | 🟢 **I can do this** | I can write the Gradio demo |

**Week 4 verdict:** Highest automation potential — I can write every single file. You just need to run the scripts (they need your GPU for embedding + inference) and verify retrieved chunks make sense.

---

## WEEK 5 — MLOps + API (Days 34–40)

| Day | Task | Who? | Why |
|---|---|---|---|
| 34 | FastAPI endpoint | 🟢 **I can do this** | I can write `main.py` completely |
| 35 | Write API tests | 🟢 **I can do this** | I can write `test_api.py` completely |
| 36 | GitHub Actions CI/CD | 🟢 **I can do this** | I can write the workflow YAML |
| 37 | DVC setup | 🟡 Hybrid | I can write commands, but DVC init is interactive on your machine |
| 38 | Evaluation script | 🟢 **I can do this** | I can write `evaluate.py` completely |
| 39 | Groq API client | 🟢 **I can do this** | I can write the code, but you need to create the account + get the API key |
| 40 | Integration check | 🟡 Hybrid | I can check code, you verify it works end-to-end |

**Week 5 verdict:** Almost fully automatable codewise. Account creation (Groq) is on you.

---

## WEEK 6 — Deployment & Polish (Days 41–47)

| Day | Task | Who? | Why |
|---|---|---|---|
| 41 | Gradio UI | 🟢 **I can do this** | I can write `app.py` completely |
| 42 | Deploy to HF Spaces | 🔴 Only you | Requires your HF account, pushing to Space, setting secrets |
| 43 | Test live demo | 🔴 Only you | User testing — judgment call |
| 44 | Write the README | 🟢 **I can do this** | I can write a polished README |
| 45 | Architecture diagram | 🟡 Hybrid | I can describe it or generate it, you finalize in draw.io |
| 46 | Demo video + LinkedIn | 🔴 Only you | Recording, narrating, posting — entirely personal |
| 47 | Final polish + resume | 🟡 Hybrid | I can clean code, you write/customize the resume bullet |

---

## The Bottom Line

### What I can build for you right now (if you say "go"):

```
src/
├── training/
│   ├── data_prep.py          ← full script
│   └── train.py              ← full script
├── rag/
│   ├── data_collection.py    ← full script
│   ├── chunking.py           ← full script
│   ├── vector_store.py       ← full script
│   ├── retriever.py          ← full script
│   └── pipeline.py           ← full script
├── api/
│   └── main.py               ← full script
├── inference/
│   └── groq_client.py        ← full script
├── evaluation/
│   └── evaluate.py           ← full script
tests/
│   └── test_api.py           ← full script
notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_analysis.ipynb
│   ├── 03_model_exploration.ipynb
│   └── 04_model_evaluation.ipynb
.github/workflows/test.yml
.gitignore
requirements.txt
README.md
app.py                        ← Gradio entry point
```

That's **~15 files** I can create with working code.

### What only you can do:

| Task | Why it can't be automated |
|---|---|
| Watch learning videos (Days 1–5) | Understanding must be yours |
| Install Python, PyTorch, CUDA | Machine-specific troubleshooting |
| Create GitHub, HuggingFace, Groq accounts | Requires your identity |
| Run training on your GPU (Days 22–23) | Requires your RTX 4050 |
| Monitor training loss | Judgment call |
| Judge model quality (Day 24) | Subjective evaluation |
| Deploy to HuggingFace Spaces | Your account, your secrets |
| User testing (Day 43) | Human judgment |
| Record demo video | Your voice, your presentation |
| Write LEARNINGS.md entries | Your personal reflection |
| LinkedIn/resume work | Your career, your words |

---

## Suggested Approach

> [!TIP]
> **Option A — Learn-first (recommended):** Follow the plan day by day. When you reach a coding day, ask me to write the code. You review it, understand it, then run it. This gives you both speed and understanding.
>
> **Option B — Scaffold-first:** Tell me to generate all the code files now. Then work through the plan using the pre-built code, focusing on understanding rather than typing. Risk: you might skip understanding and struggle to debug.

Want me to start generating the code files?
