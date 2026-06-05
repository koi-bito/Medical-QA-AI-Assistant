# Medical QA AI Assistant — Implementation Plan

### Goal: Build a medical Q&A chatbot using a fine-tuned LLM + RAG pipeline, deployed publicly on HuggingFace Spaces.

### Hardware: RTX 4050 (6GB VRAM) | ~20 hrs/week | Beginner-friendly

---

> **How to use this plan:**
> Each day has one clear goal, exact steps, and a "you're done when" checkpoint.
> Estimated time per day: 3–4 hours.
> Keep a file called `LEARNINGS.md` in your repo. Write 3–5 sentences every day — what you learned, what confused you, how you fixed it.
> **Dataset:** We're using `lavita/ChatDoctor-HealthCareMagic-100k` — a dataset of 100k+ real doctor-patient conversations. This dataset has `instruction`, `input`, and `output` fields.

---

## PRE-WEEK — Foundations (Days 1–5)

> **Goal:** Understand the concepts before writing a single line of code. This week prevents weeks of confusion later.

---

### Day 1 — How Neural Networks Work

**Time:** 3–4 hrs

**What you're learning:** What a neural network actually is and how it learns.

**Tasks:**

1. Watch 3Blue1Brown's "Neural Networks" series on YouTube — Episodes 1, 2, and 3 only (~1 hr total). Best visual explanation that exists.
2. Read "What is a Large Language Model?" on HuggingFace's blog (search it, it's free).
3. Write in `LEARNINGS.md`: What is a weight? What does "training" mean in your own words?

**You're done when:** You can explain to yourself what a neural network is and roughly how it learns.

---

### Day 2 — What Embeddings Are (Critical for RAG)

**Time:** 3 hrs

**What you're learning:** How text gets turned into numbers, and why similar sentences produce similar numbers.

**Tasks:**

1. Search YouTube: "Word Embeddings explained visually" — watch any 10-minute video.
2. Open Google Colab (free, no setup needed) and run this:

```python
# First install the library
!pip install sentence-transformers scikit-learn

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('all-MiniLM-L6-v2')
sentences = ["I have a headache", "My head hurts", "I love pizza"]
embeddings = model.encode(sentences)

print("Embedding shape:", embeddings.shape)
print("Headache vs Head hurts:", cosine_similarity([embeddings[0]], [embeddings[1]]))
print("Headache vs Pizza:", cosine_similarity([embeddings[0]], [embeddings[2]]))
# First score should be high (~0.9), second should be low (~0.1)
```

3. Write in `LEARNINGS.md`: Why are embeddings useful? What does "similar meaning = similar vector" actually mean?

**You're done when:** You understand why two sentences with different words can be "similar" mathematically.

---

### Day 3 — What Fine-tuning Means

**Time:** 3 hrs

**What you're learning:** The difference between a base model and a fine-tuned model, and why we fine-tune.

**Tasks:**

1. Read this analogy carefully: A base LLM is like a very smart person who has read the entire internet. Fine-tuning is like giving that person a 6-month internship at a hospital. They learn the specific language, format, and behavior you want — without forgetting everything they knew before.
2. Search YouTube: "Fine-tuning LLMs explained" — watch any video under 15 minutes.
3. Search "What is LoRA fine-tuning" — read one blog post. Key idea: instead of updating all billions of model weights, LoRA adds tiny new layers and only trains those. That's why it fits on a 6GB GPU.
4. Write in `LEARNINGS.md`: What is the difference between a base model and a fine-tuned model? Why does QLoRA exist?

**You're done when:** You understand why we fine-tune instead of just prompting the model.

---

### Day 4 — What RAG Is and Why It Exists

**Time:** 3 hrs

**What you're learning:** How to give an LLM access to documents it wasn't trained on.

**Tasks:**

1. Understand the core problem: LLMs are frozen in time. They can't know what's in a document you wrote yesterday. RAG (Retrieval-Augmented Generation) solves this — it searches relevant documents at query time and feeds them into the prompt as context.
2. Watch YouTube: "RAG explained" — any video under 15 minutes.
3. Draw the RAG pipeline on paper (physically draw it):
   ```
   User question
        ↓
   Question gets turned into a vector (embedded)
        ↓
   Vector search finds similar documents
        ↓
   Top documents go into the prompt as "context"
        ↓
   Model answers using that context
   ```
4. Write in `LEARNINGS.md`: What problem does RAG solve that fine-tuning doesn't?

**You're done when:** You can draw and explain the RAG pipeline from memory.

---

### Day 5 — HuggingFace Orientation + Dataset Preview

**Time:** 3–4 hrs

**What you're learning:** How HuggingFace works and what your training data actually looks like.

**Tasks:**

1. Create a free account at huggingface.co.
2. Search for `microsoft/Phi-3-mini-4k-instruct` on HuggingFace. Read the model card. Don't try to understand everything — just get familiar with how model pages look.
3. Search for `lavita/ChatDoctor-HealthCareMagic-100k` on HuggingFace Datasets. Click "Dataset Viewer" to see what the data looks like.
4. Run this in Google Colab to actually load and inspect the data:

```python
!pip install datasets

from datasets import load_dataset

dataset = load_dataset("lavita/ChatDoctor-HealthCareMagic-100k")
print(dataset)
print("\nOne example:")
print(dataset['train'][0])
```

5. Write in `LEARNINGS.md`: What does one training example look like? What fields/columns does it have?

**You're done when:** You have a HuggingFace account and you've seen what the actual training data looks like.

---

## WEEK 1 — Environment Setup & GitHub (Days 6–12)

> **Goal:** Get your full local development environment working without errors.
> **Heads up:** This is the most frustrating week. That's normal — everyone goes through it.

---

### Day 6 — Python Environment Setup

**Time:** 3–4 hrs

**What you're doing:** Setting up a clean, isolated Python environment for the project.

**Why virtual environments?** They keep this project's libraries separate from everything else on your computer. Without one, installing ML libraries can break other Python projects.

**Tasks:**

1. Install Python 3.11 from python.org. Use 3.11 specifically — some ML libraries still have issues with 3.12+.
2. Install VS Code from code.visualstudio.com. Then install the "Python" extension from the VS Code extensions panel.
3. Open a terminal and create your virtual environment:

```bash
# Create the virtual environment
python -m venv medqa_env

# Activate it — pick your OS:
# Windows:
medqa_env\Scripts\activate
# Mac/Linux:
source medqa_env/bin/activate

# You should now see (medqa_env) at the start of your terminal line
```

4. Install Jupyter inside the environment:

```bash
pip install jupyter notebook ipykernel
python -m ipykernel install --user --name=medqa_env --display-name "Medical QA"
```

5. Open VS Code, create a new `.ipynb` file, select the "Medical QA" kernel from the top right, and run:

```python
print("hello world")
```

**You're done when:** A Jupyter notebook runs successfully inside your virtual environment in VS Code.

**Write in LEARNINGS.md:** What is a virtual environment? Why is 3.11 used instead of the latest Python?

---

### Day 7 — PyTorch + CUDA Setup

**Time:** 4 hrs (budget extra — this day often takes longer)

**What you're doing:** Making sure Python can talk to your GPU.

**Tasks:**

1. Check if CUDA is installed. Open a terminal and run:

```bash
nvidia-smi
```

You should see your RTX 4050 listed. Note the CUDA version shown (e.g., 12.1).

2. Go to pytorch.org → "Get Started" → select your OS, pip, and your CUDA version → copy the install command. It'll look something like:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

3. Verify it worked:

```python
import torch
print("PyTorch version:", torch.__version__)
print("GPU available:", torch.cuda.is_available())   # Must print True
print("GPU name:", torch.cuda.get_device_name(0))    # Should show RTX 4050
```

4. If `cuda.is_available()` returns `False` — don't panic. Google "PyTorch CUDA not available RTX 4050 fix" and follow the top result. This is a very common issue.

**You're done when:** `torch.cuda.is_available()` returns `True`.

**Write in LEARNINGS.md:** What is CUDA? Why does the GPU need special drivers to work with Python?

---

### Day 8 — Install All Project Libraries

**Time:** 3 hrs

**What you're doing:** Installing every library you'll need for the entire project in one go.

**Tasks:**

1. Make sure your virtual environment is activated (you should see `(medqa_env)` in your terminal). Then run:

```bash
# Core ML libraries
pip install transformers peft trl bitsandbytes datasets accelerate

# Vector database + embeddings
pip install chromadb sentence-transformers

# Experiment tracking
pip install mlflow

# API + UI
pip install fastapi uvicorn gradio

# Data versioning + HuggingFace tools
pip install dvc huggingface_hub

# Utilities
pip install pandas numpy pytest
```

> **Windows users:** If `bitsandbytes` fails to install or gives errors, try: `pip install bitsandbytes-windows` or `pip install bitsandbytes>=0.43.0` (native Windows support was added in v0.43). If it still fails, search "bitsandbytes Windows install" for the latest fix.

2. Test that everything imports correctly:

```python
import transformers, peft, trl, datasets, accelerate
import chromadb, sentence_transformers, mlflow
import fastapi, gradio
import pandas, numpy
print("All imports successful!")
```

3. If any import fails, fix them one by one. Google the exact error message.

**You're done when:** All imports work without errors.

**Write in LEARNINGS.md:** What does each library do? Write one sentence per library.

---

### Day 9 — GitHub Repository Setup

**Time:** 3 hrs

**What you're doing:** Creating the project repository and folder structure you'll use for the entire project.

**Tasks:**

1. Create a GitHub account if you don't have one (github.com).
2. Create a new repository called `medical-qa-assistant`. Set it to public.
3. Install Git if you don't have it, then configure it:

```bash
git config --global user.name "Your Name"
git config --global user.email "your@email.com"
```

4. Clone the repo to your local machine, then create this folder structure:

```
medical-qa-assistant/
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── notebooks/
├── src/
│   ├── training/
│   ├── rag/
│   └── api/
├── tests/
├── README.md
├── LEARNINGS.md
├── requirements.txt
└── .gitignore
```

5. Create a `.gitignore` file with this content:

```
# Python
*.pyc
__pycache__/

# Environment
*.env
.env
medqa_env/

# Large files (tracked by DVC or too big for Git)
data/raw/
data/processed/
data/chroma_db/
models/

# MLflow
mlruns/
mlartifacts/

# OS
.DS_Store
Thumbs.db
```

6. Create a `requirements.txt` with the libraries you actually use (don't use `pip freeze` — it dumps hundreds of transitive dependencies that make the file brittle):

```
# Core ML
transformers
peft
trl
bitsandbytes
datasets
accelerate
torch

# RAG
chromadb
sentence-transformers

# Experiment tracking
mlflow

# API + UI
fastapi
uvicorn
gradio

# Data versioning
dvc
huggingface_hub

# Utilities
pandas
numpy
pytest
httpx
```

7. Make your first commit:

```bash
git add .
git commit -m "initial project structure"
git push origin main
```

**You're done when:** Your repository is live on GitHub with the full folder structure.

---

### Day 10 — Download and Explore the Dataset

**Time:** 4 hrs

**What you're doing:** Actually loading and understanding your training data before you touch it.

**Tasks:**

1. Create `notebooks/01_data_exploration.ipynb`.
2. Load the dataset:

```python
from datasets import load_dataset

# 100k+ real doctor-patient conversations
dataset = load_dataset("lavita/ChatDoctor-HealthCareMagic-100k")

print("Dataset structure:", dataset)
print(f"\nTotal training examples: {len(dataset['train'])}")
print("\nFirst example:")
print(dataset['train'][0])
```

3. Explore what you have:

```python
import pandas as pd

df = pd.DataFrame(dataset['train'])
print("Columns:", df.columns.tolist())
print("\nData types:")
print(df.dtypes)
print("\nSample of 5 rows:")
print(df.head())
```

4. For each field in the dataset, answer in your notebook:
   - What does this field contain?
   - What does a good example look like vs a bad one?
   - What format are the questions? What format are the answers?

**You're done when:** You've seen and understood the structure of the dataset and committed your exploration notebook.

---

### Day 11 — MLflow Setup and First Experiment Log

**Time:** 3 hrs

**What you're doing:** Setting up experiment tracking so every training run you do gets logged automatically.

**Why MLflow?** It's like a diary for your ML experiments. Every time you train a model, it records the settings you used and the results — so you can compare runs later and know what worked.

**Tasks:**

1. Test that MLflow works:

```python
import mlflow

mlflow.set_experiment("medqa-experiments")

with mlflow.start_run(run_name="test_run"):
    mlflow.log_param("learning_rate", 0.001)
    mlflow.log_param("epochs", 3)
    mlflow.log_metric("test_accuracy", 0.85)
    mlflow.log_metric("loss", 0.42)

print("Run logged successfully!")
```

2. Start the MLflow UI to view your logs:

```bash
mlflow ui
```

3. Open `http://localhost:5000` in your browser. You should see your test run listed.
4. Commit your exploration notebook.

**You're done when:** You can see experiment logs in the MLflow UI.

**Write in LEARNINGS.md:** Why is experiment tracking useful? What would happen if you didn't track your runs?

---

### Day 12 — Week 1 Review + Buffer

**Time:** 3 hrs

**Tasks:**

1. Fix any setup issues still bothering you. Don't move forward with broken tooling.
2. Make sure your GitHub has commits from every day this week.
3. Write a proper `README.md`:
   - What this project is
   - What tech stack you're using
   - Status: "Week 1 — Environment Setup Complete"
4. Update `LEARNINGS.md` with a week summary: What was the hardest part? What surprised you?
5. Push everything.

**You're done when:** Your repo is clean, README exists, and commits are visible for every day.

---

## WEEK 2 — Data Preparation (Days 13–19)

> **Goal:** Clean and format your data. Good data = good model. Don't rush this week.

---

### Day 13 — Analyze the Data Quality

**Time:** 3 hrs

**What you're doing:** Understanding what good vs bad training examples look like before you clean anything.

**Tasks:**

1. Create `notebooks/02_data_analysis.ipynb`.
2. Analyze example quality:

```python
import pandas as pd
from datasets import load_dataset

dataset = load_dataset("lavita/ChatDoctor-HealthCareMagic-100k")
df = pd.DataFrame(dataset['train'])

# Check the length distribution of answers
df['answer_length'] = df['output'].str.len()
print(df['answer_length'].describe())

# Look at very short answers (likely bad quality)
short_answers = df[df['answer_length'] < 50]
print(f"\nShort answers (< 50 chars): {len(short_answers)}")
print(short_answers['output'].head(10))

# Look at very long answers
long_answers = df[df['answer_length'] > 3000]
print(f"\nVery long answers (> 3000 chars): {len(long_answers)}")
```

3. Read through 20 random examples manually. Note: Which are helpful? Which are vague or useless?

**You're done when:** You know what good vs bad examples look like in your data.

---

### Day 14 — Clean the Data

**Time:** 4 hrs

**What you're doing:** Filtering out low-quality examples so they don't hurt your model.

**Tasks:**

1. Create `src/training/data_prep.py`:

```python
import pandas as pd
from datasets import load_dataset

def is_good_example(row):
    """Return True only if the example is good quality"""
    output = row.get('output', '') or ''
    input_text = row.get('input', '') or ''

    # Too short to be useful
    if len(output) < 100:
        return False
    # So long it'll exceed the model's context window
    if len(output) > 2000:
        return False
    # No real question
    if len(input_text) < 10:
        return False

    return True

def load_and_clean():
    print("Loading dataset...")
    dataset = load_dataset("lavita/ChatDoctor-HealthCareMagic-100k")
    df = pd.DataFrame(dataset['train'])

    print(f"Before cleaning: {len(df)} examples")
    clean_df = df[df.apply(is_good_example, axis=1)].reset_index(drop=True)
    print(f"After cleaning: {len(clean_df)} examples")

    clean_df.to_csv("data/processed/cleaned_chatdoctor.csv", index=False)
    print("Saved to data/processed/cleaned_chatdoctor.csv")
    return clean_df

if __name__ == "__main__":
    load_and_clean()
```

2. Run it:

```bash
python src/training/data_prep.py
```

**You're done when:** You have a cleaned dataset saved at `data/processed/cleaned_chatdoctor.csv`.

---

### Day 15 — Format Data for Fine-tuning

**Time:** 4 hrs

**What you're doing:** Reformatting the data into the exact structure Phi-3 Mini expects during training.

**Why formatting matters:** LLMs during fine-tuning expect a specific conversation format with special tokens that mark who said what. If the format is wrong, the model learns garbage.

**Tasks:**

1. Add a formatting function to `src/training/data_prep.py`:

```python
SYSTEM_PROMPT = """You are a knowledgeable medical assistant.
Provide clear, accurate information based on the patient's question.
Always recommend consulting a qualified healthcare professional for diagnosis and treatment.
If you are uncertain about something, say so clearly rather than guessing."""

def format_for_training(row):
    """Format one example into Phi-3 Mini's expected chat format"""
    return {
        "text": f"<|system|>\n{SYSTEM_PROMPT}<|end|>\n<|user|>\n{row['input']}<|end|>\n<|assistant|>\n{row['output']}<|end|>"
    }

def prepare_training_data():
    clean_df = pd.read_csv("data/processed/cleaned_chatdoctor.csv")
    formatted = clean_df.apply(format_for_training, axis=1, result_type='expand')
    formatted.to_csv("data/processed/formatted_training_data.csv", index=False)
    print(f"Formatted {len(formatted)} examples")
    print("\nSample formatted example:")
    print(formatted['text'][0])

if __name__ == "__main__":
    df = load_and_clean()
    prepare_training_data()
```

2. Run it and look at a few examples. Does the format look right?

**You're done when:** `data/processed/formatted_training_data.csv` exists and the format looks correct when you print it.

---

### Day 16 — Run Phi-3 Mini (Inference Only — No Training Yet)

**Time:** 4 hrs

**What you're doing:** Loading the base model and seeing how it behaves BEFORE training. This is your baseline to compare against later.

**Tasks:**

1. Create `notebooks/03_model_exploration.ipynb`:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

model_name = "microsoft/Phi-3-mini-4k-instruct"

# Load in 4-bit quantization so it fits in 6GB VRAM
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

print("Loading model (this will take a few minutes)...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)
print("Model loaded!")

# Ask it a medical question
def ask(question):
    prompt = f"<|user|>\n{question}<|end|>\n<|assistant|>\n"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("<|assistant|>")[-1].strip()

# Test with 5 questions and write down the answers
questions = [
    "I have a severe headache and sensitivity to light. What could this be?",
    "What are the symptoms of type 2 diabetes?",
    "Is it safe to take ibuprofen every day?",
    "I've had a persistent cough for 3 weeks. Should I see a doctor?",
    "What's the difference between a cold and the flu?"
]

for q in questions:
    print(f"\nQuestion: {q}")
    print(f"Answer: {ask(q)}")
    print("-" * 50)
```

2. Write down the responses. Are they helpful? Do they sound like a real doctor? Do they hallucinate?

**You're done when:** You've run the base model and documented its responses on 5 questions.

---

### Day 17 — Understand QLoRA Conceptually + Setup Config

**Time:** 3 hrs

**What you're doing:** Understanding what LoRA rank and alpha actually mean before you use them.

**Key concept:** LoRA adds small "adapter" matrices to specific layers of the model. During training, only these tiny matrices get updated — not the 3.8 billion base model weights. QLoRA does this with the model loaded in 4-bit, saving VRAM. The result: you train ~2 million parameters instead of 3.8 billion.

**Tasks:**

1. In your notebook, set up the LoRA config and check how many parameters are actually trainable:

```python
from peft import LoraConfig, get_peft_model, TaskType

lora_config = LoraConfig(
    r=16,                                          # Rank: how expressive the adapters are
    lora_alpha=32,                                 # Scaling factor (usually 2x rank)
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Which layers to adapt
    lora_dropout=0.05,                             # Light regularization
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

# Apply LoRA to the model
peft_model = get_peft_model(model, lora_config)
peft_model.print_trainable_parameters()
# Expected output: trainable params: ~2M || all params: ~3.8B || trainable%: ~0.05%
```

2. Write in `LEARNINGS.md`: What does the trainable% tell you? What would happen if you increased the rank?

**You're done when:** You understand what LoRA rank and alpha mean and have seen the trainable parameter count.

---

### Day 18 — Create the Training Script Skeleton

**Time:** 3 hrs

**What you're doing:** Writing the structure of your training script with clear TODOs for what you'll fill in next week.

**Tasks:**

1. Create `src/training/train.py`:

```python
"""
Medical QA Fine-tuning Script
Fine-tunes Phi-3 Mini on medical conversations using QLoRA.
Hardware target: RTX 4050 (6GB VRAM)
"""
import torch
import mlflow
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

# ─── Config ──────────────────────────────────────────────────────────
MODEL_NAME    = "microsoft/Phi-3-mini-4k-instruct"
DATA_PATH     = "data/processed/formatted_training_data.csv"
OUTPUT_DIR    = "models/phi3-medical-lora"
EPOCHS        = 2
BATCH_SIZE    = 2
LEARNING_RATE = 2e-4
MAX_SEQ_LEN   = 512
# ─────────────────────────────────────────────────────────────────────

def load_model_and_tokenizer():
    pass  # TODO: Day 20

def load_training_data():
    pass  # TODO: Day 20

def train():
    pass  # TODO: Day 21

if __name__ == "__main__":
    train()
```

2. Commit this to GitHub.

**You're done when:** The skeleton is committed with clear TODOs marked.

---

### Day 19 — Week 2 Review + Buffer

**Time:** 3 hrs

**Tasks:**

1. Spot-check 20 random examples in your formatted training data — do they look right?
2. Make sure the full data pipeline is clear in your head: raw dataset → clean → format → ready for training.
3. Update your README: add a "Data" section explaining what dataset you're using and why.
4. Look ahead at Week 3 — read through the training days so nothing surprises you.

**You're done when:** Data pipeline is fully committed and `LEARNINGS.md` is updated.

---

## WEEK 3 — Fine-tuning (Days 20–26)

> **Goal:** Train your model. This is the core of the project.

---

### Day 20 — Complete the Model + Data Loading Functions

**Time:** 4–5 hrs

**Tasks:**

1. Fill in the functions in `src/training/train.py`:

```python
def load_model_and_tokenizer():
    print("Loading model in 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    model.config.use_cache = False  # Required for training

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer

def load_training_data():
    df = pd.read_csv(DATA_PATH)
    # Use 10k examples — enough to see improvement, fast enough to train
    df = df.sample(10000, random_state=42).reset_index(drop=True)

    # 90/10 train/eval split
    split = int(len(df) * 0.9)
    train_df = df.iloc[:split]
    eval_df  = df.iloc[split:]

    train_dataset = Dataset.from_pandas(train_df)
    eval_dataset  = Dataset.from_pandas(eval_df)

    print(f"Train examples: {len(train_dataset)}, Eval examples: {len(eval_dataset)}")
    return train_dataset, eval_dataset
```

**You're done when:** Both functions run without errors.

---

### Day 21 — Add LoRA + Training Loop

**Time:** 4 hrs

**Tasks:**

1. Complete the `train()` function:

```python
def train():
    print("=== Starting Training ===")
    model, tokenizer = load_model_and_tokenizer()

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_dataset, eval_dataset = load_training_data()

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=4,      # Effective batch size = 2 * 4 = 8
        learning_rate=LEARNING_RATE,
        fp16=True,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=200,
        save_steps=500,
        warmup_ratio=0.03,
        max_seq_length=MAX_SEQ_LEN,
        dataset_text_field="text",
        report_to="none",
    )

    mlflow.set_experiment("medqa-experiments")
    with mlflow.start_run(run_name="phi3-medical-qlora"):
        mlflow.log_params({
            "model": MODEL_NAME,
            "epochs": EPOCHS,
            "lora_r": 16,
            "learning_rate": LEARNING_RATE,
            "training_examples": len(train_dataset)
        })

        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=training_args,
            tokenizer=tokenizer,
        )

        trainer.train()
        final_loss = trainer.state.log_history[-1].get('loss', 0)
        mlflow.log_metric("final_loss", final_loss)

    print("Saving adapter weights...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Done! Saved to {OUTPUT_DIR}")
```

**You're done when:** The full `train()` function is written and the file has no import errors.

---

### Day 22 — First Training Run (Small Test)

**Time:** 4–5 hrs

**What you're doing:** Running a tiny test to confirm everything works before committing to a 4-hour full run.

**Tasks:**

1. Temporarily change `train.py` to use tiny settings for testing:

```python
# At the top of train(), temporarily set:
df = df.sample(200, random_state=42)  # Just 200 examples
EPOCHS = 1                             # Just 1 epoch
```

2. Run it:

```bash
python src/training/train.py
```

3. Watch for errors. Common ones and fixes:
   - `CUDA out of memory` → set `BATCH_SIZE = 1` and `gradient_accumulation_steps = 8`
   - Tokenizer padding errors → check that `tokenizer.pad_token = tokenizer.eos_token` is set
   - Data format errors → print one example going into the trainer

4. When it finishes (should take ~10 minutes), open MLflow UI and verify the run was logged:

```bash
mlflow ui
# Open http://localhost:5000
```

**You're done when:** A small training run completes without errors and shows up in MLflow.

---

### Day 23 — Full Training Run

**Time:** 5–6 hrs (mostly waiting)

**What you're doing:** Running the real training on 10,000 examples.

**Tasks:**

1. Revert to full settings (10k examples, 2 epochs) and run:

```bash
python src/training/train.py
```

2. This will take 3–5 hours on your RTX 4050. While it runs:
   - Watch the loss numbers — they should generally decrease over time
   - If loss becomes `NaN`, stop immediately and investigate
   - Open MLflow UI and watch metrics update live
   - Write in `LEARNINGS.md`: What do you expect the model to be better at after training?

3. When done, verify the output folder:

```bash
# Windows:
dir models\phi3-medical-lora
# Mac/Linux:
ls models/phi3-medical-lora/
# Should contain: adapter_config.json, adapter_model.safetensors (or .bin), tokenizer files
```

**You're done when:** Training completes and adapter weights are saved in `models/phi3-medical-lora/`.

---

### Day 24 — Evaluate Your Fine-tuned Model

**Time:** 4 hrs

**What you're doing:** Comparing your fine-tuned model against the base model on the same questions.

**Tasks:**

1. Create `notebooks/04_model_evaluation.ipynb`:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch

base_model_name = "microsoft/Phi-3-mini-4k-instruct"
adapter_path    = "models/phi3-medical-lora"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

print("Loading fine-tuned model...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)
model     = PeftModel.from_pretrained(base_model, adapter_path)
tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)

def ask(question):
    prompt = f"<|user|>\n{question}<|end|>\n<|assistant|>\n"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("<|assistant|>")[-1].strip()
```

2. Ask the SAME 5 questions you tested on Day 16. Compare the answers side by side.
3. Does the fine-tuned model sound more like a real doctor? Is the format more consistent?

**You're done when:** You've documented a before/after comparison of base vs fine-tuned model responses.

---

### Day 25 — Push to HuggingFace Hub

**Time:** 3 hrs

**Tasks:**

1. Log in to HuggingFace from your terminal:

```bash
huggingface-cli login
# Paste your HF token from huggingface.co/settings/tokens
```

2. Push your adapter weights:

```python
from huggingface_hub import HfApi

api = HfApi()
# Replace "your-username" with your actual HF username
api.create_repo(repo_id="your-username/phi3-medical-lora", private=False, exist_ok=True)

model.push_to_hub("your-username/phi3-medical-lora")
tokenizer.push_to_hub("your-username/phi3-medical-lora")
print("Model pushed!")
```

3. Go to HuggingFace and check your model page. Write a short model card explaining what it is, what data it was trained on, and its limitations.

**You're done when:** Your adapter weights are live on HuggingFace Hub.

---

### Day 26 — Week 3 Review + Buffer

**Time:** 3 hrs

**Tasks:**

1. Log your training details in `LEARNINGS.md`: what hyperparameters did you use? What was the final loss?
2. If model quality seems poor, note what you'd try differently (lower learning rate? more data? more epochs?).
3. Update README: add a "Model" section linking to your HuggingFace model.
4. Commit everything.

> **Milestone:** You have a fine-tuned medical LLM live on HuggingFace. For a beginner, that's genuinely impressive.

---

## WEEK 4 — RAG Pipeline (Days 27–33)

> **Goal:** Give your model access to external medical knowledge via document retrieval.

---

### Day 27 — Download Medical Documents

**Time:** 4 hrs

**What you're doing:** Collecting the documents your RAG system will search through.

**Tasks:**

1. Create `src/rag/data_collection.py`:

```python
import requests
import json
import os

def fetch_medlineplus_articles():
    """Fetch health topic summaries from MedlinePlus API (public domain, free)"""
    os.makedirs("data/raw/medlineplus", exist_ok=True)
    articles = []

    topics = [
        "diabetes", "hypertension", "headache", "fever", "chest pain",
        "back pain", "depression", "anxiety", "asthma", "arthritis",
        "heart disease", "stroke", "kidney disease", "liver disease",
        "thyroid", "anemia", "pneumonia", "bronchitis", "sinusitis", "migraine"
    ]

    for topic in topics:
        url = "https://wsearch.nlm.nih.gov/ws/query"
        params = {"db": "healthTopics", "term": topic, "retmax": 25}
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                articles.append({"topic": topic, "content": response.text})
                print(f"Fetched: {topic}")
        except Exception as e:
            print(f"Failed: {topic} — {e}")

    with open("data/raw/medlineplus/articles.json", "w") as f:
        json.dump(articles, f)

    print(f"\nTotal articles fetched: {len(articles)}")
    return articles

if __name__ == "__main__":
    fetch_medlineplus_articles()
```

2. Run it:

```bash
python src/rag/data_collection.py
```

**You're done when:** You have articles downloaded in `data/raw/medlineplus/articles.json`.

---

### Day 28 — Clean and Chunk Documents

**Time:** 4 hrs

**What you're doing:** Splitting large documents into small chunks that can be searched individually.

**Why chunking?** You can't put a 10-page document into a prompt — context windows are limited. Instead, you split it into ~400-word chunks. When a user asks a question, you find the most relevant chunk and include only that.

**Tasks:**

1. Create `src/rag/chunking.py`:

```python
import json
import re

def clean_text(text):
    """Remove XML/HTML tags and extra whitespace"""
    text = re.sub(r'<[^>]+>', ' ', text)     # Remove XML tags
    text = re.sub(r'\s+', ' ', text).strip() # Collapse whitespace
    return text

def chunk_text(text, chunk_size=400, overlap=50):
    """
    Split text into overlapping chunks.
    Overlap ensures a sentence isn't cut off at a boundary and lost.
    """
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if len(chunk.strip()) > 100:  # Skip very short chunks
            chunks.append(chunk)

    return chunks

def process_all_articles():
    with open("data/raw/medlineplus/articles.json") as f:
        articles = json.load(f)

    all_chunks = []
    for article in articles:
        cleaned = clean_text(article['content'])
        chunks   = chunk_text(cleaned)
        for chunk in chunks:
            all_chunks.append({"topic": article['topic'], "text": chunk})

    with open("data/processed/chunks.json", "w") as f:
        json.dump(all_chunks, f)

    print(f"Total chunks created: {len(all_chunks)}")
    return all_chunks

if __name__ == "__main__":
    process_all_articles()
```

2. Run it:

```bash
python src/rag/chunking.py
```

**You're done when:** `data/processed/chunks.json` exists with your document chunks.

---

### Day 29 — Build the Vector Database

**Time:** 4 hrs

**What you're doing:** Converting all chunks into vectors and storing them in ChromaDB for fast similarity search.

**Tasks:**

1. Create `src/rag/vector_store.py`:

```python
import chromadb
from sentence_transformers import SentenceTransformer
import json

def build_vector_store(chunks_path="data/processed/chunks.json"):
    with open(chunks_path) as f:
        chunks = json.load(f)

    print(f"Loading embedding model...")
    # PubMedBERT is trained on medical text — better than a generic embedder
    embedder = SentenceTransformer("pritamdeka/S-PubMedBert-MS-MARCO")

    print("Setting up ChromaDB...")
    client     = chromadb.PersistentClient(path="data/chroma_db")
    collection = client.get_or_create_collection("medical_docs")

    print(f"Embedding {len(chunks)} chunks...")
    batch_size = 100

    for i in range(0, len(chunks), batch_size):
        batch      = chunks[i:i + batch_size]
        texts      = [c['text'] for c in batch]
        embeddings = embedder.encode(texts).tolist()
        ids        = [f"chunk_{i + j}" for j in range(len(batch))]
        metadatas  = [{"topic": c['topic']} for c in batch]

        collection.add(embeddings=embeddings, documents=texts, ids=ids, metadatas=metadatas)

        if (i // batch_size) % 5 == 0:
            print(f"  Progress: {min(i + batch_size, len(chunks))}/{len(chunks)}")

    print("Vector store built and saved to data/chroma_db/")
    return collection

if __name__ == "__main__":
    build_vector_store()
```

2. Run it:

```bash
python src/rag/vector_store.py
```

**You're done when:** ChromaDB is populated and persisted to `data/chroma_db/`.

---

### Day 30 — Build the Retrieval Function

**Time:** 3 hrs

**What you're doing:** Writing the function that searches ChromaDB for relevant chunks given a question.

**Tasks:**

1. Create `src/rag/retriever.py`:

```python
import chromadb
from sentence_transformers import SentenceTransformer

def load_retriever():
    embedder   = SentenceTransformer("pritamdeka/S-PubMedBert-MS-MARCO")
    client     = chromadb.PersistentClient(path="data/chroma_db")
    collection = client.get_collection("medical_docs")
    return embedder, collection

def retrieve(query, collection, embedder, top_k=10):
    """Find the most similar document chunks to a user's question"""
    query_embedding = embedder.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=top_k)
    return results['documents'][0]  # List of matching chunks

if __name__ == "__main__":
    embedder, collection = load_retriever()
    test_questions = [
        "What are symptoms of type 2 diabetes?",
        "How is hypertension treated?",
        "What causes a migraine?"
    ]
    for q in test_questions:
        chunks = retrieve(q, collection, embedder)
        print(f"\nQuestion: {q}")
        print(f"Top chunk: {chunks[0][:200]}...")
```

2. Run it and check: do the retrieved chunks seem relevant to each question?

**You're done when:** Retrieval is working and returning relevant chunks.

---

### Day 31 — Add Cross-Encoder Re-ranking

**Time:** 4 hrs

**What you're doing:** Adding a second-pass model that re-scores retrieved chunks for better accuracy.

**Why re-ranking?** Vector search finds chunks that are semantically _similar_ to your question. Re-ranking uses a more powerful model to score each chunk for actual _relevance_. It's slower but more accurate — you get 10 candidates from vector search, then re-rank to keep only the best 3.

**Tasks:**

1. Add re-ranking to `src/rag/retriever.py`:

```python
from sentence_transformers import CrossEncoder

def load_reranker():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def retrieve_and_rerank(query, collection, embedder, reranker, top_k=10, final_k=3):
    # Step 1: Get top 10 candidates via vector search
    candidates = retrieve(query, collection, embedder, top_k=top_k)

    # Step 2: Score each candidate against the query
    pairs  = [(query, chunk) for chunk in candidates]
    scores = reranker.predict(pairs)

    # Step 3: Sort by score, keep best 3
    ranked     = sorted(zip(scores, candidates), reverse=True)
    top_chunks = [chunk for score, chunk in ranked[:final_k]]

    return top_chunks
```

2. Compare results with and without re-ranking on 5 questions. Does it improve?

**You're done when:** Re-ranking is working and improving result quality.

---

### Day 32 — Wire It All Together: Full RAG Pipeline

**Time:** 4 hrs

**What you're doing:** Connecting retrieval to the model so it can answer questions using document context.

**Tasks:**

1. Create `src/rag/pipeline.py`:

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from src.rag.retriever import load_retriever, load_reranker, retrieve_and_rerank

RAG_SYSTEM_PROMPT = """You are a helpful medical assistant.
Answer the patient's question using the provided medical context.
If the context doesn't contain enough information, say so honestly.
Always recommend consulting a real doctor for diagnosis and treatment."""

def load_all():
    """Load everything needed to answer questions"""
    # Load fine-tuned model
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    base       = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3-mini-4k-instruct",
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    model     = PeftModel.from_pretrained(base, "models/phi3-medical-lora")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct", trust_remote_code=True)

    embedder, collection = load_retriever()
    reranker             = load_reranker()

    return model, tokenizer, embedder, collection, reranker

def answer_question(question, model, tokenizer, embedder, collection, reranker):
    # Step 1: Retrieve relevant chunks
    context_chunks = retrieve_and_rerank(question, collection, embedder, reranker)
    context        = "\n\n".join(context_chunks)

    # Step 2: Build the prompt with context
    prompt = f"""<|system|>
{RAG_SYSTEM_PROMPT}

Medical context:
{context}<|end|>
<|user|>
{question}<|end|>
<|assistant|>
"""

    # Step 3: Generate answer
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=300, do_sample=False)

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer   = response.split("<|assistant|>")[-1].strip()

    return {"answer": answer, "sources": context_chunks}
```

2. Test the full pipeline end-to-end on 5 different medical questions.

**You're done when:** You can ask a question and receive an answer with source chunks.

---

### Day 33 — Week 4 Review + Local Gradio Demo

**Time:** 3 hrs

**Tasks:**

1. Build a quick local demo to see the system in action:

```python
import gradio as gr
from src.rag.pipeline import load_all, answer_question

print("Loading all components...")
model, tokenizer, embedder, collection, reranker = load_all()

def chat(question):
    if not question.strip():
        return "Please enter a question.", ""
    result       = answer_question(question, model, tokenizer, embedder, collection, reranker)
    sources_text = "\n\n---\n\n".join(result['sources'])
    return result['answer'], sources_text

demo = gr.Interface(
    fn=chat,
    inputs=gr.Textbox(label="Ask a medical question"),
    outputs=[gr.Textbox(label="Answer"), gr.Textbox(label="Sources Used")]
)
demo.launch()
```

2. Run it and show it to someone. Get their reaction.
3. Update README with a "RAG Pipeline" section.

> **Milestone:** You have a working RAG-augmented medical QA system running locally.

---

## WEEK 5 — MLOps + API (Days 34–40)

> **Goal:** Make the project look like real engineering, not just a notebook.

---

### Day 34 — Build the FastAPI Endpoint

**Time:** 4 hrs

**Tasks:**

1. Create `src/api/main.py`:

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import time
import numpy as np

app = FastAPI(title="Medical QA API", version="1.0")

# Load everything at startup (once)
from src.rag.pipeline import load_all, answer_question
print("Loading models...")
model, tokenizer, embedder, collection, reranker = load_all()
print("Ready!")

class QuestionRequest(BaseModel):
    question: str

class QuestionResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: str
    latency_seconds: float

def estimate_confidence(question, sources, embedder):
    """Estimate confidence based on how well the sources match the question.
    Uses cosine similarity between the question and retrieved chunks."""
    if not sources:
        return "low"
    query_emb  = embedder.encode([question])
    source_emb = embedder.encode(sources)
    similarities = np.dot(source_emb, query_emb.T).flatten()
    avg_similarity = float(np.mean(similarities))
    if avg_similarity > 0.5:
        return "high"
    elif avg_similarity > 0.3:
        return "medium"
    return "low"

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/ask", response_model=QuestionResponse)
def ask(request: QuestionRequest):
    if len(request.question.strip()) < 5:
        raise HTTPException(status_code=400, detail="Question too short")

    start  = time.time()
    result = answer_question(request.question, model, tokenizer, embedder, collection, reranker)
    latency = round(time.time() - start, 2)

    confidence = estimate_confidence(request.question, result['sources'], embedder)

    return QuestionResponse(
        answer=result['answer'],
        sources=result['sources'],
        confidence=confidence,
        latency_seconds=latency
    )
```

2. Run the API:

```bash
uvicorn src.api.main:app --reload
```

3. Open `http://localhost:8000/docs` — FastAPI auto-generates interactive API docs. Test your `/ask` endpoint from there.

**You're done when:** The API is running and you can make requests from the browser.

---

### Day 35 — Write Tests for the API

**Time:** 3 hrs

**Tasks:**

1. Install the test client:

```bash
pip install httpx pytest
```

2. Create `tests/test_api.py`:

```python
import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_ask_valid_question():
    response = client.post("/ask", json={"question": "What are symptoms of diabetes?"})
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert len(data["answer"]) > 10

def test_ask_empty_question():
    response = client.post("/ask", json={"question": ""})
    assert response.status_code == 400

def test_ask_returns_sources():
    response = client.post("/ask", json={"question": "What causes high blood pressure?"})
    data = response.json()
    assert "sources" in data
    assert len(data["sources"]) > 0
```

3. Run tests:

```bash
pytest tests/ -v
```

**You're done when:** All tests pass.

---

### Day 36 — GitHub Actions CI/CD

**Time:** 3 hrs

**What you're doing:** Setting up automatic test runs every time you push code to GitHub.

**Tasks:**

1. Create `.github/workflows/test.yml`:

```yaml
name: Run Tests

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: |
          pip install fastapi httpx pytest pydantic uvicorn

      - name: Run tests
        run: pytest tests/ -v
```

2. Push to GitHub. Go to the "Actions" tab and watch it run.
3. Once it passes, add a badge to your README:

```markdown
![Tests](https://github.com/YOUR_USERNAME/medical-qa-assistant/actions/workflows/test.yml/badge.svg)
```

**You're done when:** Green CI badge is on your README.

---

### Day 37 — DVC for Data Versioning

**Time:** 3 hrs

**What you're doing:** Tracking your large data files with DVC so they're versioned without being in Git.

**Tasks:**

1. Initialize DVC:

```bash
pip install dvc
dvc init
git add .dvc .dvcignore
git commit -m "initialize DVC"
```

2. Track your processed data:

```bash
dvc add data/processed/chunks.json
dvc add data/processed/formatted_training_data.csv
git add data/processed/*.dvc
git commit -m "track processed datasets with DVC"
```

**You're done when:** Your key data files are DVC-tracked and committed.

---

### Day 38 — Evaluation Script

**Time:** 4 hrs

**Tasks:**

1. Create `src/evaluation/evaluate.py`:

```python
"""
Evaluate the QA system on a fixed set of test questions.
Checks whether the answer contains expected medical topics.
Uses multiple aliases per topic to handle medical synonyms
(e.g., "urination" also matches "polyuria", "frequent urination").
"""
import json
from src.rag.pipeline import load_all, answer_question

# Each topic has a list of aliases — if ANY alias matches, the topic counts as a hit.
# This handles cases where the model uses medical terminology instead of plain language.
TEST_QUESTIONS = [
    {"question": "What are symptoms of type 2 diabetes?",
     "expected_topics": [
         ["blood sugar", "glucose", "hyperglycemia"],
         ["thirst", "polydipsia"],
         ["urination", "polyuria", "frequent urination"],
         ["fatigue", "tired", "exhaustion"],
         ["blurred", "vision"]
     ]},
    {"question": "How is high blood pressure diagnosed?",
     "expected_topics": [
         ["blood pressure"],
         ["mmhg", "mm hg", "millimeters"],
         ["measurement", "monitor", "reading"],
         ["systolic", "diastolic", "120/80"]
     ]},
    {"question": "What is ibuprofen used for?",
     "expected_topics": [
         ["pain", "analgesic"],
         ["inflammation", "anti-inflammatory", "swelling"],
         ["fever", "antipyretic", "temperature"],
         ["nsaid", "nonsteroidal", "non-steroidal"]
     ]},
    {"question": "What causes asthma?",
     "expected_topics": [
         ["airway", "airways", "bronchial"],
         ["inflammation", "inflamed", "swelling"],
         ["trigger", "allergen", "irritant"],
         ["breathing", "breathe", "respiratory", "wheez"]
     ]},
    {"question": "What are symptoms of depression?",
     "expected_topics": [
         ["mood", "feeling"],
         ["sleep", "insomnia", "fatigue"],
         ["interest", "anhedonia", "pleasure"],
         ["energy", "tired", "exhaustion"],
         ["sad", "hopeless", "worthless"]
     ]},
]

def topic_hit(answer, topic_aliases):
    """Check if any alias for a topic appears in the answer"""
    return any(alias in answer for alias in topic_aliases)

def evaluate():
    print("Loading models...")
    model, tokenizer, embedder, collection, reranker = load_all()

    results = []
    for q in TEST_QUESTIONS:
        answer = answer_question(
            q['question'], model, tokenizer, embedder, collection, reranker
        )['answer'].lower()

        hits  = [aliases for aliases in q['expected_topics'] if topic_hit(answer, aliases)]
        score = len(hits) / len(q['expected_topics'])
        results.append({
            "question": q['question'],
            "score": score,
            "matched_topics": len(hits),
            "total_topics": len(q['expected_topics'])
        })
        print(f"Q: {q['question'][:50]}... | Score: {score:.0%}")

    avg_score = sum(r['score'] for r in results) / len(results)
    print(f"\nAverage score: {avg_score:.1%}")

    with open("evaluation_results.json", "w") as f:
        json.dump({"avg_score": avg_score, "results": results}, f, indent=2)

    return avg_score

if __name__ == "__main__":
    score = evaluate()
    if score < 0.5:
        print("WARNING: Score below 50%")
        exit(1)
```

> **Note:** This evaluation uses keyword matching with medical synonyms. It's a simple heuristic — a model could technically mention the right words in the wrong context. For a more robust evaluation, you could use an LLM-as-judge approach (e.g., ask GPT-4 to rate the answers), but keyword matching is a solid starting point.

**You're done when:** Evaluation script runs and produces a score.

---

### Day 39 — Groq API for Fast Cloud Inference

**Time:** 3 hrs

**What you're doing:** Adding Groq as an alternative to local inference — much faster for the public demo.

> **Important trade-off:** Groq doesn't support custom fine-tuned models. The public demo will use Groq's Llama 3 (8B) instead of your fine-tuned Phi-3. Your fine-tuned model is still used for local inference and evaluation. Think of it this way: Groq gives you speed for the demo, your fine-tuned model gives you quality for local use. In your README, be transparent about this — mention that the live demo uses Llama 3 via Groq, while the local version uses your fine-tuned Phi-3.

**Tasks:**

1. Create a free account at console.groq.com. Get your API key.
2. Install Groq:

```bash
pip install groq
```

3. Create `src/inference/groq_client.py`:

```python
from groq import Groq
import os

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def answer_with_groq(question, context_chunks, model="llama3-8b-8192"):
    """Use Groq for fast cloud inference — good for the public demo"""
    context = "\n\n".join(context_chunks)
    prompt  = f"""You are a helpful medical assistant. Use the context below to answer the question.
Always recommend consulting a healthcare professional.

Context:
{context}

Question: {question}
Answer:"""

    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        max_tokens=500
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    # Quick test
    test_answer = answer_with_groq(
        "What are symptoms of diabetes?",
        ["Diabetes causes high blood sugar. Common symptoms include increased thirst, frequent urination, fatigue, and blurred vision."]
    )
    print(test_answer)
```

4. Set your key and test it:

```bash
# Windows:
set GROQ_API_KEY=your_key_here
# Mac/Linux:
export GROQ_API_KEY=your_key_here

python src/inference/groq_client.py
```

**You're done when:** Groq inference is working and returning answers.

---

### Day 40 — Week 5 Review + Integration Check

**Time:** 3 hrs

**Tasks:**

1. Make sure all components work together: FastAPI → RAG pipeline → Groq or local inference.
2. Fix any integration bugs.
3. Update README with an "API Documentation" section showing example requests.
4. Look ahead: Week 6 is deployment — the finish line.

---

## WEEK 6 — Deployment & Polish (Days 41–47)

> **Goal:** Get it live. The world can see it.

---

### Day 41 — Build the Gradio UI

**Time:** 4 hrs

**Tasks:**

1. Create `app.py` (the HuggingFace Spaces entry point):

```python
import gradio as gr
import os
from src.rag.retriever import load_retriever, load_reranker, retrieve_and_rerank
from src.inference.groq_client import answer_with_groq

DISCLAIMER = """⚠️ **Medical Disclaimer:** This tool is for educational purposes only.
It is NOT a substitute for professional medical advice, diagnosis, or treatment.
Always consult a qualified healthcare provider."""

# Load retrieval components (no local model needed on Spaces — we use Groq)
print("Loading retriever...")
embedder, collection = load_retriever()
reranker             = load_reranker()
print("Ready!")

def ask_medical_question(question, history):
    if not question.strip():
        return history, "", "Please enter a question."

    chunks       = retrieve_and_rerank(question, collection, embedder, reranker)
    answer       = answer_with_groq(question, chunks)
    sources_text = "\n\n---\n\n".join([f"**Source {i+1}:** {s[:300]}..." for i, s in enumerate(chunks)])

    history.append((question, answer))
    return history, sources_text, ""

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏥 Medical QA Assistant")
    gr.Markdown(DISCLAIMER)

    with gr.Row():
        with gr.Column(scale=2):
            chatbot       = gr.Chatbot(height=400)
            question_input = gr.Textbox(placeholder="Ask a medical question...", label="Your Question")
            submit_btn    = gr.Button("Ask", variant="primary")
        with gr.Column(scale=1):
            sources_output = gr.Markdown(label="Sources Used")

    submit_btn.click(
        ask_medical_question,
        inputs=[question_input, chatbot],
        outputs=[chatbot, sources_output, question_input]
    )

demo.launch()
```

**You're done when:** Gradio app runs locally with a clean UI.

---

### Day 42 — Deploy to HuggingFace Spaces

**Time:** 4 hrs

**What you're doing:** Getting your app live on HuggingFace Spaces so anyone can use it.

**Key challenge:** Your ChromaDB vector store is a local directory — it won't exist on HuggingFace Spaces. You need to upload it as a HuggingFace dataset so the Space can download it at startup.

**Tasks:**

1. First, upload your ChromaDB data as a HuggingFace dataset:

```python
from huggingface_hub import HfApi
import os

api = HfApi()

# Create a dataset repo to hold your vector store
api.create_repo(repo_id="your-username/medical-qa-vectorstore", repo_type="dataset", private=False, exist_ok=True)

# Upload the entire chroma_db directory
api.upload_folder(
    folder_path="data/chroma_db",
    repo_id="your-username/medical-qa-vectorstore",
    repo_type="dataset",
    path_in_repo="chroma_db"
)
print("Vector store uploaded!")
```

2. Update `app.py` to download the vector store on startup:

```python
# Add this at the top of app.py, before loading the retriever:
from huggingface_hub import snapshot_download
import os

if not os.path.exists("data/chroma_db"):
    print("Downloading vector store...")
    snapshot_download(
        repo_id="your-username/medical-qa-vectorstore",
        repo_type="dataset",
        local_dir="data"
    )
    print("Vector store ready!")
```

3. Create a new Space on huggingface.co/spaces (type: Gradio).
4. Create a `requirements.txt` for the Space:

```
gradio
sentence-transformers
chromadb
groq
transformers
torch
huggingface_hub
```

5. Create a `README.md` for the Space with the metadata header:

```yaml
---
title: Medical QA Assistant
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
---
```

6. Add your `GROQ_API_KEY` as a Space secret (Settings → Secrets).
7. Push your code to the Space:

```bash
git remote add space https://huggingface.co/spaces/your-username/medical-qa
git push space main
```

8. Wait for the build. Check the "Logs" tab if it fails.

**You're done when:** Your app is live at `huggingface.co/spaces/your-username/medical-qa`.

---

### Day 43 — Test the Live Demo

**Time:** 3 hrs

**Tasks:**

1. Test 20+ different questions. Check:
   - Does it answer common questions well?
   - Does it say "I don't know" for obscure things?
   - Is the disclaimer visible?
   - Are sources showing?
   - Is response time acceptable?
2. Fix any bugs.
3. Share the link with a friend or family member. Watch them use it. Note where they get confused.

**You're done when:** Demo is stable and you've done real user testing.

---

### Day 44 — Write the README

**Time:** 4 hrs

**Tasks:**
Write a README that a recruiter can understand in 60 seconds:

```markdown
# 🏥 Medical QA Assistant

![Tests](badge_link) | [Live Demo](hf_spaces_link) | [Model on HuggingFace](model_link)

> A medical Q&A system built with fine-tuned Phi-3 Mini + RAG pipeline

## Live Demo

[Link to HuggingFace Space]

## Architecture

[Diagram — see Day 45]

## What It Does

- Fine-tunes Phi-3 Mini (3.8B) on 100k+ real doctor-patient conversations
- Retrieves relevant medical documents using domain-specific BioMedical embeddings
- Re-ranks retrieved documents with a cross-encoder for better accuracy
- Serves answers via FastAPI with CI/CD through GitHub Actions
- Deployed publicly on HuggingFace Spaces

## Setup

[Clear installation steps]

## How It Works

[Explain the pipeline in plain English — 3-4 paragraphs]

## Limitations

- Not a substitute for real medical advice
- Model may hallucinate — always verify with a professional
- Trained on a limited dataset; rare conditions may not be covered
```

**You're done when:** README is polished and someone with no context can understand the project.

---

### Day 45 — Architecture Diagram

**Time:** 2 hrs

**Tasks:**

1. Go to draw.io (free, browser-based) and draw this diagram:

```
User Question
     ↓
[Gradio UI / FastAPI]
     ↓
[PubMedBERT Embedder]
     ↓
[ChromaDB Vector Search] → Top 10 chunks
     ↓
[Cross-Encoder Re-ranker] → Top 3 chunks
     ↓
[Prompt Builder]
     ↓
[Fine-tuned Phi-3 Mini / Groq API]
     ↓
Answer + Sources
```

2. Export as PNG, add to README. This diagram alone impresses interviewers.

**You're done when:** Architecture diagram is in your README.

---

### Day 46 — Record Demo Video + LinkedIn Post

**Time:** 3 hrs

**Tasks:**

1. Download Loom (free screen recorder).
2. Record a 2-minute demo:
   - 0:00–0:20: "This is a medical QA assistant I built. Here's how it works." (show diagram)
   - 0:20–1:20: Ask 3 real questions live. Show answers + sources.
   - 1:20–1:40: "Under the hood: fine-tuned Phi-3, RAG with re-ranking, deployed on HuggingFace"
   - 1:40–2:00: "Here's the GitHub with full code and CI/CD pipeline"
3. Add the Loom link to your README.
4. Post on LinkedIn with a short write-up of what you built and what you learned.

**You're done when:** Demo video is live and LinkedIn post is up.

---

### Day 47 — Final Polish + Resume Bullet

**Time:** 3 hrs

**Tasks:**

1. Go through every file. Remove debug print statements, clean up commented code.
2. Make sure every notebook has a markdown cell at the top explaining what it does.
3. Write your resume bullet:

```
Built and deployed a medical Q&A assistant — fine-tuned Phi-3 Mini (3.8B)
on 100k+ real doctor-patient conversations using QLoRA on an RTX 4050, layered
a RAG pipeline with domain-specific BioMedical embeddings and cross-encoder
re-ranking, built a FastAPI backend with CI/CD via GitHub Actions, and deployed
publicly on HuggingFace Spaces.
```

4. Add it to your resume and LinkedIn.

> 🎉 **Project complete.** You built a production-quality ML project from scratch. That's not a small thing.

---

## WEEKS 7–8 — Optional Improvements (Days 48–56)

Pick any that interest you:

**Make the model better:**

- Train on the full 100k examples instead of 10k
- Experiment with different LoRA ranks (r=8 vs r=32) and compare results
- Try a higher `lora_alpha` value

**Make the RAG better:**

- Add more documents (PubMed abstracts via the free PubMed API)
- Try different chunk sizes (200 tokens vs 600 tokens) and compare quality
- Add hybrid search (keyword + vector combined)

**Make the product better:**

- Add conversation history (multi-turn chat)
- Add a "Was this helpful?" feedback button
- Add a disclaimer checkbox the user must confirm before getting answers

**Make the engineering better:**

- Add proper logging with Python's built-in `logging` module
- Add rate limiting to the FastAPI endpoint
- Add Docker containerization (`Dockerfile` + `docker-compose.yml`)
- Write more comprehensive tests with edge cases

---

## Daily Checklist

Use this every single day:

- [ ] Did I commit to GitHub today?
- [ ] Did I write in LEARNINGS.md?
- [ ] Did I actually understand what I coded, or just copy-paste it?
- [ ] What will I do first tomorrow?

---

## When You Get Stuck (You Will)

1. **Read the error message carefully** — it usually tells you exactly what's wrong
2. **Google the exact error message** in quotes
3. **Check HuggingFace forums** at discuss.huggingface.co — most errors have been seen before
4. **Check GPU memory:** `nvidia-smi` in terminal
5. **Restart the kernel** — fixes mysterious issues more often than it should
6. **Reduce batch size** if you get "CUDA out of memory" errors

---

_Good luck. You're going to build something real._
