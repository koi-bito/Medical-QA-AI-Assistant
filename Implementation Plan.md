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

1. Install Python 3.13 from python.org. Use 3.13 specifically — libraries have updated since this plan was written and 3.13 is the stable stable release standard now.
2. Install VS Code from code.visualstudio.com. Then install the "Python" extension from the VS Code extensions panel.
3. Open a terminal and create your virtual environment:

```bash
# Create the virtual environment
py -3.13 -m venv medqa_env

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

**Write in LEARNINGS.md:** What is a virtual environment? Why is 3.13 used instead of the latest Python (e.g. 3.14)?

---

### Day 7 — PyTorch + CUDA Setup

**Time:** 4 hrs (budget extra — this day often takes longer)

**What you're doing:** Making sure Python can talk to your GPU.

**Tasks:**

1. Check if CUDA is installed. Open a terminal and run:

```bash
nvidia-smi
```

You should see your RTX 4050 listed. Note the CUDA version shown (e.g., 12.1 or newer like 12.4 / 12.6).

2. Go to pytorch.org → "Get Started" → select your OS, pip, and your CUDA version → copy the install command. It'll look something like:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
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
          python-version: "3.13"

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
sdk_version: 5.0.0
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

## PHASE 2 — Full-Stack AI Product (Days 46–73)

> **Goal:** Transform this from a machine learning demo into a real, production-quality web application with user authentication, persistent conversation history, and a custom frontend. The HuggingFace Space stays as a lightweight public demo. This new website is the real product.

> **Tech Stack:**
>
> - **Backend:** FastAPI (already built — we'll extend it)
> - **Database:** MySQL (installed locally) + SQLAlchemy ORM
> - **Auth:** JWT (JSON Web Tokens) with bcrypt password hashing
> - **Frontend:** Next.js (React-based)
> - **Deployment:** Docker → Vercel (frontend) + Render (backend)

---

## WEEK 7 — Database & Authentication (Days 46–52)

> **Goal:** Build a secure backend with user registration, login, and protected endpoints. After this week, your API will only serve answers to authenticated users.

---

### Day 46 — MySQL Database + SQLAlchemy Setup

**Time:** 4 hrs

**What you're doing:** Connecting your FastAPI app to your local MySQL database and defining the tables (models) for users and conversations.

**Why SQLAlchemy?** You could write raw SQL, but SQLAlchemy gives you an ORM (Object-Relational Mapper) — you interact with Python objects instead of writing SQL strings. This prevents SQL injection, makes migrations easier, and is the industry standard for FastAPI + databases.

**Tasks:**

1. Open MySQL and create a database for this project:

```sql
CREATE DATABASE medical_qa;
```

2. Install the required packages:

```bash
pip install sqlalchemy pymysql python-dotenv
```

> **Why `pymysql`?** SQLAlchemy needs a "driver" to talk to MySQL. `pymysql` is a pure-Python MySQL driver that works out of the box on Windows without needing to compile C extensions.

3. Create a `.env` file in the project root to store your database credentials securely:

```
DATABASE_URL=mysql+pymysql://root:your_password@localhost:3306/medical_qa
SECRET_KEY=generate-a-random-string-here-at-least-32-characters
GROQ_API_KEY=your_existing_groq_key
```

> **Important:** Add `.env` to your `.gitignore` immediately. Never commit secrets to GitHub.

4. Create `src/database/config.py` — the database connection setup:

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from dotenv import load_dotenv
import os

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(bind=engine)

class Base(DeclarativeBase):
    pass

def get_db():
    """Dependency that provides a database session per request.
    FastAPI calls this automatically when an endpoint needs a DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

> **What's happening here?**
>
> - `create_engine` opens a connection pool to your MySQL database.
> - `SessionLocal` is a factory that creates individual database sessions (one per API request).
> - `get_db()` is a FastAPI "dependency" — it gives each request its own session and automatically closes it when done. This prevents connection leaks.

5. Create `src/database/models.py` — your database tables as Python classes:

```python
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Float
from sqlalchemy.orm import relationship
from datetime import datetime, timezone
from src.database.config import Base

class User(Base):
    __tablename__ = "users"

    id         = Column(Integer, primary_key=True, index=True)
    email      = Column(String(255), unique=True, nullable=False, index=True)
    username   = Column(String(100), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationship: one user has many conversations
    conversations = relationship("Conversation", back_populates="user")

class Conversation(Base):
    __tablename__ = "conversations"

    id         = Column(Integer, primary_key=True, index=True)
    user_id    = Column(Integer, ForeignKey("users.id"), nullable=False)
    title      = Column(String(255), default="New Conversation")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    user     = relationship("User", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation", order_by="Message.created_at")

class Message(Base):
    __tablename__ = "messages"

    id              = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=False)
    role            = Column(String(20), nullable=False)   # "user" or "assistant"
    content         = Column(Text, nullable=False)
    sources         = Column(Text, nullable=True)          # JSON string of source chunks
    confidence      = Column(String(20), nullable=True)
    latency_seconds = Column(Float, nullable=True)
    created_at      = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    conversation = relationship("Conversation", back_populates="messages")
```

> **What are these tables?**
>
> - `users` — stores email, username, and hashed password. Never store plain text passwords.
> - `conversations` — groups messages into threads (like how ChatGPT has separate chats in the sidebar). Each conversation belongs to one user.
> - `messages` — individual messages within a conversation. The `role` field marks whether the message is from the user or the assistant. Sources and confidence are stored so you can display them later.

6. Create a small script `src/database/init_db.py` to create the tables:

```python
from src.database.config import engine, Base
from src.database.models import User, Conversation, Message

def init():
    """Create all tables in the database.
    Safe to run multiple times — it won't drop existing tables."""
    Base.metadata.create_all(bind=engine)
    print("Database tables created successfully!")

if __name__ == "__main__":
    init()
```

7. Run it:

```bash
python -m src.database.init_db
```

8. Verify the tables exist in MySQL:

```sql
USE medical_qa;
SHOW TABLES;
DESCRIBE users;
```

**You're done when:** You see `users`, `conversations`, and `messages` tables in your MySQL database.

---

### Day 47 — Password Hashing & JWT Authentication

**Time:** 4 hrs

**What you're doing:** Building the authentication layer. Users will register with a password (which we hash before storing), and log in to receive a JWT token that they include in every subsequent API request.

**Why JWT?** Traditional session-based auth stores sessions on the server, which doesn't scale. JWT (JSON Web Tokens) encodes the user's identity into a signed token that the client stores. The server never needs to remember who's logged in — it just verifies the token's signature. This is the standard approach for API-first applications.

**Tasks:**

1. Install auth packages:

```bash
pip install bcrypt python-jose[cryptography]
```

> **`bcrypt`** — the gold standard for password hashing. It's intentionally slow, making brute-force attacks impractical.
> **`python-jose`** — a library for creating and verifying JWTs. The `[cryptography]` extra gives it fast cryptographic primitives.

2. Create `src/auth/security.py` — the core auth utilities:

```python
from datetime import datetime, timedelta, timezone
from jose import JWTError, jwt
from bcrypt import hashpw, gensalt, checkpw
from dotenv import load_dotenv
import os

load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM  = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60  # Token valid for 1 hour

def hash_password(password: str) -> str:
    """Hash a plain text password using bcrypt."""
    return hashpw(password.encode('utf-8'), gensalt()).decode('utf-8')

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Check if a plain text password matches the stored hash."""
    return checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

def create_access_token(data: dict) -> str:
    """Create a signed JWT token with an expiration time."""
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def decode_access_token(token: str) -> dict | None:
    """Decode and verify a JWT token. Returns None if invalid or expired."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None
```

> **How password hashing works:** When a user registers with password "mypassword123", `hash_password()` converts it into something like `$2b$12$LJ3m4...` — a one-way hash. Even if someone steals your database, they can't reverse the hash back to the original password. When the user logs in, `verify_password()` hashes the provided password again and checks if it matches the stored hash.

> **How JWT works:** When a user logs in successfully, the server creates a token containing `{"sub": "user@email.com", "exp": 1234567890}` and signs it with your `SECRET_KEY`. The client sends this token in every request. The server verifies the signature — if it's valid and not expired, the request is authenticated. No database lookup needed.

3. Create `src/auth/dependencies.py` — the FastAPI dependency that protects endpoints:

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from src.auth.security import decode_access_token
from src.database.config import get_db
from src.database.models import User

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
) -> User:
    """Extract and verify the current user from the JWT token.
    This is used as a dependency on protected endpoints."""
    payload = decode_access_token(token)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    email = payload.get("sub")
    user = db.query(User).filter(User.email == email).first()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )
    return user
```

> **How this protects endpoints:** Any endpoint that includes `current_user: User = Depends(get_current_user)` in its function signature will automatically require a valid JWT token. If the token is missing, expired, or forged, FastAPI returns a 401 Unauthorized response before the endpoint code even runs.

4. Test password hashing manually in a Python shell:

```python
from src.auth.security import hash_password, verify_password

hashed = hash_password("test123")
print(hashed)                                   # $2b$12$...
print(verify_password("test123", hashed))        # True
print(verify_password("wrongpassword", hashed))  # False
```

**You're done when:** Password hashing and JWT creation/verification work correctly in a Python shell.

---

### Day 48 — Auth Endpoints (Register, Login, Me)

**Time:** 4 hrs

**What you're doing:** Building the actual API routes that let users create accounts, log in, and check who they are.

**Tasks:**

1. Create `src/auth/schemas.py` — the request/response shapes for auth endpoints:

```python
from pydantic import BaseModel, EmailStr

class UserRegisterRequest(BaseModel):
    email: str
    username: str
    password: str

class UserLoginRequest(BaseModel):
    email: str
    password: str

class UserResponse(BaseModel):
    id: int
    email: str
    username: str

    class Config:
        from_attributes = True

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
```

2. Create `src/auth/router.py` — the auth routes:

```python
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from src.database.config import get_db
from src.database.models import User
from src.auth.security import hash_password, verify_password, create_access_token
from src.auth.schemas import UserRegisterRequest, UserLoginRequest, UserResponse, TokenResponse
from src.auth.dependencies import get_current_user

router = APIRouter(prefix="/auth", tags=["Authentication"])

@router.post("/register", response_model=UserResponse, status_code=201)
def register(request: UserRegisterRequest, db: Session = Depends(get_db)):
    """Create a new user account."""
    # Check if email already exists
    if db.query(User).filter(User.email == request.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")

    # Check if username already exists
    if db.query(User).filter(User.username == request.username).first():
        raise HTTPException(status_code=400, detail="Username already taken")

    # Create user with hashed password
    user = User(
        email=request.email,
        username=request.username,
        password_hash=hash_password(request.password)
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user

@router.post("/login", response_model=TokenResponse)
def login(request: UserLoginRequest, db: Session = Depends(get_db)):
    """Authenticate and return a JWT token."""
    user = db.query(User).filter(User.email == request.email).first()
    if not user or not verify_password(request.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )

    token = create_access_token(data={"sub": user.email})
    return TokenResponse(access_token=token)

@router.get("/me", response_model=UserResponse)
def get_me(current_user: User = Depends(get_current_user)):
    """Return the currently logged-in user's info. Requires a valid JWT."""
    return current_user
```

> **The flow:**
>
> 1. User calls `POST /auth/register` with email, username, and password → gets back their user info.
> 2. User calls `POST /auth/login` with email and password → gets back a JWT token.
> 3. User includes the token in headers (`Authorization: Bearer <token>`) on every subsequent request.
> 4. User calls `GET /auth/me` with the token → gets back their info, confirming the token works.

3. Register the auth router in your main app. Update `src/api/main.py` — add these lines near the top after creating the `app`:

```python
from src.auth.router import router as auth_router
from fastapi.middleware.cors import CORSMiddleware

# Allow frontend to talk to backend (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
```

> **What is CORS?** When your Next.js frontend (running on `localhost:3000`) tries to call your FastAPI backend (running on `localhost:8000`), the browser blocks it by default for security. CORS (Cross-Origin Resource Sharing) middleware tells the browser "it's okay, let requests from `localhost:3000` through."

4. Start the API and test the full auth flow:

```bash
uvicorn src.api.main:app --reload
```

5. Open `http://localhost:8000/docs` and test manually:
   - Hit `POST /auth/register` with a test email, username, and password.
   - Hit `POST /auth/login` with the same credentials → copy the `access_token` from the response.
   - Click the green "Authorize" button at the top of the Swagger page, paste your token, and hit `GET /auth/me`.

**You're done when:** You can register, login, and access `/auth/me` with a valid token through the Swagger UI.

---

### Day 49 — Conversation History Endpoints

**Time:** 4 hrs

**What you're doing:** Building endpoints to create conversations, save messages, and retrieve past chat history — all tied to the logged-in user.

**Tasks:**

1. Create `src/conversations/schemas.py`:

```python
from pydantic import BaseModel
from datetime import datetime

class ConversationCreate(BaseModel):
    title: str = "New Conversation"

class ConversationResponse(BaseModel):
    id: int
    title: str
    created_at: datetime

    class Config:
        from_attributes = True

class MessageResponse(BaseModel):
    id: int
    role: str
    content: str
    sources: str | None
    confidence: str | None
    latency_seconds: float | None
    created_at: datetime

    class Config:
        from_attributes = True

class ConversationDetailResponse(BaseModel):
    id: int
    title: str
    created_at: datetime
    messages: list[MessageResponse]

    class Config:
        from_attributes = True
```

2. Create `src/conversations/router.py`:

```python
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from src.database.config import get_db
from src.database.models import User, Conversation, Message
from src.auth.dependencies import get_current_user
from src.conversations.schemas import (
    ConversationCreate, ConversationResponse, ConversationDetailResponse
)

router = APIRouter(prefix="/conversations", tags=["Conversations"])

@router.post("/", response_model=ConversationResponse, status_code=201)
def create_conversation(
    request: ConversationCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Start a new conversation thread."""
    conv = Conversation(user_id=current_user.id, title=request.title)
    db.add(conv)
    db.commit()
    db.refresh(conv)
    return conv

@router.get("/", response_model=list[ConversationResponse])
def list_conversations(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """List all conversations for the logged-in user, newest first."""
    return (
        db.query(Conversation)
        .filter(Conversation.user_id == current_user.id)
        .order_by(Conversation.created_at.desc())
        .all()
    )

@router.get("/{conversation_id}", response_model=ConversationDetailResponse)
def get_conversation(
    conversation_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get a conversation with all its messages."""
    conv = (
        db.query(Conversation)
        .filter(Conversation.id == conversation_id, Conversation.user_id == current_user.id)
        .first()
    )
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conv

@router.delete("/{conversation_id}", status_code=204)
def delete_conversation(
    conversation_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete a conversation and all its messages."""
    conv = (
        db.query(Conversation)
        .filter(Conversation.id == conversation_id, Conversation.user_id == current_user.id)
        .first()
    )
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    db.delete(conv)
    db.commit()
```

3. Register the conversations router in `src/api/main.py`:

```python
from src.conversations.router import router as conversations_router

app.include_router(conversations_router)
```

4. Test through Swagger:
   - Create a conversation (`POST /conversations/`).
   - List your conversations (`GET /conversations/`).
   - Get a specific conversation (`GET /conversations/{id}`).

**You're done when:** You can create, list, view, and delete conversations through the Swagger UI while authenticated.

---

### Day 50 — Protect the /ask Endpoint + Save Messages

**Time:** 4 hrs

**What you're doing:** The existing `/ask` endpoint is completely open — anyone can call it. Today you lock it behind authentication and save every question + answer into the database as part of a conversation.

**Tasks:**

1. Refactor the `/ask` endpoint in `src/api/main.py` to require auth and save messages:

```python
import json
from fastapi import Depends
from sqlalchemy.orm import Session
from src.database.config import get_db
from src.database.models import User, Conversation, Message
from src.auth.dependencies import get_current_user

class AskRequest(BaseModel):
    question: str
    conversation_id: int | None = None  # Optional: attach to existing conversation

@app.post("/ask", response_model=QuestionResponse)
def ask(
    request: AskRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if len(request.question.strip()) < 5:
        raise HTTPException(status_code=400, detail="Question too short")

    # Create a new conversation if none provided
    if request.conversation_id:
        conv = db.query(Conversation).filter(
            Conversation.id == request.conversation_id,
            Conversation.user_id == current_user.id
        ).first()
        if not conv:
            raise HTTPException(status_code=404, detail="Conversation not found")
    else:
        # Auto-title using the first few words of the question
        title = request.question[:50] + ("..." if len(request.question) > 50 else "")
        conv = Conversation(user_id=current_user.id, title=title)
        db.add(conv)
        db.commit()
        db.refresh(conv)

    start = time.time()

    if USE_GROQ:
        chunks = retrieve_and_rerank(request.question, collection, embedder, reranker)
        answer = answer_with_groq(request.question, chunks)
        sources = chunks
    else:
        result = answer_question(request.question, model, tokenizer, embedder, collection, reranker)
        answer = result['answer']
        sources = result['sources']

    latency = round(time.time() - start, 2)
    confidence = estimate_confidence(request.question, sources, embedder)

    # Save user message
    user_msg = Message(
        conversation_id=conv.id,
        role="user",
        content=request.question
    )
    db.add(user_msg)

    # Save assistant message
    assistant_msg = Message(
        conversation_id=conv.id,
        role="assistant",
        content=answer,
        sources=json.dumps(sources),
        confidence=confidence,
        latency_seconds=latency
    )
    db.add(assistant_msg)
    db.commit()

    return QuestionResponse(
        answer=answer,
        sources=sources,
        confidence=confidence,
        latency_seconds=latency
    )
```

> **What changed?**
>
> - The endpoint now requires a valid JWT token (`current_user: User = Depends(get_current_user)`).
> - Every question and answer is saved as `Message` objects in the database, tied to a `Conversation`.
> - If no `conversation_id` is provided, a new conversation is created automatically.

2. Also update the `QuestionResponse` schema to include the `conversation_id`:

```python
class QuestionResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: str
    latency_seconds: float
    conversation_id: int
```

3. Test the full flow in Swagger:
   - Login to get a token.
   - Call `/ask` with a medical question.
   - Call `GET /conversations/` — you should see a new conversation.
   - Call `GET /conversations/{id}` — you should see the user question and assistant answer as messages.

**You're done when:** Questions and answers are being saved to MySQL and you can retrieve full conversation histories.

---

### Day 51 — Rate Limiting + Input Validation

**Time:** 3 hrs

**What you're doing:** Preventing users from abusing your API. Without rate limiting, one user could send 1000 requests per minute and burn through your entire Groq API quota.

**Tasks:**

1. Install the rate limiting package:

```bash
pip install slowapi
```

2. Add rate limiting to `src/api/main.py`:

```python
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi.responses import JSONResponse

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request, exc):
    return JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded. Please try again later."}
    )
```

3. Apply the rate limit to your `/ask` endpoint by adding the decorator:

```python
@app.post("/ask", response_model=QuestionResponse)
@limiter.limit("10/hour")  # Max 10 questions per hour per IP
def ask(request: AskRequest, ...):
    ...
```

> **Why 10/hour?** This is a reasonable starting point. Groq's free tier has limits, and most legitimate users won't ask more than 10 medical questions in an hour. You can adjust this later.

4. Add input validation — update the `AskRequest` model to enforce limits:

```python
from pydantic import BaseModel, field_validator

class AskRequest(BaseModel):
    question: str
    conversation_id: int | None = None

    @field_validator('question')
    @classmethod
    def validate_question(cls, v):
        v = v.strip()
        if len(v) < 5:
            raise ValueError("Question must be at least 5 characters")
        if len(v) > 1000:
            raise ValueError("Question must be under 1000 characters")
        return v
```

5. Test rate limiting by sending 11 quick requests — the 11th should return a 429 status code.

**You're done when:** Rate limiting is active and returns a 429 error when exceeded.

---

### Day 52 — Backend Tests + CI/CD Update

**Time:** 3 hrs

**What you're doing:** Updating your test suite to test the new auth flow. Your existing tests in `tests/test_api.py` will break because `/ask` now requires authentication. Fix them.

**Tasks:**

1. Update `tests/test_api.py` to handle authentication:

```python
import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

# Helper: create a test user and get a token
def get_auth_token():
    """Register a test user and return a valid JWT token."""
    # Try to register (might already exist from a previous run)
    client.post("/auth/register", json={
        "email": "test@test.com",
        "username": "testuser",
        "password": "testpass123"
    })
    # Login
    response = client.post("/auth/login", json={
        "email": "test@test.com",
        "password": "testpass123"
    })
    return response.json()["access_token"]

def auth_headers():
    token = get_auth_token()
    return {"Authorization": f"Bearer {token}"}

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_register_new_user():
    response = client.post("/auth/register", json={
        "email": "newuser@test.com",
        "username": "newuser",
        "password": "password123"
    })
    assert response.status_code in [201, 400]  # 400 if already exists

def test_login():
    response = client.post("/auth/login", json={
        "email": "test@test.com",
        "password": "testpass123"
    })
    assert response.status_code == 200
    assert "access_token" in response.json()

def test_ask_without_auth():
    """Verify that /ask rejects unauthenticated requests."""
    response = client.post("/ask", json={"question": "What are symptoms of diabetes?"})
    assert response.status_code == 401

def test_ask_with_auth():
    response = client.post(
        "/ask",
        json={"question": "What are symptoms of diabetes?"},
        headers=auth_headers()
    )
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "conversation_id" in data

def test_conversation_history():
    headers = auth_headers()
    # Ask a question (creates a conversation)
    ask_response = client.post(
        "/ask",
        json={"question": "What causes high blood pressure?"},
        headers=headers
    )
    conv_id = ask_response.json()["conversation_id"]

    # Retrieve the conversation
    conv_response = client.get(f"/conversations/{conv_id}", headers=headers)
    assert conv_response.status_code == 200
    messages = conv_response.json()["messages"]
    assert len(messages) >= 2  # At least 1 user + 1 assistant message
```

2. Update your `.github/workflows/test.yml` to include new dependencies:

```yaml
- name: Install dependencies
  run: |
    pip install fastapi httpx pytest pydantic uvicorn
    pip install sqlalchemy pymysql python-dotenv
    pip install bcrypt python-jose[cryptography] slowapi
```

> **Note:** For CI, you may need to use SQLite instead of MySQL (since GitHub Actions runners don't have MySQL by default). You can add a conditional in `config.py` that uses SQLite when the `DATABASE_URL` env var isn't set.

3. Run tests locally:

```bash
pytest tests/ -v
```

4. Fix any failures, then push and confirm CI passes.

**You're done when:** All tests pass locally and in GitHub Actions.

---

## WEEK 8 — Next.js Frontend Setup (Days 53–59)

> **Goal:** Build the frontend shell — project setup, auth pages, and the main layout. By the end of this week, users can register, log in, and see the main chat layout in their browser.

---

### Day 53 — Initialize the Next.js Project

**Time:** 3 hrs

**What you're doing:** Setting up a Next.js frontend project inside your repository. This will be a separate folder (`frontend/`) that talks to your FastAPI backend.

**Tasks:**

1. Create the Next.js project:

```bash
npx -y create-next-app@latest frontend --typescript --tailwind --eslint --app --src-dir --no-import-alias
```

> **Why these flags?**
>
> - `--typescript` — type safety, catches bugs early.
> - `--tailwind` — utility-first CSS framework for rapid styling (we'll keep the colors clean and minimal as you requested — no purple/blue gradients).
> - `--app` — uses Next.js 14+ App Router (the modern approach).
> - `--src-dir` — puts code in `frontend/src/` for clean organization.

2. Navigate into the frontend directory and start it:

```bash
cd frontend
npm run dev
```

3. Open `http://localhost:3000` — you should see the default Next.js welcome page.

4. Install additional packages you'll need:

```bash
npm install axios
```

> **`axios`** — a popular HTTP client for making API calls to your FastAPI backend. It handles headers, JSON parsing, and error handling more cleanly than the native `fetch` API.

5. Create `frontend/src/lib/api.ts` — a centralized API client:

```typescript
import axios from "axios";

const api = axios.create({
  baseURL: "http://localhost:8000",
  headers: { "Content-Type": "application/json" },
});

// Automatically attach JWT token to every request
api.interceptors.request.use((config) => {
  const token = localStorage.getItem("token");
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// Handle 401 errors globally (redirect to login)
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem("token");
      window.location.href = "/login";
    }
    return Promise.reject(error);
  },
);

export default api;
```

> **What are interceptors?** Instead of manually adding `Authorization: Bearer <token>` to every single API call, the request interceptor does it automatically. The response interceptor catches 401 errors and redirects to login — so if a token expires mid-session, the user is seamlessly sent to re-authenticate.

**You're done when:** Next.js dev server is running on `localhost:3000` and the API client is configured.

---

### Day 54 — Authentication Pages (Login + Register)

**Time:** 4 hrs

**What you're doing:** Building the login and registration pages. These are the first pages a user will see.

**Design principle:** Keep it clean and minimal. No AI-gradient decorations. Think professional medical SaaS — clean whites, soft grays, clear typography.

**Tasks:**

1. Create `frontend/src/app/login/page.tsx` — the login page. It should have:
   - An email input field.
   - A password input field.
   - A "Log In" button that calls `POST /auth/login`.
   - On success, store the returned JWT token in `localStorage` and redirect to the main page (`/`).
   - On failure, show the error message from the API.
   - A link to the registration page ("Don't have an account? Register").

2. Create `frontend/src/app/register/page.tsx` — the registration page. It should have:
   - Email, username, and password input fields.
   - A "Create Account" button that calls `POST /auth/register`.
   - On success, redirect to the login page.
   - On failure, show the error message (e.g., "Email already registered").
   - A link back to the login page ("Already have an account? Log in").

3. Create `frontend/src/context/AuthContext.tsx` — a React context to manage auth state across the entire app:
   - Store the current user object (from `/auth/me`) and the loading state.
   - Provide `login()`, `register()`, and `logout()` functions.
   - On page load, check if a token exists in `localStorage` and call `/auth/me` to verify it's still valid.
   - Export a `useAuth()` hook for easy access.

4. Create a protected route wrapper `frontend/src/components/ProtectedRoute.tsx`:
   - If the user is not authenticated, redirect to `/login`.
   - If the user is authenticated, render the child components.
   - Show a loading spinner while the auth check is in progress.

5. Test the flow:
   - Open `localhost:3000/register` → create an account.
   - Open `localhost:3000/login` → log in.
   - Verify you're redirected to the main page.
   - Open browser DevTools → Application → Local Storage → confirm the token is stored.

**You're done when:** You can register and log in through the browser, and the auth state persists across page refreshes.

---

### Day 55 — Main Layout + Chat Interface

**Time:** 4 hrs

**What you're doing:** Building the main chat interface where authenticated users can ask medical questions and see responses.

**Layout:** Think ChatGPT's layout — a sidebar on the left listing conversations, and a main chat area on the right.

**Tasks:**

1. Create the main layout `frontend/src/app/(dashboard)/layout.tsx`:
   - A sidebar (left panel, ~280px wide) showing the user's conversation list.
   - A "New Chat" button at the top of the sidebar.
   - The user's name/email at the bottom of the sidebar with a "Logout" button.
   - The main content area (right panel) where the chat lives.
   - Wrap the entire layout in your `ProtectedRoute` component.

2. Create `frontend/src/app/(dashboard)/page.tsx` — the main chat page:
   - A scrollable message area showing the conversation.
   - Each message shows the role (user or assistant) with different styling.
   - Assistant messages should render markdown (install `react-markdown`).
   - A fixed input bar at the bottom with a text input and a "Send" button.

3. Wire it up to the API:
   - When the user types a question and hits Send, call `POST /ask` with the question and `conversation_id`.
   - While waiting for the response, show a loading indicator.
   - When the response arrives, append both the user message and assistant answer to the chat.
   - Display the sources in a collapsible section below the answer.
   - Display the confidence badge (high/medium/low) next to the answer.

4. Add the medical disclaimer — show it prominently at the top of every new conversation.

**You're done when:** You can log in, ask a medical question, and see the answer with sources displayed in a clean chat interface.

---

### Day 56 — Conversation Sidebar + History

**Time:** 4 hrs

**What you're doing:** Making the sidebar functional — users can create new conversations, switch between them, and see their full history.

**Tasks:**

1. Populate the sidebar by calling `GET /conversations/` on page load. Show each conversation's title and date.

2. When a user clicks a conversation in the sidebar, call `GET /conversations/{id}` and render all its messages in the chat area.

3. Implement the "New Chat" button:
   - Clear the current chat area.
   - The next question sent will create a new conversation (no `conversation_id` in the request).

4. Implement "Delete Conversation":
   - Add a small delete icon/button on each sidebar item.
   - Call `DELETE /conversations/{id}` and remove it from the list.

5. Add visual polish:
   - Highlight the currently active conversation in the sidebar.
   - Show "No conversations yet" if the list is empty.
   - Sort conversations by most recent first.

**You're done when:** The sidebar shows conversation history, you can switch between conversations, and new chats work correctly.

---

### Day 57 — Feedback System + UI Polish

**Time:** 4 hrs

**What you're doing:** Adding a thumbs up/thumbs down feedback mechanism to assistant responses. This is the foundation of a data flywheel — you can use this feedback data to improve the model later via RLHF.

**Tasks:**

1. Add a `feedback` table to the database in `src/database/models.py`:

```python
class Feedback(Base):
    __tablename__ = "feedback"

    id         = Column(Integer, primary_key=True, index=True)
    message_id = Column(Integer, ForeignKey("messages.id"), nullable=False)
    user_id    = Column(Integer, ForeignKey("users.id"), nullable=False)
    rating     = Column(String(10), nullable=False)  # "up" or "down"
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
```

2. Create a feedback endpoint `POST /feedback`:

```python
@app.post("/feedback")
def submit_feedback(
    message_id: int,
    rating: str,  # "up" or "down"
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    feedback = Feedback(
        message_id=message_id,
        user_id=current_user.id,
        rating=rating
    )
    db.add(feedback)
    db.commit()
    return {"status": "ok"}
```

3. In the frontend, add thumbs up/thumbs down icons below each assistant message. When clicked, call the feedback endpoint.

4. UI polish pass:
   - Add smooth transitions when switching conversations.
   - Add a loading skeleton while messages are being fetched.
   - Make the layout responsive (sidebar collapses on mobile).
   - Add proper error toasts/notifications instead of raw alerts.

5. Run the database migration to create the new `feedback` table:

```bash
python -m src.database.init_db
```

**You're done when:** Users can rate responses with thumbs up/down, and the feedback is saved to the database.

---

### Day 58 — Responsive Design + Mobile Support

**Time:** 3 hrs

**What you're doing:** Making the website work well on phones and tablets. A real-world website must be responsive.

**Tasks:**

1. Make the sidebar toggle-able on mobile:
   - On screens smaller than 768px, hide the sidebar by default.
   - Add a hamburger menu button that slides the sidebar in and out.

2. Adjust the chat input area:
   - On mobile, the input bar should be fixed at the bottom of the screen.
   - The message area should be scrollable above it.

3. Test on multiple screen sizes using browser DevTools:
   - Desktop (1920px)
   - Tablet (768px)
   - Mobile (375px)

4. Add `<meta>` viewport tags and proper SEO metadata in `frontend/src/app/layout.tsx`.

**You're done when:** The website looks and works well on desktop, tablet, and mobile screens.

---

### Day 59 — Week 8 Review + Integration Testing

**Time:** 3 hrs

**What you're doing:** End-to-end testing of the full flow — register, login, ask questions, switch conversations, give feedback, and logout.

**Tasks:**

1. Do a full end-to-end walkthrough:
   - Register a brand new account.
   - Log in.
   - Ask 5 different medical questions.
   - Check that conversations appear in the sidebar.
   - Switch between conversations.
   - Give thumbs up/down feedback on answers.
   - Log out and verify you're redirected to the login page.
   - Log back in and confirm your conversation history is preserved.

2. Fix any bugs found during testing.

3. Update `requirements.txt` with all new Python dependencies.

4. Commit everything and push to GitHub.

**You're done when:** The full flow works without bugs end-to-end.

---

## WEEK 9 — Docker + Deployment (Days 60–66)

> **Goal:** Containerize everything with Docker and deploy to the cloud. After this week, your website is live on the internet.

---

### Day 60 — Dockerize the FastAPI Backend

**Time:** 4 hrs

**What you're doing:** Creating a Docker container for your backend so it runs identically on any machine.

**Tasks:**

1. Create `Dockerfile` in the project root:

```dockerfile
FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY .env .env

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

2. Create a `.dockerignore` file:

```
medqa_env/
__pycache__/
.git/
node_modules/
frontend/
notebooks/
data/raw/
models/
mlruns/
*.pyc
```

3. Build and test:

```bash
docker build -t medical-qa-backend .
docker run -p 8000:8000 --env-file .env medical-qa-backend
```

4. Verify `http://localhost:8000/health` returns `{"status": "ok"}`.

**You're done when:** The backend runs correctly inside a Docker container.

---

### Day 61 — Docker Compose (Backend + MySQL)

**Time:** 3 hrs

**What you're doing:** Using Docker Compose to run the backend and a MySQL container together with a single command.

**Tasks:**

1. Create `docker-compose.yml` in the project root:

```yaml
version: "3.8"

services:
  db:
    image: mysql:8.0
    environment:
      MYSQL_ROOT_PASSWORD: rootpassword
      MYSQL_DATABASE: medical_qa
    ports:
      - "3307:3306" # 3307 to avoid conflict with your local MySQL
    volumes:
      - mysql_data:/var/lib/mysql

  backend:
    build: .
    ports:
      - "8000:8000"
    environment:
      DATABASE_URL: mysql+pymysql://root:rootpassword@db:3306/medical_qa
      SECRET_KEY: ${SECRET_KEY}
      GROQ_API_KEY: ${GROQ_API_KEY}
      USE_GROQ: "true"
    depends_on:
      - db

volumes:
  mysql_data:
```

2. Run it:

```bash
docker-compose up --build
```

3. Test the full API through `http://localhost:8000/docs`.

**You're done when:** `docker-compose up` starts both MySQL and the backend, and the API works.

---

### Day 62 — Deploy the Backend to Render

**Time:** 4 hrs

**What you're doing:** Deploying your FastAPI backend to Render's free tier so it's accessible on the internet.

> **Important trade-off about the database:** Render's free tier does not include MySQL. For the deployed version, you have two options:
>
> 1. Use Render's free PostgreSQL (change `pymysql` to `psycopg2` — SQLAlchemy makes this a one-line change in the connection string).
> 2. Use a free MySQL service like TiDB Cloud Serverless.
>    We'll cover the exact steps on this day.

**Tasks:**

1. Create a free account on render.com.
2. Create a new "Web Service" connected to your GitHub repo.
3. Set the build command: `pip install -r requirements.txt`
4. Set the start command: `uvicorn src.api.main:app --host 0.0.0.0 --port $PORT`
5. Add environment variables in Render's dashboard: `DATABASE_URL`, `SECRET_KEY`, `GROQ_API_KEY`, `USE_GROQ`.
6. Deploy and test the `/health` endpoint at your Render URL.

**You're done when:** Your backend API is live at `https://your-app.onrender.com`.

---

### Day 63 — Deploy the Frontend to Vercel

**Time:** 3 hrs

**What you're doing:** Deploying your Next.js frontend to Vercel (which is built by the same team that created Next.js — best possible hosting for it, free forever for personal projects).

**Tasks:**

1. Create a free account on vercel.com.
2. Import your GitHub repo and point it to the `frontend/` directory.
3. Set the environment variable `NEXT_PUBLIC_API_URL` to your Render backend URL.
4. Update `frontend/src/lib/api.ts` to use the environment variable instead of `localhost`:

```typescript
const api = axios.create({
  baseURL: process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000",
  headers: { "Content-Type": "application/json" },
});
```

5. Deploy. Vercel will auto-deploy on every push to `main`.
6. Update CORS in `src/api/main.py` to include your Vercel domain:

```python
allow_origins=["http://localhost:3000", "https://your-app.vercel.app"],
```

7. Test the full flow on the live site.

**You're done when:** The live website is accessible at `https://your-app.vercel.app` and can register users, login, and ask questions.

---

### Day 64 — Custom Domain + SSL (Optional)

**Time:** 2 hrs

**What you're doing:** If you own a domain name, connecting it to your Vercel deployment so your site is at `yourdomain.com` instead of `your-app.vercel.app`. If you don't have a domain, skip this day.

**Tasks:**

1. Buy a domain from Namecheap, Google Domains, or Cloudflare (~$10/year for a `.com`).
2. In Vercel, go to your project → Settings → Domains → Add your domain.
3. Update DNS records as Vercel instructs (usually an A record or CNAME).
4. Vercel auto-provisions an SSL certificate (HTTPS) for free.
5. Update CORS origins in FastAPI to include your custom domain.

**You're done when:** Your website is live at your custom domain with HTTPS.

---

### Day 65 — Logging + Error Monitoring

**Time:** 3 hrs

**What you're doing:** Adding proper logging to the backend so you can debug production issues. Right now, if something breaks on the live site, you have no way to know what happened.

**Tasks:**

1. Replace all `print()` statements in your backend with Python's `logging` module:

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Instead of: print("Loading models...")
# Use: logger.info("Loading models...")
```

2. Add structured logging to key events:
   - `logger.info(f"User {user.email} asked: {question[:50]}")` — log every question.
   - `logger.warning(f"Rate limit hit by {request.client.host}")` — log rate limit events.
   - `logger.error(f"Groq API error: {str(e)}")` — log API failures.

3. Add a global error handler in FastAPI:

```python
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled error: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal error occurred. Please try again."}
    )
```

**You're done when:** All key events are logged and errors are caught gracefully.

---

### Day 66 — Week 9 Review + Full E2E Test on Production

**Time:** 3 hrs

**Tasks:**

1. Run the complete end-to-end flow on the live production site:
   - Register, login, ask 10 questions, switch conversations, give feedback, logout.
2. Check Render logs for any errors.
3. Share the live URL with someone and watch them use it. Note pain points.
4. Fix any bugs found.

**You're done when:** The production site is stable and tested.

---

## WEEK 10 — Final Polish + Portfolio (Days 67–73)

> **Goal:** Polish everything and make this portfolio-ready.

---

### Day 67 — Design Inspiration Pass

**Time:** 4 hrs

**What you're doing:** Applying the specific design inspiration you chose for the website (you mentioned you'll share this later). This is the day to overhaul the visual design.

**Tasks:**

1. Study your chosen design inspiration — note the color palette, typography, spacing, and component styles.
2. Update the frontend's global styles and Tailwind config to match.
3. Apply the design to all pages: login, register, chat, sidebar.
4. Ensure consistency across all components.

**You're done when:** The website matches your design inspiration and feels polished.

---

### Day 68 — Update the README + Architecture Diagram

**Time:** 3 hrs

**Tasks:**

1. Update the README to reflect the full-stack architecture:
   - Add a new architecture diagram showing the frontend → backend → database flow.
   - Update the "Setup" section with instructions for running both frontend and backend.
   - Add a "Tech Stack" section listing all technologies used.
   - Update the "What It Does" section to mention user auth, conversation history, and feedback.

2. Update the Mermaid architecture diagram to include the new components:

```
User → Next.js Frontend → FastAPI Backend → MySQL Database
                                          → RAG Pipeline → ChromaDB
                                          → Groq API
```

**You're done when:** README accurately describes the full-stack application.

---

### Day 69 — Security Audit

**Time:** 3 hrs

**What you're doing:** Reviewing your application for common security vulnerabilities.

**Checklist:**

1. **Passwords:** Verify all passwords are hashed with bcrypt. Never stored in plain text.
2. **SQL Injection:** Confirm you're using SQLAlchemy ORM (parameterized queries) everywhere and never building raw SQL strings.
3. **JWT Security:** Verify tokens expire properly. Test with an expired token to confirm it's rejected.
4. **CORS:** Make sure `allow_origins` only includes your actual domains, not `"*"`.
5. **Rate Limiting:** Confirm rate limiting works on production.
6. **Environment Variables:** Verify `.env` is gitignored and secrets are only in Render/Vercel dashboards.
7. **Input Validation:** Confirm all user inputs are validated (question length, email format, etc.).
8. **HTTPS:** Verify the live site uses HTTPS (Vercel handles this automatically).

**You're done when:** All 8 checks pass.

---

### Day 70 — Record Demo Video + LinkedIn Post

**Time:** 3 hrs

**Tasks:**

1. Download Loom (free screen recorder).
2. Record a 2–3 minute demo:
   - 0:00–0:20: "I built a medical QA assistant — here's the live site." (show the login page)
   - 0:20–0:40: Register a new account and log in.
   - 0:40–1:30: Ask 3 medical questions. Show answers with sources and confidence.
   - 1:30–1:50: Show conversation history in the sidebar. Switch between chats.
   - 1:50–2:10: Show the thumbs up/down feedback system.
   - 2:10–2:40: "Under the hood: fine-tuned Phi-3, RAG with re-ranking, FastAPI with JWT auth, MySQL database, Next.js frontend."
   - 2:40–3:00: "Here's the GitHub with full code, CI/CD, and Docker support."
3. Add the Loom link to your README.
4. Post on LinkedIn.

**You're done when:** Demo video is live and LinkedIn post is up.

---

### Day 71 — Code Cleanup + Documentation

**Time:** 3 hrs

**Tasks:**

1. Go through every file. Remove debug `print()` statements, clean up commented-out code.
2. Add docstrings to every Python function that doesn't have one.
3. Add JSDoc comments to key TypeScript functions.
4. Make sure every notebook has a markdown cell at the top explaining what it does.
5. Create a `docs/API_REFERENCE.md` documenting all API endpoints with example requests and responses.

**You're done when:** The codebase is clean and well-documented.

---

### Day 72 — Final Polish + Resume Bullet

**Time:** 3 hrs

**Tasks:**

1. Do one final end-to-end test on production.
2. Write your updated resume bullet:

```
Built and deployed a full-stack medical Q&A web application — fine-tuned Phi-3
Mini (3.8B) on 100k+ doctor-patient conversations using QLoRA, implemented a RAG
pipeline with BioMedical embeddings and cross-encoder re-ranking, built a secure
FastAPI backend with JWT authentication, MySQL database, and rate limiting,
designed a Next.js frontend with conversation history and user feedback, and
deployed with Docker on Vercel + Render with CI/CD via GitHub Actions.
```

3. Add it to your resume and LinkedIn.

**You're done when:** Resume bullet is updated and you're proud of it.

---

### Day 73 — Ship It 🚀

**Time:** 2 hrs

**Tasks:**

1. Share the live URL with 3–5 people. Collect feedback.
2. Create a GitHub release tag (`v2.0.0 — Full-Stack Release`).
3. Pin the repo on your GitHub profile.
4. Celebrate. You built a real, production-quality AI product from scratch.

> 🎉 **Project complete.** You didn't just train a model — you shipped a product. That's what AI Engineers do.

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
