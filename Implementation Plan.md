# 🏥 Medical QA AI Assistant — Complete Day-by-Day Plan
### Tailored for: Beginner | Learning Focus | 20+ hrs/week

---

> **How to use this plan:**
> Each day has a clear goal, step-by-step tasks, and what "done" looks like.
> Daily time estimate: ~3–4 hours. You have buffer time each week — use it when you get stuck (you will, and that's normal).
> Keep a file called `LEARNINGS.md` in your repo. Write 3–5 sentences every single day about what you learned or what confused you.

---

## 📅 PRE-WEEK — Foundations (Days 1–5)
> **Goal:** Understand the concepts before writing a single line of project code. This week prevents weeks of confusion later.

---

### Day 1 — How Neural Networks Actually Work
**Time:** 3–4 hrs

**Tasks:**
1. Watch 3Blue1Brown "Neural Networks" series on YouTube — Episodes 1, 2, and 3 only (free, ~1 hr total). This is the best visual explanation that exists.
2. Read this one article: "What is a Large Language Model?" on HuggingFace's blog (free, search it).
3. Write in `LEARNINGS.md`: What is a weight? What does "training" mean in your own words?

**You're done when:** You can explain to yourself what a neural network is and roughly how it learns.

---

### Day 2 — What Embeddings Are (Critical for RAG)
**Time:** 3 hrs

**Tasks:**
1. Search YouTube: "Word Embeddings explained visually" — watch any 10-minute video.
2. Open a Python notebook (Google Colab is free, no setup needed) and run this:
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
sentences = ["I have a headache", "My head hurts", "I love pizza"]
embeddings = model.encode(sentences)
print(embeddings.shape)
# Try to find which two sentences are most similar
from sklearn.metrics.pairwise import cosine_similarity
print(cosine_similarity([embeddings[0]], [embeddings[1]]))  # headache vs head hurts
print(cosine_similarity([embeddings[0]], [embeddings[2]]))  # headache vs pizza
```
3. Write in `LEARNINGS.md`: Why are embeddings useful? What does "similar meaning = similar vector" actually mean?

**You're done when:** You understand why two sentences with different words can be "similar" mathematically.

---

### Day 3 — What Fine-tuning Means
**Time:** 3 hrs

**Tasks:**
1. Read this analogy and really think about it: A base LLM is like a very smart person who has read the entire internet. Fine-tuning is like giving that person a 6-month internship at a hospital. They learn the specific language, format, and behavior you want — without forgetting everything they knew before.
2. Search YouTube: "Fine-tuning LLMs explained" — watch any video under 15 minutes.
3. Search "What is LoRA fine-tuning" — read one blog post. Key idea: instead of changing all billions of weights, you only train a tiny set of new weights. That's why it fits on your GPU.
4. Write in `LEARNINGS.md`: What is the difference between a base model and a fine-tuned model? Why does QLoRA exist?

**You're done when:** You understand why we fine-tune instead of just prompting the model.

---

### Day 4 — What RAG Is and Why It Exists
**Time:** 3 hrs

**Tasks:**
1. Understand the core problem: LLMs are frozen in time. They can't know what's in a new document you wrote yesterday. RAG solves this by looking up relevant documents at query time and feeding them to the model as context.
2. Watch YouTube: "RAG explained" — any video under 15 minutes.
3. Draw the RAG pipeline on paper (yes, physically draw it):
   - User asks question → Question gets embedded → Vector search finds similar docs → Top docs go into the prompt → Model answers using those docs
4. Write in `LEARNINGS.md`: What problem does RAG solve that fine-tuning doesn't?

**You're done when:** You can draw and explain the RAG pipeline from memory.

---

### Day 5 — HuggingFace Ecosystem Orientation
**Time:** 3–4 hrs

**Tasks:**
1. Create a free account on HuggingFace (huggingface.co).
2. Search for "Phi-3-mini-4k-instruct" on HuggingFace. Read the model card. Don't understand everything — just get familiar with how model pages look.
3. Search for "bigbio/med_qa" on HuggingFace Datasets. Look at the dataset card and click "Dataset Viewer" to see what the data actually looks like.
4. Run this in Google Colab:
```python
from datasets import load_dataset
dataset = load_dataset("bigbio/med._qa", trust_remote_code=True)
print(dataset)
print(dataset['train'][0])  # Look at one example
```
5. Write in `LEARNINGS.md`: What does one training example look like in med_qa? What fields does it have?

**You're done when:** You have a HuggingFace account and you've seen the actual training data.

---

## 📅 WEEK 1 — Environment Setup & GitHub (Days 6–12)
> **Goal:** Get your full local environment working without errors. This is the most frustrating week — that's normal.
.
-.--
.
#.## Day 6 — Python Environment Setup
**.Time:** 3–4 hrs
.
*.*Tasks:**
1.. Install Python 3.11 from python.org (not 3.12 — some ML libraries have issues with it).
2. .Install VS Code and the Python extension.
3. C.reate a virtual environment:
```ba.sh
python. -m venv medqa_env
# On Wi.ndows:
medqa_en.v\Scripts\activate
# On Mac/.Linux:
source med.qa_env/bin/activate
```.
4. I.nstall base packages:
```ba.sh
pip in.stall jupyter notebook ipykernel
python .-m ipykernel install --user --name=medqa_env
```.
5. O.pen VS Code, select the `medqa_env` kernel, run `print("hello world")` in a notebook.
.
*.*You're done when:** A Jupyter notebook runs inside your virtual environment in VS Code.
.
-.--
.
#.## Day 7 — PyTorch + CUDA Setup
**.Time:** 4 hrs (this day often takes longer — budget extra time)
.
*.*Tasks:**
1.. Check if CUDA is installed: Open CMD and type `nvidia-smi`. You should see your RTX 4050 lis.ted with a CUDA version.
2. G.o to pytorch.org → "Get Started" → select your OS, pip, and your CUDA version → copy the i.nstall command. Run it.
3. Ver.ify PyTorch sees your GPU:
```pyth.on
import t.orch
print(tor.ch.__version__)
print(torc.h.cuda.is_available())  # Must print True
print(torch..cuda.get_device_name(0))  # Should show RTX 4050
```.
4. I.f `cuda.is_available()` returns False — don't panic. Google "PyTorch CUDA not available RTX 4.050" and follow the fixes. This is a common issue.
.
*.*You're done when:** `torch.cuda.is_available()` returns `True`.
.
-.--
.
#.## Day 8 — Install All Project Libraries
**.Time:** 3 hrs
.
*.*Tasks:**
1.. With your virtual env activated, install everything:
```.bash
pip .install transformers peft trl bitsandbytes datasets
pip i.nstall chromadb sentence-transformers mlflow
pip in.stall fastapi uvicorn gradio
pip install dvc accelerate huggingface_hub
```
2. Test each one imports without error:
```python
import transformers, peft, trl, datasets
import chromadb, sentence_transformers, mlflow
import .fastapi, gradio
print("All imports successful!")
```
3. If any fail, fix them one by one. Google the exact error message.
.
*.*You're done when:** All imports work without errors.
.
-.--
.
#.## Day 9 — GitHub Repository Setup
**.Time:** 3 hrs
.
*.*Tasks:**
1.. Create a GitHub account if you don't have one.
2. .Create a new repo called `medical-qa-assistant`. Make it public.
3. I.nstall Git locally and configure it:
```ba.sh
git co.nfig --global user.name "Your Name"
git con.fig --global user.email "youremail@gmail.com"
```.
4. C.lone your repo locally, then create this folder structure:
```.
medi.cal-qa-assistant/
├── d.ata/
│   ├─.─ raw/
│   └──. processed/
├── mode.ls/
├── noteb.ooks/
├── src/.
│   ├── t.raining/
│   ├── ra.g/
│   └── api./
├── tests/.
├── README..md
├── LEARNING.S.md
├── requireme.nts.txt
└── .gitignore.
```.
5. A.dd a `.gitignore` that excludes: `*.pyc`, `__pycache__/`, `*.env`, `data/raw/` (data.sets are too large for GitHub), `models/` (model weights are too large).
.6. Make your first commit:
`.``bash
git add .
git commit -m "Initial project structure"
git push origin main
```

**You're done when:** Your repo is live on GitHub with the folder structure committed.

---

### Day 10 — Download and Explore Datasets
**Time:** 4 hrs

**Tasks:**
1. Create `notebooks/01_data_exploration.ipynb`.
2. Download and explore med_qa:
```python
from datasets import load_dataset

med_qa = load_dataset("bigbio/med_qa", trust_remote_code=True)
print(f"Train size: {len(med_qa['train'])}")
print(f"Example: {med_qa['train'][0]}")
```
3. Download and explore ChatDoctor:
```python
chatdoctor = load_dataset("Kent0n-Li/ChatDoctor-HealthCareMagic-100k")
print(f"Total examples: {len(chatdoctor['train'])}")
print(f"Example: {chatdoctor['train'][0]}")
```
4. For each dataset, answer these in your notebook:
   - How many examples?
   - What fields/columns does each example have?
   - What does a "good" example look like vs a "bad" one?
   - What format is the question? What format is the answer?

**You're done when:** You've seen and understood the structure of both datasets.

---

### Day 11 — MLflow Setup and First Experiment
**Time:** 3 hrs

**Tasks:**
1. Understand what MLflow does: it's like a diary for your ML experiments. Every time you train a model, it logs the settings you used and the results so you can compare them later.
2. Set up MLflow locally:
```python
import mlflow

mlflow.set_experiment("medqa-experiments")

with mlflow.start_run(run_name="test_run"):
    mlflow.log_param("learning_rate", 0.001)
    mlflow.log_param("epochs", 3)
    mlflow.log_metric("test_accuracy", 0.85)
    mlflow.log_metric("loss", 0.42)

print("Run logged!")
```
3. Start the MLflow UI:
```bash
mlflow ui
```
4. Open `http://localhost:5000` in your browser. You should see your test run logged.
5. Commit your exploration notebook.

**You're done when:** You can see experiment logs in the MLflow UI.

---

### Day 12 — Week 1 Review + Buffer
**Time:** 3 hrs

**Tasks:**
1. Review everything from this week. Fix any setup issues that are still nagging you.
2. Make sure your GitHub has commits from every day this week.
3. Write a proper README.md introduction:
   - What this project is
   - What tech stack you're using
   - Status: "In Progress - Week 1"
4. Update `LEARNINGS.md` with a week summary.
5. Push everything to GitHub.

**You're done when:** Your repo is clean, README exists, daily commits are visible.

---

## 📅 WEEK 2 — Data Preparation & Understanding (Days 13–19)
> **Goal:** Clean your data and understand it deeply. Good data = good model. Don't rush this.

---

### Day 13 — Understand What Good Training Data Looks Like
**Time:** 3 hrs

**Tasks:**
1. Open `notebooks/02_data_analysis.ipynb`
2. For ChatDoctor dataset, analyze:
```python
import pandas as pd

df = pd.DataFrame(chatdoctor['train'])
print(df.columns)
print(df.head())

# Check answer lengths
df['answer_length'] = df['output'].str.len()
print(df['answer_length'].describe())

# Look at very short answers (likely bad quality)
short_answers = df[df['answer_length'] < 50]
print(f"Short answers: {len(short_answers)}")
print(short_answers['output'].head(10))
```
3. Identify patterns of BAD examples (too short, no medical content, just "see a doctor").

**You're done when:** You know what good vs bad training examples look like in your data.

---

### Day 14 — Data Cleaning
**Time:** 4 hrs

**Tasks:**
1. Create `src/training/data_prep.py`
2. Write cleaning functions:
```python
def is_good_example(example):
    """Filter out low quality examples"""
    output = example.get('output', '')
    input_text = example.get('input', '')
    
    # Too short - not useful
    if len(output) < 100:
        return False
    
    # Too long - will exceed context window
    if len(output) > 2000:
        return False
        
    # No actual question
    if len(input_text) < 10:
        return False
    
    return True

# Apply filter
chatdoctor_df = pd.DataFrame(chatdoctor['train'])
clean_df = chatdoctor_df[chatdoctor_df.apply(is_good_example, axis=1)]
print(f"Before: {len(chatdoctor_df)}, After: {len(clean_df)}")
```
3. Save cleaned data to `data/processed/cleaned_chatdoctor.csv`

**You're done when:** You have a cleaned dataset saved locally.

---

### Day 15 — Format Data as Instruction-Response Pairs
**Time:** 4 hrs

**Tasks:**
1. LLMs during fine-tuning expect a specific format. The most common is:
```
<|system|>
You are a helpful medical assistant. Answer questions clearly and always recommend seeing a doctor for serious concerns.
<|user|>
{patient question here}
<|assistant|>
{doctor response here}
```
2. Write a formatter:
```python
SYSTEM_PROMPT = """You are a knowledgeable medical assistant. 
Provide clear, accurate information based on the question.
Always recommend consulting a healthcare professional for diagnosis and treatment.
If you are uncertain, say so clearly rather than guessing."""

def format_example(row):
    return {
        "text": f"<|system|>\n{SYSTEM_PROMPT}\n<|user|>\n{row['input']}\n<|assistant|>\n{row['output']}"
    }

formatted_df = clean_df.apply(format_example, axis=1, result_type='expand')
```
3. Save formatted data to `data/processed/formatted_training_data.csv`
4. Look at 10 examples — does the format look right?

**You're done when:** Your data is in instruction-response format and looks correct when you print it.

---

### Day 16 — Run Phi-3 Mini (Just Inference, No Training Yet)
**Time:** 4 hrs

**Tasks:**
1. Create `notebooks/03_model_exploration.ipynb`
2. Before training, see how the BASE model behaves:
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "microsoft/Phi-3-mini-4k-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load in 4-bit to fit in 6GB VRAM
from transformers import BitsAndBytesConfig
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

# Ask it a medical question
prompt = "<|user|>\nI have a severe headache and sensitivity to light. What could this be?<|assistant|>\n"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```
3. Ask it 5 different medical questions. Write down the responses.
4. Note: Is it helpful? Does it hallucinate? Does it hedge appropriately?

**You're done when:** You've seen the base model's behavior and documented it.

---

### Day 17 — Understand QLoRA (Conceptually + Setup)
**Time:** 3 hrs

**Tasks:**
1. Read this explanation: LoRA adds small "adapter" matrices to specific layers of the model. During training, only these tiny matrices are updated — not the billions of base model weights. QLoRA does this with the model loaded in 4-bit, saving VRAM.
2. Set up the LoRA config:
```python
from peft import LoraConfig, TaskType

lora_config = LoraConfig(
    r=16,                    # Rank - higher = more parameters = more expressive but slower
    lora_alpha=32,           # Scaling factor (usually 2x rank)
    target_modules=["q_proj", "v_proj"],  # Which layers to add adapters to
    lora_dropout=0.05,       # Regularization
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
```
3. In your notebook, print how many parameters are trainable vs total:
```python
from peft import get_peft_model
peft_model = get_peft_model(model, lora_config)
peft_model.print_trainable_parameters()
# Should show something like: trainable params: 2,097,152 || all params: 3,823,423,488 || trainable%: 0.05%
```
4. Write in `LEARNINGS.md`: What does the trainable% tell you about QLoRA?

**You're done when:** You understand what LoRA rank and alpha mean conceptually.

---

### Day 18 — Create Training Script Skeleton
**Time:** 3 hrs

**Tasks:**
1. Create `src/training/train.py` — just the structure, not the full training loop yet:
```python
"""
Medical QA Fine-tuning Script
Uses QLoRA to fine-tune Phi-3 Mini on medical conversations
"""
import torch
import mlflow
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
import pandas as pd

# --- Config ---
MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
DATA_PATH = "data/processed/formatted_training_data.csv"
OUTPUT_DIR = "models/phi3-medical-lora"
EPOCHS = 2
BATCH_SIZE = 2
LEARNING_RATE = 2e-4

def load_model_and_tokenizer():
    pass  # TODO Day 19

def load_training_data():
    pass  # TODO Day 19

def train():
    pass  # TODO Day 19

if __name__ == "__main__":
    train()
```
2. Commit this skeleton to GitHub.

**You're done when:** The skeleton file is committed with clear TODOs for what you'll implement.

---

### Day 19 — Week 2 Review + Buffer
**Time:** 3 hrs

**Tasks:**
1. Review your cleaned and formatted data. Spot-check 20 random examples — do they look right?
2. Make sure you understand the full pipeline so far: raw data → clean → format → ready for training.
3. Update README with "Week 2 complete" status.
4. Look ahead at Week 3 — read through the training days so nothing surprises you.

**You're done when:** Data pipeline is complete and committed, LEARNINGS.md is updated.

---

## 📅 WEEK 3 — Fine-tuning (Days 20–26)
> **Goal:** Train your model. This is the heart of the project.

---

### Day 20 — Complete the Training Script
**Time:** 4–5 hrs

**Tasks:**
1. Fill in `src/training/train.py`:
```python
def load_model_and_tokenizer():
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
    model.config.use_cache = False
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    return model, tokenizer

def load_training_data():
    df = pd.read_csv(DATA_PATH)
    # Use 10k examples for speed — you can increase later
    df = df.sample(10000, random_state=42)
    # 90/10 split
    train_df = df.iloc[:9000]
    eval_df = df.iloc[9000:]
    return train_df, eval_df
```

**You're done when:** Functions are complete and importable without errors.

---

### Day 21 — Add LoRA + Training Loop
**Time:** 4 hrs

**Tasks:**
1. Complete the `train()` function:
```python
def train():
    print("Loading model...")
    model, tokenizer = load_model_and_tokenizer()
    
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    train_df, eval_df = load_training_data()
    
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=4,
        learning_rate=LEARNING_RATE,
        fp16=True,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=200,
        save_steps=500,
        warmup_ratio=0.03,
        report_to="none",  # We'll log to MLflow manually
    )
    
    with mlflow.start_run(run_name="phi3-medical-qlora"):
        mlflow.log_params({
            "model": MODEL_NAME,
            "epochs": EPOCHS,
            "lora_r": 16,
            "learning_rate": LEARNING_RATE,
            "training_examples": 9000
        })
        
        trainer = SFTTrainer(
            model=model,
            train_dataset=train_df.to_dict('records'),  # adjust as needed
            eval_dataset=eval_df.to_dict('records'),
            args=training_args,
            tokenizer=tokenizer,
        )
        
        trainer.train()
        mlflow.log_metric("final_loss", trainer.state.log_history[-1].get('loss', 0))
    
    print("Saving adapter weights...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Done!")
```

**You're done when:** The training script runs without import errors.

---

### Day 22 — First Training Run (Test Run)
**Time:** 4–5 hrs

**Tasks:**
1. Before the full run, test with tiny data so you know it works:
```python
# Temporarily change in train.py:
df = df.sample(200, random_state=42)  # Just 200 examples
EPOCHS = 1                              # Just 1 epoch
```
2. Run the training:
```bash
python src/training/train.py
```
3. Watch for errors. Common ones:
   - CUDA out of memory → reduce BATCH_SIZE to 1
   - Tokenizer issues → check padding settings
   - Data format issues → print one example going into the trainer
4. If it completes without error, check MLflow UI to see the run logged.

**You're done when:** A small training run completes without error and is logged in MLflow.

---

### Day 23 — Full Training Run
**Time:** 5–6 hrs (mostly waiting)

**Tasks:**
1. Change back to full settings (10k examples, 2 epochs).
2. Start training and let it run. This will take 3–5 hours on your 4050.
3. While it runs:
   - Watch the loss numbers — they should generally go DOWN over time
   - If loss goes to NaN, something is wrong (stop and investigate)
   - Open MLflow UI and watch metrics update
4. Write in `LEARNINGS.md` while you wait: What do you expect the model to be better at after training?

**You're done when:** Training completes, adapter weights saved in `models/phi3-medical-lora/`.

---

### Day 24 — Evaluate Your Fine-tuned Model
**Time:** 4 hrs

**Tasks:**
1. Create `notebooks/04_model_evaluation.ipynb`
2. Load the fine-tuned model:
```python
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch

base_model_name = "microsoft/Phi-3-mini-4k-instruct"
adapter_path = "models/phi3-medical-lora"

bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
base_model = AutoModelForCausalLM.from_pretrained(base_model_name, quantization_config=bnb_config, device_map="auto")
model = PeftModel.from_pretrained(base_model, adapter_path)
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
```
3. Ask the SAME 5 questions you asked on Day 16. Compare base model vs fine-tuned model answers.
4. Does the fine-tuned model sound more like the training data? Is it more structured?

**You're done when:** You have a documented comparison of base vs fine-tuned model responses.

---

### Day 25 — Push to HuggingFace Hub
**Time:** 3 hrs

**Tasks:**
1. Log in to HuggingFace from terminal:
```bash
huggingface-cli login
# Paste your HF token (from huggingface.co/settings/tokens)
```
2. Push your adapter weights (NOT the full model):
```python
from huggingface_hub import HfApi

api = HfApi()
api.create_repo(repo_id="your-username/phi3-medical-lora", private=False)

model.push_to_hub("your-username/phi3-medical-lora")
tokenizer.push_to_hub("your-username/phi3-medical-lora")
```
3. Go to HuggingFace and verify your model is there. Write a model card (the README on the model page) explaining what it is.

**You're done when:** Your adapter weights are live on HuggingFace Hub.

---

### Day 26 — Week 3 Review + Buffer
**Time:** 3 hrs

**Tasks:**
1. Log your training run details in `LEARNINGS.md` — what hyperparameters did you use and why? What was the final loss?
2. If training quality seems poor, note what you'd try differently (lower learning rate? more epochs? more data?).
3. Update README: add a "Model" section with a link to your HuggingFace model.
4. Commit everything.

> 🎉 **Milestone:** You have a fine-tuned medical LLM live on HuggingFace. That's genuinely impressive for a beginner.

---

## 📅 WEEK 4 — RAG Pipeline (Days 27–33)
> **Goal:** Give your model access to medical knowledge via document retrieval.

---

### Day 27 — Download Medical Documents
**Time:** 4 hrs

**Tasks:**
1. Create `src/rag/data_collection.py`
2. Download MedlinePlus articles (public domain, free):
```python
import requests
import json

def fetch_medlineplus_articles():
    """Fetch health topic summaries from MedlinePlus API"""
    url = "https://wsearch.nlm.nih.gov/ws/query"
    articles = []
    
    # Search for common health topics
    topics = ["diabetes", "hypertension", "headache", "fever", "chest pain", 
              "back pain", "depression", "anxiety", "asthma", "arthritis"]
    
    for topic in topics:
        params = {"db": "healthTopics", "term": topic, "retmax": 20}
        response = requests.get(url, params=params)
        if response.status_code == 200:
            articles.append({"topic": topic, "content": response.text})
    
    return articles
```
3. Save raw articles to `data/raw/medlineplus/`

**You're done when:** You have at least 500 medical articles downloaded locally.

---

### Day 28 — Clean and Chunk Documents
**Time:** 4 hrs

**Tasks:**
1. Create `src/rag/chunking.py`
2. Understand chunking: You can't feed a 10-page document into a prompt. Instead, you split it into ~400-token chunks. When a user asks a question, you find the most relevant chunks and include only those.
3. Write a chunker:
```python
def chunk_text(text, chunk_size=400, overlap=50):
    """
    Split text into overlapping chunks.
    Overlap ensures context isn't lost at chunk boundaries.
    """
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if len(chunk.strip()) > 100:  # Skip very short chunks
            chunks.append({
                "text": chunk,
                "start_word": i,
            })
    
    return chunks
```
4. Process all your documents and save chunks to `data/processed/chunks.json`

**You're done when:** You have a JSON file with all document chunks.

---

### Day 29 — Build the Vector Database
**Time:** 4 hrs

**Tasks:**
1. Create `src/rag/vector_store.py`
2. Embed all chunks and store in ChromaDB:
```python
import chromadb
from sentence_transformers import SentenceTransformer
import json

def build_vector_store(chunks_path="data/processed/chunks.json"):
    # Load chunks
    with open(chunks_path) as f:
        chunks = json.load(f)
    
    # Load embedding model (domain-specific for medical text)
    embedder = SentenceTransformer("pritamdeka/S-PubMedBert-MS-MARCO")
    
    # Set up ChromaDB
    client = chromadb.PersistentClient(path="data/chroma_db")
    collection = client.get_or_create_collection("medical_docs")
    
    print(f"Embedding {len(chunks)} chunks... this may take a while")
    
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        texts = [c['text'] for c in batch]
        embeddings = embedder.encode(texts).tolist()
        ids = [f"chunk_{i+j}" for j in range(len(batch))]
        
        collection.add(embeddings=embeddings, documents=texts, ids=ids)
        print(f"Progress: {min(i+batch_size, len(chunks))}/{len(chunks)}")
    
    print("Vector store built!")
    return collection
```

**You're done when:** ChromaDB is populated with embedded chunks and persisted to disk.

---

### Day 30 — Build the Retrieval Function
**Time:** 3 hrs

**Tasks:**
1. Write the retrieval function in `src/rag/retriever.py`:
```python
def retrieve(query, collection, embedder, top_k=10):
    """Find the most similar document chunks to a query"""
    query_embedding = embedder.encode([query]).tolist()
    
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )
    
    return results['documents'][0]  # List of top chunks

# Test it
chunks = retrieve("What are symptoms of type 2 diabetes?", collection, embedder)
for i, chunk in enumerate(chunks):
    print(f"\n--- Chunk {i+1} ---")
    print(chunk[:200])
```
2. Test with 10 different medical questions. Do the retrieved chunks seem relevant?
3. Write in `LEARNINGS.md`: When does retrieval work well? When does it fail?

**You're done when:** You can retrieve relevant chunks for any medical question.

---

### Day 31 — Add Cross-Encoder Re-ranking
**Time:** 4 hrs

**Tasks:**
1. Understand re-ranking: Vector search finds chunks that are *semantically similar* to your query. Re-ranking uses a more powerful model to score each chunk for actual *relevance*. It's slower but more accurate.
2. Add re-ranking to `src/rag/retriever.py`:
```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def retrieve_and_rerank(query, collection, embedder, top_k=10, rerank_top_k=3):
    # Step 1: Vector search - get top 10
    candidates = retrieve(query, collection, embedder, top_k=top_k)
    
    # Step 2: Re-rank - score each candidate against the query
    pairs = [(query, chunk) for chunk in candidates]
    scores = reranker.predict(pairs)
    
    # Step 3: Sort by score, keep top 3
    ranked = sorted(zip(scores, candidates), reverse=True)
    top_chunks = [chunk for score, chunk in ranked[:rerank_top_k]]
    
    return top_chunks
```
3. Compare results with and without re-ranking on 5 questions. Does it improve?

**You're done when:** Re-ranking is working and improving result quality.

---

### Day 32 — Wire Everything Together: Full RAG Pipeline
**Time:** 4 hrs

**Tasks:**
1. Create `src/rag/pipeline.py` — the full pipeline:
```python
SYSTEM_PROMPT = """You are a medical assistant. Answer the patient's question using ONLY the provided context.
If the context doesn't contain enough information, say so clearly.
Always recommend consulting a real doctor for diagnosis and treatment."""

def answer_question(question, collection, embedder, reranker, model, tokenizer):
    # Step 1: Retrieve relevant chunks
    context_chunks = retrieve_and_rerank(question, collection, embedder)
    context = "\n\n".join(context_chunks)
    
    # Step 2: Build prompt
    prompt = f"""<|system|>
{SYSTEM_PROMPT}

Context from medical literature:
{context}
<|user|>
{question}
<|assistant|>
"""
    
    # Step 3: Generate answer
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=300, temperature=0.1)
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = response.split("<|assistant|>")[-1].strip()
    
    return {
        "answer": answer,
        "sources": context_chunks
    }
```
2. Test the full pipeline end-to-end with 5 questions.

**You're done when:** You can ask a question and get an answer with source chunks in one call.

---

### Day 33 — Week 4 Review + Buffer
**Time:** 3 hrs

**Tasks:**
1. Deploy a basic Gradio demo just for yourself (not public yet):
```python
import gradio as gr

def chat(question):
    result = answer_question(question, ...)
    return result['answer'], "\n\n".join(result['sources'])

demo = gr.Interface(
    fn=chat,
    inputs=gr.Textbox(label="Ask a medical question"),
    outputs=[gr.Textbox(label="Answer"), gr.Textbox(label="Sources")]
)
demo.launch()
```
2. Show it to someone. Get their reaction.
3. Update README with a RAG section.

> 🎉 **Milestone:** You have a working RAG-augmented medical QA system running locally. This is the core of the project.

---

## 📅 WEEK 5 — MLOps + API (Days 34–40)
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

app = FastAPI(title="Medical QA API", version="1.0")

class QuestionRequest(BaseModel):
    question: str

class QuestionResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: str
    latency_seconds: float

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/ask", response_model=QuestionResponse)
def ask_question(request: QuestionRequest):
    if len(request.question.strip()) < 5:
        raise HTTPException(status_code=400, detail="Question too short")
    
    start = time.time()
    result = answer_question(request.question, ...)
    latency = round(time.time() - start, 2)
    
    # Simple confidence heuristic: did we find good sources?
    confidence = "high" if len(result['sources']) >= 2 else "low"
    
    return QuestionResponse(
        answer=result['answer'],
        sources=result['sources'],
        confidence=confidence,
        latency_seconds=latency
    )
```
2. Run it: `uvicorn src.api.main:app --reload`
3. Open `http://localhost:8000/docs` — FastAPI auto-generates interactive API docs.
4. Test your `/ask` endpoint from the docs page.

**You're done when:** API is running and you can make requests from the browser.

---

### Day 35 — Write Tests for Your API
**Time:** 3 hrs

**Tasks:**
1. Create `tests/test_api.py`:
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
2. Run tests: `pytest tests/`
3. All should pass.

**You're done when:** Tests pass and are committed to GitHub.

---

### Day 36 — GitHub Actions CI/CD
**Time:** 3 hrs

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
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: "3.11"
      
      - name: Install dependencies
        run: |
          pip install fastapi httpx pytest pydantic uvicorn
      
      - name: Run tests
        run: pytest tests/ -v
```
2. Push to GitHub. Go to the Actions tab — watch the workflow run.
3. If it fails, read the error and fix it.
4. Add a badge to your README:
```markdown
![Tests](https://github.com/YOUR_USERNAME/medical-qa-assistant/actions/workflows/test.yml/badge.svg)
```

**You're done when:** Green badge on your README. This is a real MLOps talking point.

---

### Day 37 — DVC for Data Versioning (Optional but valuable)
**Time:** 3 hrs

**Tasks:**
1. DVC tracks large files (datasets, model weights) that you can't put in Git.
2. Initialize DVC:
```bash
dvc init
git add .dvc .dvcignore
git commit -m "Initialize DVC"
```
3. Track your processed data:
```bash
dvc add data/processed/chunks.json
dvc add data/processed/formatted_training_data.csv
git add data/processed/*.dvc
git commit -m "Track processed datasets with DVC"
```
4. Now if you change your data processing, run `dvc repro` to regenerate.

**You're done when:** Your key data files are DVC-tracked.

---

### Day 38 — Evaluation Script
**Time:** 4 hrs

**Tasks:**
1. Create `src/evaluation/evaluate.py`:
```python
"""
Evaluate the QA system on test questions.
This script runs on every push via GitHub Actions.
"""
import json

# 20 test questions with expected answer topics
TEST_QUESTIONS = [
    {"question": "What are symptoms of type 2 diabetes?", "expected_topics": ["blood sugar", "thirst", "urination", "fatigue"]},
    {"question": "How is hypertension diagnosed?", "expected_topics": ["blood pressure", "mmhg", "measurement"]},
    {"question": "What is ibuprofen used for?", "expected_topics": ["pain", "inflammation", "fever"]},
    # ... add 17 more
]

def evaluate(questions):
    results = []
    for q in questions:
        answer = answer_question(q['question'], ...)['answer'].lower()
        # Check if expected topics appear in answer
        hits = [topic for topic in q['expected_topics'] if topic in answer]
        score = len(hits) / len(q['expected_topics'])
        results.append({"question": q['question'], "score": score})
    
    avg_score = sum(r['score'] for r in results) / len(results)
    print(f"Average score: {avg_score:.2%}")
    
    # Save results
    with open("evaluation_results.json", "w") as f:
        json.dump({"score": avg_score, "results": results}, f)
    
    return avg_score

if __name__ == "__main__":
    score = evaluate(TEST_QUESTIONS)
    if score < 0.5:
        print("WARNING: Evaluation score below 50%")
        exit(1)  # Fail CI if quality drops
```

**You're done when:** Evaluation script runs and produces a score.

---

### Day 39 — Set Up Groq API for Fast Inference
**Time:** 3 hrs

**Tasks:**
1. Go to console.groq.com, create a free account, get an API key.
2. Groq runs Llama models at very high speed for free (within rate limits).
3. Create `src/inference/groq_client.py`:
```python
from groq import Groq
import os

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def generate_with_groq(prompt, model="llama3-8b-8192"):
    """Use Groq for fast cloud inference instead of local GPU"""
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        max_tokens=500
    )
    return chat_completion.choices[0].message.content

# Add medical system prompt
def answer_with_groq(question, context_chunks):
    context = "\n\n".join(context_chunks)
    prompt = f"""You are a helpful medical assistant. Use the provided context to answer the question.
Always recommend consulting a healthcare professional.

Context:
{context}

Question: {question}

Answer:"""
    return generate_with_groq(prompt)
```
4. Test it — it should be much faster than local inference.

**You're done when:** Groq inference is working and returns good answers.

---

### Day 40 — Week 5 Review + Buffer
**Time:** 3 hrs

**Tasks:**
1. Make sure all components work together: FastAPI → RAG pipeline → Groq or local inference.
2. Fix any integration bugs.
3. Update README with API documentation section.
4. Look ahead: Week 6 is all about the public demo.

---

## 📅 WEEK 6 — Deployment & Polish (Days 41–47)
> **Goal:** Get it live. The world can see it.

---

### Day 41 — Build the Gradio UI
**Time:** 4 hrs

**Tasks:**
1. Create `app.py` (the HuggingFace Spaces entry point):
```python
import gradio as gr
import os

DISCLAIMER = """⚠️ **Medical Disclaimer:** This tool is for educational purposes only. 
It is NOT a substitute for professional medical advice, diagnosis, or treatment. 
Always consult a qualified healthcare provider."""

def ask_medical_question(question, history):
    if not question.strip():
        return history, "", "Please enter a question."
    
    result = answer_question(question)
    answer = result['answer']
    sources_text = "\n\n".join([f"**Source {i+1}:** {s[:300]}..." for i, s in enumerate(result['sources'])])
    
    history.append((question, answer))
    return history, sources_text, ""

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏥 Medical QA Assistant")
    gr.Markdown(DISCLAIMER)
    
    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(height=400)
            question_input = gr.Textbox(placeholder="Ask a medical question...", label="Your Question")
            submit_btn = gr.Button("Ask", variant="primary")
        
        with gr.Column(scale=1):
            sources_output = gr.Markdown(label="Sources Used")
    
    submit_btn.click(
        ask_medical_question,
        inputs=[question_input, chatbot],
        outputs=[chatbot, sources_output, question_input]
    )

demo.launch()
```

**You're done when:** Gradio app runs locally with a clean, professional UI.

---

### Day 42 — Deploy to HuggingFace Spaces
**Time:** 4 hrs

**Tasks:**
1. Create a new Space on huggingface.co/spaces (type: Gradio).
2. Create a `requirements.txt` for the Space:
```
gradio
sentence-transformers
chromadb
groq
transformers
torch
```
3. Create a `README.md` for the Space with the metadata header:
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
4. Push your code to the Space (it auto-deploys).
5. Wait for build. Fix any errors in the build logs.

**You're done when:** Your app is live at `huggingface.co/spaces/yourname/medical-qa`.

---

### Day 43 — Test the Live Demo Thoroughly
**Time:** 3 hrs

**Tasks:**
1. Test 20+ questions on the live demo. Check for:
   - Does it answer common medical questions well?
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
Your README is often the first thing a recruiter sees. Make it excellent.

1. Structure:
   ```markdown
   # 🏥 Medical QA Assistant
   
   [badges: Tests passing | Live Demo | Model on HuggingFace]
   
   > A medical Q&A system built with fine-tuned Phi-3 Mini + RAG pipeline
   
   ## Live Demo
   [Link to HuggingFace Space]
   
   ## Architecture
   [Diagram — see Day 45]
   
   ## Features
   - Fine-tuned Phi-3 Mini (3.8B) on 100k+ medical conversations
   - RAG pipeline with domain-specific BioMedical embeddings
   - Cross-encoder re-ranking for answer quality
   - FastAPI backend with CI/CD via GitHub Actions
   - Deployed on HuggingFace Spaces
   
   ## Setup
   [Clear installation steps]
   
   ## How It Works
   [Explain the pipeline in plain English]
   
   ## Limitations
   [Be honest: not a real medical tool, model limitations]
   ```

**You're done when:** README is polished and someone with no context can understand the project.

---

### Day 45 — Draw the Architecture Diagram
**Time:** 2 hrs

**Tasks:**
1. Use draw.io (free, browser-based) to draw the full system:
```
User Question
     ↓
[Gradio UI / FastAPI]
     ↓
[Sentence Transformer Embedder]
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
2. Export as PNG, add to README.
3. This diagram alone impresses interviewers.

**You're done when:** Architecture diagram is in the README.

---

### Day 46 — Record Demo Video
**Time:** 3 hrs

**Tasks:**
1. Download Loom (free screen recorder).
2. Record a 2-minute demo:
   - 0:00–0:20: "This is a medical QA assistant I built. Here's the architecture." (show diagram)
   - 0:20–1:20: Ask 3 real questions live, show answers + sources
   - 1:20–1:40: "Under the hood: fine-tuned Phi-3, RAG with re-ranking, deployed on HuggingFace"
   - 1:40–2:00: "Here's the GitHub with full code and CI/CD pipeline"
3. Add the Loom link to your README.
4. Post on LinkedIn with a short caption explaining what you built and what you learned.

**You're done when:** Demo video is live and LinkedIn post is up.

---

### Day 47 — Final Polish + Week 6 Review
**Time:** 3 hrs

**Tasks:**
1. Go through every file in your repo. Remove debug print statements, clean up commented code.
2. Make sure every notebook has markdown cells explaining what it does.
3. Write your resume bullet:
   ```
   Built and deployed a medical Q&A assistant — fine-tuned Phi-3 Mini (3.8B) 
   on 100k+ medical conversations using QLoRA on an RTX 4050, layered a RAG 
   pipeline with domain-specific BioMedical embeddings and cross-encoder 
   re-ranking over a 50k-document corpus, built an MLOps pipeline with DVC + 
   GitHub Actions CI, and deployed publicly on HuggingFace Spaces.
   ```
4. Add it to your resume and LinkedIn.

> 🎉 **Project Complete.** You built a production-quality ML project from scratch as a beginner. That's not a small thing.

---

## 📅 WEEKS 7–8 — Optional Improvements (Days 48–56)
> Use this time if you want to go deeper, or move on to your next project.

---

### Optional Improvements (pick any):

**Make the model better:**
- Train on all 100k ChatDoctor examples instead of 10k
- Add PubMed fine-tuning data
- Experiment with different LoRA ranks (r=8 vs r=32)

**Make the RAG better:**
- Add more documents (PubMed abstracts via the API)
- Try different chunk sizes (200 tokens vs 600 tokens) and compare quality
- Add hybrid search (keyword + vector)

**Make the product better:**
- Add conversation history (multi-turn chat)
- Add a feedback button ("Was this helpful?")
- Log questions anonymously to improve future training data

**Make the engineering better:**
- Add proper logging with Python's `logging` module
- Add rate limiting to the FastAPI endpoint
- Write more comprehensive tests
- Add Docker containerization

---

## 📋 Daily Checklist

Use this every day:
- [ ] Did I commit to GitHub today?
- [ ] Did I write in LEARNINGS.md?
- [ ] Did I understand what I coded, or just copy-paste it?
- [ ] What will I do first tomorrow?

---

## 🆘 When You Get Stuck (You Will)

1. **Read the error message carefully** — it usually tells you exactly what's wrong
2. **Google the exact error message** in quotes
3. **Check HuggingFace forums** (discuss.huggingface.co) — most errors have been seen before
4. **Check your GPU memory**: `nvidia-smi` in terminal
5. **Restart the kernel** — sometimes fixes mysterious issues
6. **Reduce batch size** — if you're getting CUDA out of memory errors

---

*Good luck. You're going to build something real.*
