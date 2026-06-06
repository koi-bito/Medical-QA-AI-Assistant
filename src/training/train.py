"""
Medical QA Fine-tuning Script — QLoRA
Fine-tunes Phi-3 Mini on medical conversations using QLoRA.

Hardware target: RTX 4050 (6GB VRAM)
Days 18, 20, 21 of the implementation plan.

Usage:
    python src/training/train.py
"""

import os
import torch
import mlflow
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

# ─── Config ───────────────────────────────────────────────────────────────────
MODEL_NAME    = "microsoft/Phi-3-mini-4k-instruct"
DATA_PATH     = "data/processed/formatted_training_data.csv"
OUTPUT_DIR    = "models/phi3-medical-lora"

# Training hyperparameters
EPOCHS        = 2
BATCH_SIZE    = 2           # Keep low for 6GB VRAM
LEARNING_RATE = 2e-4
MAX_SEQ_LEN   = 512         # Phi-3 has 4k context; 512 keeps VRAM usage low

# LoRA hyperparameters
LORA_R        = 16          # Rank — higher = more expressive, more memory
LORA_ALPHA    = 32          # Scaling factor (rule of thumb: 2x rank)
LORA_DROPOUT  = 0.05        # Light regularization

# Data split
N_EXAMPLES    = 10_000      # Out of ~100k — enough to see improvement fast
TRAIN_RATIO   = 0.9         # 90% train, 10% eval
# ──────────────────────────────────────────────────────────────────────────────


def load_model_and_tokenizer():
    """Load Phi-3 Mini in 4-bit quantization."""
    print("Loading model in 4-bit quantization...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True  # Nested quantization — saves extra VRAM
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    model.config.use_cache = False  # Required for gradient checkpointing during training

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Required for causal LMs

    vram_used = torch.cuda.memory_allocated() / 1e9
    print(f"Model loaded. VRAM used: {vram_used:.2f} GB")

    return model, tokenizer


def load_training_data():
    """Load and split the formatted training data."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Training data not found at {DATA_PATH}.\n"
            "Run: python src/training/data_prep.py first."
        )

    df = pd.read_csv(DATA_PATH)
    print(f"Total formatted examples available: {len(df):,}")

    # Sample N_EXAMPLES for manageable training time on RTX 4050
    if len(df) > N_EXAMPLES:
        df = df.sample(N_EXAMPLES, random_state=42).reset_index(drop=True)
        print(f"Sampled {N_EXAMPLES:,} examples for training.")

    # 90/10 train/eval split
    split = int(len(df) * TRAIN_RATIO)
    train_df = df.iloc[:split]
    eval_df  = df.iloc[split:]

    train_dataset = Dataset.from_pandas(train_df, preserve_index=False)
    eval_dataset  = Dataset.from_pandas(eval_df, preserve_index=False)

    print(f"Train: {len(train_dataset):,} | Eval: {len(eval_dataset):,}")
    return train_dataset, eval_dataset


def train():
    """Main training function."""
    print("=" * 60)
    print("  Medical QA Fine-tuning — QLoRA on Phi-3 Mini")
    print("=" * 60)

    # ── Step 1: Load model ────────────────────────────────────────────────────
    model, tokenizer = load_model_and_tokenizer()

    # ── Step 2: Apply LoRA adapters ──────────────────────────────────────────
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        # Target the attention projection layers in each transformer block
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    # Expected output: trainable params: ~2M || all params: ~3.8B || trainable%: ~0.05%

    # ── Step 3: Load data ─────────────────────────────────────────────────────
    train_dataset, eval_dataset = load_training_data()

    # ── Step 4: Configure training ────────────────────────────────────────────
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=4,   # Effective batch = BATCH_SIZE * 4 = 8
        gradient_checkpointing=True,     # Trade compute for VRAM — necessary on 6GB
        learning_rate=LEARNING_RATE,
        fp16=True,                        # Float16 compute — faster on RTX 4050
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,              # Keep only the 2 most recent checkpoints
        warmup_ratio=0.03,               # 3% of steps as warmup
        lr_scheduler_type="cosine",
        max_length=MAX_SEQ_LEN,
        dataset_text_field="text",
        report_to="none",                # We use MLflow instead of wandb/tensorboard
    )

    # ── Step 5: Train with MLflow tracking ───────────────────────────────────
    os.makedirs("models", exist_ok=True)
    mlflow.set_experiment("medqa-experiments")

    with mlflow.start_run(run_name="phi3-medical-qlora"):
        # Log all hyperparameters
        mlflow.log_params({
            "model": MODEL_NAME,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "gradient_accumulation_steps": 4,
            "lora_r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "learning_rate": LEARNING_RATE,
            "max_seq_len": MAX_SEQ_LEN,
            "training_examples": len(train_dataset),
            "eval_examples": len(eval_dataset),
        })

        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=training_args,
            processing_class=tokenizer,  # TRL >= 1.0: renamed from tokenizer=
        )

        print("\nStarting training... (this will take 3-5 hours on RTX 4050)")
        print("Watch the loss — it should decrease steadily.\n")

        trainer.train()

        # Log the final training loss
        history = trainer.state.log_history
        final_loss = next(
            (h.get("loss") for h in reversed(history) if "loss" in h),
            None
        )
        if final_loss is not None:
            mlflow.log_metric("final_train_loss", final_loss)
            print(f"\nFinal training loss: {final_loss:.4f}")

    # ── Step 6: Save adapter weights ──────────────────────────────────────────
    print(f"\nSaving adapter weights to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"✓ Done! Adapter saved to {OUTPUT_DIR}/")
    print("  Files: adapter_config.json, adapter_model.safetensors, tokenizer files")


if __name__ == "__main__":
    train()
