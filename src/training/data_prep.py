"""
Data Preparation Script — Medical QA Assistant
Days 14 & 15 of the implementation plan.

Steps:
  1. Load the raw ChatDoctor dataset from HuggingFace
  2. Filter out low-quality examples (too short, too long, empty)
  3. Format the remaining examples into Phi-3 Mini's expected chat format
  4. Save cleaned + formatted data to data/processed/

Usage:
    python src/training/data_prep.py
"""

import os
import pandas as pd
from datasets import load_dataset

# ─── Phi-3 Mini system prompt ─────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a knowledgeable medical assistant.
Provide clear, accurate information based on the patient's question.
Always recommend consulting a qualified healthcare professional for diagnosis and treatment.
If you are uncertain about something, say so clearly rather than guessing."""


# ─── Filtering ────────────────────────────────────────────────────────────────

def is_good_example(row) -> bool:
    """Return True only if the example meets quality thresholds."""
    output = row.get("output", "") or ""
    input_text = row.get("input", "") or ""

    # Too short to be clinically useful
    if len(output) < 100:
        return False
    # Exceeds Phi-3 Mini's 4k context window when combined with the prompt
    if len(output) > 2000:
        return False
    # No real patient question
    if len(input_text) < 10:
        return False

    return True


def load_and_clean() -> pd.DataFrame:
    """Download the dataset and filter out low-quality examples."""
    print("Loading dataset from HuggingFace...")
    dataset = load_dataset("lavita/ChatDoctor-HealthCareMagic-100k")
    df = pd.DataFrame(dataset["train"])

    print(f"Before cleaning: {len(df):,} examples")
    clean_df = df[df.apply(is_good_example, axis=1)].reset_index(drop=True)
    removed = len(df) - len(clean_df)
    print(f"After cleaning:  {len(clean_df):,} examples  ({removed:,} removed)")

    os.makedirs("data/processed", exist_ok=True)
    clean_df.to_csv("data/processed/cleaned_chatdoctor.csv", index=False)
    print("✓ Saved → data/processed/cleaned_chatdoctor.csv")

    return clean_df


# ─── Formatting ───────────────────────────────────────────────────────────────

def format_for_training(row) -> dict:
    """
    Format one example into Phi-3 Mini's expected chat format.

    Format:
        <|system|>
        {system_prompt}<|end|>
        <|user|>
        {patient_question}<|end|>
        <|assistant|>
        {doctor_answer}<|end|>
    """
    return {
        "text": (
            f"<|system|>\n{SYSTEM_PROMPT}<|end|>\n"
            f"<|user|>\n{row['input']}<|end|>\n"
            f"<|assistant|>\n{row['output']}<|end|>"
        )
    }


def prepare_training_data() -> pd.DataFrame:
    """Load cleaned data and format it for fine-tuning."""
    clean_df = pd.read_csv("data/processed/cleaned_chatdoctor.csv")

    print(f"\nFormatting {len(clean_df):,} examples...")
    formatted = clean_df.apply(format_for_training, axis=1, result_type="expand")

    formatted.to_csv("data/processed/formatted_training_data.csv", index=False)
    print("✓ Saved → data/processed/formatted_training_data.csv")

    # Sanity check — print one example
    print("\n--- Sample formatted example ---")
    print(formatted["text"].iloc[0])
    print("--- End sample ---\n")

    return formatted


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    load_and_clean()
    prepare_training_data()
    print("Data preparation complete!")
