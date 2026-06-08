# 🎯 Week 3: Model Fine-tuning via QLoRA

This document outlines the fine-tuning setup, training parameters, performance metrics, and adapter deployment accomplished during Days 20–26.

---

## 1. Fine-tuning Infrastructure & Optimization

* **Base Model:** `microsoft/Phi-3-mini-4k-instruct` (3.8B parameters).
* **Hardware Constrains:** Executed on a single laptop GPU (RTX 4050, 6GB VRAM).
* **Memory Optimizations:**
  * **4-Bit Quantization (`BitsAndBytesConfig`):** Loads the base model weights in NF4 4-bit format with nested quantization, reducing model memory footprint to fit in VRAM.
  * **Gradient Checkpointing:** Trades computing operations for memory by recomputing intermediate activation weights on the fly during backpropagation (saving ~40% VRAM).
  * **Mixed-Precision (`bf16`):** Faster calculation speed and avoids gradient scaling errors on RTX GPUs.

---

## 2. LoRA Hyperparameters

We configured Parameter-Efficient Fine-Tuning (PEFT) targeting the attention modules:
* **Rank ($r$):** `16` (Balances representation capability with parameter footprint).
* **Alpha ($\alpha$):** `32` (Scaling multiplier).
* **Dropout:** `0.05` (Light regularization).
* **Target Modules:** Query, Key, Value, and Output projection matrices (`q_proj`, `k_proj`, `v_proj`, `o_proj`).
* **Trainable Parameters:** **~2.1 million parameters** updated (~0.05% of the total 3.8B model parameters).

---

## 3. Training Loop Details

* **Dataset Size:** 10,000 examples sampled from the formatted training corpus (9,000 train, 1,000 evaluation split).
* **Epochs:** `2`
* **Batch Size:** `2` (with `4` gradient accumulation steps, yielding an effective batch size of `8`).
* **Learning Rate:** `2e-4` (using a cosine decay scheduler and a 3% warmup phase).
* **Max Sequence Length:** `512` tokens.
* **Logging & Tracking:** MLflow tracked validation steps every 200 iterations.

---

## 4. Key Metrics Achieved

* **Final Training Loss:** `~5.589`
* **Final Evaluation Loss:** `~5.613`
* **Adapter Output Size:** The resulting LoRA adapter files (`adapter_model.safetensors`, config, and tokenizers) occupy only **~20 MB** on disk, unlike full model weights which would require several gigabytes.

---

## 5. Weights Deployment

* **Local Target:** `models/phi3-medical-lora/` (gitignored).
* **Hugging Face Hub:** Pushed directly to [**koi-bito/phi3-medical-lora**](https://huggingface.co/koi-bito/phi3-medical-lora) using `HfApi.upload_folder`.
* **Persona Comparison (Base vs. Fine-tuned):** 
  * The fine-tuned adapter shifts the model response style, adopting a conversational, supportive, and safety-focused tone that consistently advises clinical validation when appropriate, matching the target style in `ChatDoctor`.
