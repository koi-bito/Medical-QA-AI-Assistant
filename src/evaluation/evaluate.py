import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoConfig
from peft import PeftModel

MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
ADAPTER_PATH = "models/phi3-medical-lora"

SYSTEM_PROMPT = """You are a knowledgeable medical assistant.
Provide clear, accurate information based on the patient's question.
Always recommend consulting a qualified healthcare professional for diagnosis and treatment.
If you are uncertain about something, say so clearly rather than guessing."""

def generate_response(model, tokenizer, question):
    prompt = f"<|system|>\n{SYSTEM_PROMPT}<|end|>\n<|user|>\n{question}<|end|>\n<|assistant|>\n"
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=256,
            temperature=0.1,
            repetition_penalty=1.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
    # Extract only the newly generated tokens
    prompt_length = inputs.input_ids.shape[1]
    generated_tokens = outputs[0][prompt_length:]
    answer = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    
    return answer

def main():
    print("Loading base model (in 4-bit) & tokenizer...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=False
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=False)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading LoRA adapters from {ADAPTER_PATH}...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

    questions = [
        "I have had a mild fever and a dry cough for the past two days. What should I do?",
        "My 6-year-old child accidentally swallowed a small plastic toy. She is breathing perfectly fine and not choking. What should I do?",
        "What are the typical symptoms of a migraine compared to a regular headache?"
    ]

    print("\n" + "=" * 60)
    print("  Medical QA Evaluation — Phi-3 Fine-tuned")
    print("=" * 60 + "\n")
    
    for q in questions:
        print(f"Patient: {q}")
        print("-" * 60)
        ans = generate_response(model, tokenizer, q)
        print(f"Assistant: {ans}\n")
        print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
