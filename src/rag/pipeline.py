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
        trust_remote_code=False,
        attn_implementation="eager"
    )
    
    import os
    if os.path.exists("models/phi3-medical-lora"):
        model = PeftModel.from_pretrained(base, "models/phi3-medical-lora")
    else:
        print("Warning: Fine-tuned model not found at models/phi3-medical-lora. Using base model instead.")
        model = base
        
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct", trust_remote_code=False)

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
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=300, do_sample=False)

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer   = response.split("<|assistant|>")[-1].strip()

    return {"answer": answer, "sources": context_chunks}

if __name__ == "__main__":
    model, tokenizer, embedder, collection, reranker = load_all()
    test_questions = [
        "What are symptoms of type 2 diabetes?",
        "How is hypertension treated?",
        "What causes a migraine?",
        "What are the side effects of aspirin?",
        "How do you prevent heart disease?"
    ]
    for q in test_questions:
        print(f"\nQuestion: {q}")
        res = answer_question(q, model, tokenizer, embedder, collection, reranker)
        print(f"Answer: {res['answer']}")
