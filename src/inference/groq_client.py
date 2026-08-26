from groq import Groq
import os

# It's best practice to handle cases where the key might not be set yet
api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    print("WARNING: GROQ_API_KEY environment variable not set. Groq client will fail.")

client = Groq(api_key=api_key)

def answer_with_groq(question, context_chunks, model="openai/gpt-oss-20b"):
    """Use Groq for fast cloud inference — good for the public demo"""
    context = "\n\n".join(context_chunks)
    prompt  = f"""You are a strict and helpful medical assistant. Use the provided medical context to answer the question if it is relevant.
If the user is just greeting you or making small talk, respond conversationally and ignore the context.
If the context is completely irrelevant to the question, ignore it.
If the user asks a non-medical question, you MUST politely refuse to answer and state that you are a medical assistant and can only answer medical questions.
Always recommend consulting a healthcare professional for medical questions.

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

def answer_with_groq_no_context(question, model="openai/gpt-oss-20b"):
    """Lightweight mode: answer using Groq's built-in knowledge (no RAG retrieval).
    Used on Render free tier where we can't load torch/sentence-transformers."""
    prompt = f"""You are a strict and helpful medical assistant. Answer the user's question using your medical knowledge.
If the user is just greeting you or making small talk, respond conversationally.
If the user asks a non-medical question, you MUST politely refuse to answer and state that you are a medical assistant and can only answer medical questions.
Always recommend consulting a healthcare professional for medical questions.
Be concise but thorough.

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
    print("Groq API Test Result:")
    print("-" * 20)
    print(test_answer)
