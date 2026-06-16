from groq import Groq
import os

# It's best practice to handle cases where the key might not be set yet
api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    print("WARNING: GROQ_API_KEY environment variable not set. Groq client will fail.")

client = Groq(api_key=api_key)

def answer_with_groq(question, context_chunks, model="llama-3.1-8b-instant"):
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
    print("Groq API Test Result:")
    print("-" * 20)
    print(test_answer)
