import gradio as gr
import os
from huggingface_hub import snapshot_download

HF_USERNAME = "koi-bito"

# Download the vector store snapshot from HuggingFace if it doesn't exist locally
if not os.path.exists("data/chroma_db"):
    print("Downloading vector store from HuggingFace...")
    snapshot_download(
        repo_id=f"{HF_USERNAME}/medical-qa-vectorstore",
        repo_type="dataset",
        local_dir="data"
    )
    print("Vector store downloaded successfully!")

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

    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": answer})
    return history, sources_text, ""

with gr.Blocks() as demo:
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

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
