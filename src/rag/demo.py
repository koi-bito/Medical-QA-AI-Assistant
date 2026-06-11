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

if __name__ == "__main__":
    demo.launch()
