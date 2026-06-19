# Week 6: Deployment & Polish

This week marked the culmination of Phase 1: deploying the fine-tuned model and RAG pipeline into a robust, publicly accessible application.

## Key Learnings

### Gradio UI Integration (Day 41)
- **Gradio Simplification:** Wrapping the complex RAG architecture in a `gr.Blocks` chat interface is fast and highly effective. The key was separating the generation logic into modular functions so that Gradio only needed to act as the presentation layer.
- **Formatting History:** With newer Gradio versions, maintaining the exact format for the `Chatbot` component is critical to prevent visual layout bugs and data-passing errors to the backend.

### Hugging Face Spaces & Dataset Uploads (Day 42-43)
- **Separation of Code and Data:** Direct Git deployment to Spaces fails or slows down if heavy binary files (like ChromaDB vector stores) are included. The correct strategy is to upload the vector database to a separate Hugging Face **Dataset** repository and configure the Space to download it asynchronously on startup using `snapshot_download`.
- **Environment Parity:** Debugging missing dependencies (e.g., `audioop-lts` required on newer Python versions in the Space) highlighted why explicit dependency tracking and strict environment parity between local testing and remote hosting are paramount.

### Presentation and Architecture (Day 44-45)
- **Readme Efficacy:** The project README serves as the project's storefront. Placing live demo links, build status badges, and a high-level plain-English explanation at the top provides maximum value to readers and recruiters.
- **Mermaid Diagrams:** Using Mermaid for architecture flowcharts natively within Markdown cleanly visualizes complex systemic interactions (such as User -> UI -> Embedder -> VectorDB -> Reranker -> Groq -> Output) and keeps documentation seamlessly version-controlled alongside code.
