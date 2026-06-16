---
title: Medical QA Assistant
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.0.0
app_file: app.py
pinned: false
python_version: 3.13
---

# 🏥 Medical QA Assistant

This is a public demo of the Medical Q&A chatbot. The application uses a Retrieval-Augmented Generation (RAG) pipeline powered by a medical vector database to retrieve relevant context and answers user queries using Groq's cloud API.

## Features
* **RAG Retrieval:** Retrieves real medical reference chunks to answer patient questions.
* **Groq Cloud Inference:** Sub-second response times using Llama-3.1-8b-instant.
* **Source Attribution:** Shows the reference documents/conversations retrieved for transparency.
* **Disclaimer Guardrail:** Built-in medical disclaimer.
