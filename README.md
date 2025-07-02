# 🤖 RAG PDF Chatbot with LangChain, FAISS, and Ollama (LLaMA3)

A simple yet powerful chatbot system that enables natural language Q&A over one or more PDF documents using **Retrieval-Augmented Generation (RAG)**.

Built with:
- 🧠 LangChain for RAG pipeline
- 📚 FAISS for vector search
- 🧾 PyPDF2 for document parsing
- 💬 LLaMA3 running via [Ollama](https://ollama.com/)
- 🌐 Streamlit for a fast interactive UI

---

## 🚀 Features

- 🔍 Query one or more PDF documents using natural questions.
- ⚡ Local semantic search powered by **FAISS** and **Instructor Embeddings**.
- 💡 Supports **streaming conversation** with history memory.
- 📦 Run LLaMA3 locally with **Ollama** – no OpenAI API required.
- 🖥️ Web interface with Streamlit, ready to deploy or extend.

---

## 🧱 Project Structure

```bash
.
├── app.py                 # Main Streamlit app
├── htmlTemplate.py        # Custom HTML/CSS for chat UI
├── requirements.txt       # Dependencies
└── .env                   # For storing API keys (if needed)
