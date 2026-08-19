# 📄 Intelligent Document Q&A — RAG Chatbot

An AI-powered document question-answering system that allows users to upload PDF documents and ask questions about their contents.

The application uses Retrieval-Augmented Generation (RAG) to retrieve relevant information from documents before generating answers with Google Gemini.

## 🚀 Live Demo

Coming soon...

## ✨ Features

- 📄 PDF document upload
- 📚 Multiple PDF support
- 🔎 Semantic search using Sentence Transformers
- ⚡ FAISS vector database
- 🎯 Cross-Encoder reranking
- 🤖 Google Gemini-powered answers
- 💬 Conversation memory
- 🛡️ Hallucination protection
- 📑 Source/page references
- 📊 RAG evaluation metrics
- 🔍 Retrieval score threshold
- 🧩 Improved text chunking

## 🛠️ Tech Stack

- Python
- Streamlit
- PyMuPDF
- Sentence Transformers
- FAISS
- Cross-Encoder
- Google Gemini API
- NumPy
- LangChain Text Splitters

## 🧠 RAG Architecture

```text
PDF Upload
    ↓
PyMuPDF
    ↓
Text Extraction
    ↓
Text Chunking
    ↓
Sentence Transformer Embeddings
    ↓
FAISS Vector Database
    ↓
Relevant Chunk Retrieval
    ↓
Retrieval Threshold
    ↓
Cross-Encoder Reranking
    ↓
Gemini
    ↓
Final Answer
    ↓
Source References + Evaluation