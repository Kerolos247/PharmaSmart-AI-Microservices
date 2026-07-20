# 💊 PharmaSmart AI Microservices

This repository contains the AI layer of **PharmaSmart**, consisting of four independent, containerized microservices built with **FastAPI**, **Docker**, and modern AI technologies. The services integrate with the core **ASP.NET Core MVC** application through REST APIs to provide intelligent pharmaceutical capabilities.

---

# 🏗️ System Architecture

![PharmaSmart AI Microservices Architecture](https://raw.githubusercontent.com/Kerolos247/PharmaSmart-AI-Microservices/main/Diagram_Ai.png)

---

# ⚡ Microservices

## 📖 Consultation Service (Stateful RAG)

Answers pharmaceutical questions using FDA drug labels through a Retrieval-Augmented Generation (RAG) pipeline.

### Technologies

- FastAPI
- LangChain
- HuggingFace Embeddings
- Qdrant
- FlashRank
- Groq API
- Llama 3.3
- Docker

### Highlights

- FlashRank reranking
- Context-aware response generation
- Medical safety guardrails
- Multi-key Groq API rotation

---

## 🎙️ Voice Assistant Service

Egyptian Arabic voice assistant supporting speech-to-text and AI-powered conversations.

### Technologies

- FastAPI
- Groq Whisper Large v3
- Llama 3.3
- Docker

### Highlights

- Speech-to-text
- Egyptian Arabic conversations
- Stateless architecture

---

## 😊 Sentiment Service

Analyzes Egyptian Arabic customer feedback using a fine-tuned BERT model.

### Technologies

- FastAPI
- HuggingFace Transformers
- PyTorch
- Fine-tuned BERT

### Highlights

- Sentiment classification

---

## 📋 Complaint Service

Classifies customer complaints into pharmacy operational categories.

### Technologies

- FastAPI
- HuggingFace Transformers
- PyTorch
- Fine-tuned BERT

### Highlights

- Multi-class classification

---

# 🛠️ AI Stack

- FastAPI
- Docker
- LangChain
- HuggingFace Transformers
- HuggingFace Embeddings
- PyTorch
- Qdrant
- FlashRank
- Groq API
- Llama 3.3
- Whisper Large v3

---

# 📦 Repository Notice

This repository is published for portfolio and code review purposes.

Configuration files, API keys, model artifacts, and deployment-specific settings have been intentionally excluded.
