# 💊 PharmaSmart AI Microservices

This repository contains the AI layer of **PharmaSmart**, a smart pharmacy management platform. It consists of four independent, containerized microservices built with **FastAPI**, **Docker**, and modern AI technologies including **Retrieval-Augmented Generation (RAG)**, **Large Language Models (LLMs)**, **Speech-to-Text**, **Sentiment Analysis**, and **Text Classification**.

The services are designed to run independently while integrating seamlessly with the core **ASP.NET MVC / .NET** backend through REST APIs.

---

# 🏗️ System Architecture

```text
PharmaSmart-AI-Microservices
│
├── consultation-service      # Stateful RAG pipeline
├── voice-assistant-service   # Voice & conversational AI
├── sentiment-service         # Egyptian sentiment analysis
└── complaint-service         # Complaint classification
```

---

# ⚡ Microservices

## 📖 Consultation Service (Stateful RAG)

A Retrieval-Augmented Generation (RAG) service specialized in answering pharmaceutical questions using FDA drug labels.

### Tech Stack

* FastAPI
* LangChain
* HuggingFace Embeddings
* Qdrant Cloud
* FlashRank
* Groq API
* Llama 3.1
* Docker

### Features

* Semantic vector search
* Two-stage retrieval pipeline

  * Qdrant Vector Search
  * FlashRank reranking
* Context-aware response generation
* Guardrails to reduce medical hallucinations
* Multi-key Groq API rotation using Round Robin
* Dockerized deployment

---

## 🎙️ Voice Assistant Service

A voice-enabled virtual assistant capable of understanding and responding in Egyptian Arabic.

### Tech Stack

* FastAPI
* Groq Whisper Large v3
* Llama 3.3 70B
* Docker

### Features

* Speech-to-Text transcription
* Egyptian Arabic conversational assistant
* In-memory audio processing using `io.BytesIO`
* Thread-safe request handling
* Medical safety guardrails
* Stateless architecture

---

## 😊 Sentiment Service

A FastAPI service for analyzing customer and patient feedback.

### Tech Stack

* FastAPI
* HuggingFace Transformers
* PyTorch
* Fine-tuned BERT Model

### Features

* Egyptian Arabic sentiment classification
* Optimized inference using `torch.no_grad()`
* FastAPI Lifespan model initialization
* Cached model pipeline
* Lightweight REST API

---

## 📋 Complaint Service

A text classification service that categorizes pharmacy customer complaints into operational departments.

### Tech Stack

* FastAPI
* HuggingFace Transformers
* PyTorch
* Fine-tuned BERT Model

### Features

* Multi-class complaint classification
* Optimized CPU inference
* Long-text truncation support
* Lightweight deployment
* REST API

---

# 🧠 AI Stack

* FastAPI
* Docker
* LangChain
* HuggingFace Transformers
* HuggingFace Embeddings
* PyTorch
* Groq API
* Llama 3.1
* Llama 3.3
* Whisper Large v3
* FlashRank
* Qdrant Cloud

---

# 🚀 Production Features

* Retrieval-Augmented Generation (RAG)
* Semantic Vector Search
* FlashRank Reranking
* Semantic Retrieval
* Multi-Key API Rotation
* Stateless & Stateful Services
* Dockerized Deployment
* CPU-Optimized Inference
* In-Memory Audio Processing
* RESTful API Integration

---

# 🐳 Docker

Each microservice is fully containerized and can be built and deployed independently.

Example:

```bash
cd consultation-service

docker build -t pharmasmart-consultation .

docker run -p 7860:7860 --env-file .env pharmasmart-consultation
```

---

# ⚙️ Environment Variables

Example:

```env
QDRANT_URL=https://your-qdrant-cloud-url
QDRANT_API_KEY=your-qdrant-api-key
COLLECTION_NAME=fda_collection
GROQ_API_KEYS=gsk_key1,gsk_key2,gsk_key3
```

---
# ▶️ Running Locally

Clone the repository:

```bash
git clone https://github.com/YourUsername/PharmaSmart-AI-Microservices.git

cd PharmaSmart-AI-Microservices
```

Navigate to any service:

```bash
cd consultation-service
```

Build and run with Docker:

```bash
docker build -t pharmasmart-consultation .

docker run -p 7860:7860 --env-file .env pharmasmart-consultation
```
