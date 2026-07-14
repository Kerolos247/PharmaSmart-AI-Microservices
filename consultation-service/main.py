import os
import asyncio
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import itertools  


from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from flashrank import Ranker, RerankRequest
from groq import Groq

app = FastAPI(
    title="FDA Drug RAG API - Production",
    description="Live FastAPI Backend connected to Qdrant Cloud with Llama 3.3",
    version="3.0"
)


GROQ_API_KEYS = [
    "key1",  
    "key2",  
    "key3", 
    "key4",  
    "key5",  
    "key6",  
    "key7", 
    "key8"   
]


api_key_cycle = itertools.cycle(GROQ_API_KEYS)

QDRANT_URL = "URL_QDRANT"
QDRANT_API_KEY = "ApiKey_QDRANT"
COLLECTION_NAME = "Name_collection"

LLM_MODEL = "llama-3.1-8b-instant"


print("🧠 Loading Embeddings Model...")
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={'device': 'cpu'}
)


print("🌐 Connecting to Qdrant Cloud...")
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
vector_db = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding=embeddings
)
base_retriever = vector_db.as_retriever(search_kwargs={"k": 6})


print("🎯 Loading Re-ranker (MiniLM-L-12)...")
flashrank_client = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="/tmp/flashrank_cache")


class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    question: str
    answer: str
    source_documents: List[Dict[str, Any]]

class EmbedRequest(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {"status": "running", "message": "FDA Drug RAG API (Qdrant Cloud) is live 24/7!"}

@app.post("/embed")
async def get_embedding(request: EmbedRequest):
    text_to_embed = request.text.strip()
    if not text_to_embed:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    try:
        vector = embeddings.embed_query(text_to_embed)
        return {"vector": vector}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask", response_model=QueryResponse)
async def ask_drug_bot(request: QueryRequest):
    user_query = request.question.strip()

    if not user_query:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        initial_docs = base_retriever.invoke(user_query)

        if not initial_docs:
            raise Exception("No documents retrieved from Qdrant Cloud Vector DB.")

        pass_passages = []
        for i, doc in enumerate(initial_docs):
            pass_passages.append({
                "id": i,
                "text": doc.page_content,
                "meta": doc.metadata
            })

        rerank_req = RerankRequest(query=user_query, passages=pass_passages)
        reranked_results = flashrank_client.rerank(rerank_req)

        top_2_results = reranked_results[:2]

        context_list = []
        source_docs_metadata = []

        for res in top_2_results:
            context_list.append(res['text'])
            source_docs_metadata.append({
                "content": res['text'],
                "metadata": res['meta']
            })

        combined_context = "\n\n---\n\n".join(context_list)

        system_prompt = (
            "You are an expert medical AI assistant specialized in FDA drug labels.\n"
            "Your task is to answer the user's question accurately based ONLY on the provided context below.\n"
            "If the context does not contain the answer, say 'I cannot find the answer in the official FDA database documents provided.' Do not make up information.\n\n"
            f"Context:\n{combined_context}"
        )

        
        current_api_key = next(api_key_cycle)
        groq_client = Groq(api_key=current_api_key)

        completion = groq_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ],
            temperature=0.0,
            max_tokens=500
        )

        ai_answer = completion.choices[0].message.content

        return QueryResponse(
            question=user_query,
            answer=ai_answer,
            source_documents=source_docs_metadata
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))