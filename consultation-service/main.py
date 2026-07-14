import os
import asyncio
import itertools
import traceback
from typing import List, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from flashrank import Ranker, RerankRequest
from groq import Groq


QDRANT_URL = os.getenv("QDRANT_URL", "https://your-qdrant-cloud-url")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "your-qdrant-api-key")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "fda_collection")
LLM_MODEL = "llama-3.1-8b-instant"


GROQ_KEYS_RAW = os.getenv("GROQ_API_KEYS", "")
GROQ_API_KEYS = [k.strip() for k in GROQ_KEYS_RAW.split(",") if k.strip()] if GROQ_KEYS_RAW else ["default_key_if_local"]


groq_clients = [Groq(api_key=key) for key in GROQ_API_KEYS]
client_cycle = itertools.cycle(groq_clients)


embeddings = None
vector_db = None
base_retriever = None
flashrank_client = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global embeddings, vector_db, base_retriever, flashrank_client
    try:
        print("Loading Embeddings Model...")
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={'device': 'cpu'}
        )

        print("Connecting to Qdrant Cloud...")
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        vector_db = QdrantVectorStore(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding=embeddings
        )
        base_retriever = vector_db.as_retriever(search_kwargs={"k": 6})

        print("Loading Re-ranker (MiniLM-L-12)...")
        flashrank_client = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="/tmp/flashrank_cache")
        
        print("All Models & Databases loaded successfully!")
    except Exception as e:
        print("--- CRITICAL ERROR DURING STARTUP ---")
        print(traceback.format_exc())
    yield
    
    pass

app = FastAPI(
    title="FDA Drug RAG API - Production",
    description="Live FastAPI Backend connected to Qdrant Cloud with Llama 3.3",
    version="3.0",
    lifespan=lifespan
)

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
    
    if embeddings is None:
        raise HTTPException(status_code=503, detail="Embeddings model is not initialized.")
    
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
        
    if base_retriever is None or flashrank_client is None:
         raise HTTPException(status_code=503, detail="Search services are initializing, please try again shortly.")

    try:
       
        initial_docs = base_retriever.invoke(user_query)

        if not initial_docs:
            raise HTTPException(status_code=404, detail="No documents retrieved from Qdrant Cloud Vector DB.")

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

        
        groq_client = next(client_cycle)

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

    except HTTPException as he:
        raise he
    except Exception as e:
        print("--- CRITICAL EXCEPTION IN /ASK ROUTE ---")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
