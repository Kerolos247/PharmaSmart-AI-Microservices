import os
import traceback
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch  
from transformers import pipeline

MODEL_NAME = "kerolos1/analysis-of-Egyptian-sentiments"
sentiment_pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global sentiment_pipeline
    print(f"⏳ Loading sentiment model from Hub: {MODEL_NAME}...")
    try:
        
        with torch.no_grad():
            sentiment_pipeline = pipeline(
                "sentiment-analysis", 
                model=MODEL_NAME, 
                tokenizer=MODEL_NAME,
                device=-1 
            )
        print("✅ Sentiment model loaded successfully and ready!")
    except Exception as e:
        print("--- CRITICAL ERROR DURING MODEL LOADING ---")
        print(traceback.format_exc())
        sentiment_pipeline = None
    yield
    
    sentiment_pipeline = None

app = FastAPI(
    title="Egyptian Sentiment Analysis API",
    description="Backend service to analyze the sentiment of Egyptian text using kerolos1/analysis-of-Egyptian-sentiments",
    version="3.0",
    lifespan=lifespan  
)

class TextRequest(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    label: str
    confidence_score: float

@app.get("/")
async def root():
    return {"status": "healthy", "message": "Egyptian Sentiment Analysis Service is running!"}

@app.post("/predict", response_model=SentimentResponse)
async def predict_sentiment(request: TextRequest):
    global sentiment_pipeline
    if not sentiment_pipeline:
        raise HTTPException(status_code=503, detail="Model is not initialized properly or currently unavailable.")
    
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    
    try:
        print(f"Incoming request text: '{request.text}'")
        
        
        with torch.no_grad():
            result = sentiment_pipeline(request.text, truncation=True)[0]
        
        print(f"Raw Model Output: Label={result['label']}, Score={result['score']}")
        
        confidence = round(result['score'] * 100, 2)
        
        return SentimentResponse(
            label=str(result['label']),
            confidence_score=confidence
        )
    except Exception as e:
        print("--- CRITICAL EXCEPTION IN /PREDICT ROUTE ---")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Prediction internal error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
