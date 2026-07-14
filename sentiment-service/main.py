import os
import traceback
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI(
    title="Egyptian Sentiment Analysis API",
    description="Backend service to analyze the sentiment of Egyptian text using kerolos1/analysis-of-Egyptian-sentiments"
)


MODEL_NAME = "kerolos1/analysis-of-Egyptian-sentiments"

print(f"Loading sentiment model: {MODEL_NAME}...")
try:
   
    sentiment_pipeline = pipeline(
        "sentiment-analysis", 
        model=MODEL_NAME, 
        tokenizer=MODEL_NAME,
        device=-1 
    )
    print("Model loaded successfully!")
except Exception as e:
    print("--- CRITICAL ERROR DURING MODEL LOADING ---")
    print(traceback.format_exc())
    sentiment_pipeline = None

class TextRequest(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    label: str
    confidence_score: float

@app.post("/predict", response_model=SentimentResponse)
async def predict_sentiment(request: TextRequest):
    if not sentiment_pipeline:
        raise HTTPException(status_code=500, detail="Model is not initialized properly on the server.")
    
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    
    try:
       
        print(f"Incoming request text: '{request.text}'")
        
        
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

@app.get("/")
async def root():
    return {"message": "Egyptian Sentiment Analysis Service is running!"}

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(app, host="0.0.0.0", port=7860)