import os
import traceback
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI(
    title="Kerolos Pharmacy Complaints V3-Ultra API",
    description="Backend API for classifying pharmacy complaints into 10 categories.",
    version="3.0"
)

MODEL_NAME = "kerolos1/pharmacy-complaints-v3-ultra"
classifier = None

print(f"⏳ Downloading and loading model from Hub: {MODEL_NAME}...")
try:
  
    classifier = pipeline(
        "text-classification", 
        model=MODEL_NAME, 
        tokenizer=MODEL_NAME, 
        device=-1
    )
    print("✅ Model loaded successfully and ready for inference!")
except Exception as e:
    print("--- CRITICAL ERROR DURING COMPLAINTS MODEL LOADING ---")
    print(traceback.format_exc())
    classifier = None

class ComplaintRequest(BaseModel):
    text: str

@app.get("/")
def home():
    return {"status": "healthy", "message": "Kerolos Pharmacy V3-Ultra API is running successfully!"}

@app.post("/predict")
def predict_complaint(payload: ComplaintRequest):
    global classifier
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet or unavailable.")
        
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")
        
    try:
        print(f"Incoming complaint text: '{payload.text}'")
        
        
        result = classifier(payload.text, truncation=True)[0]
        
        print(f"Raw Complaints Model Output: Label={result['label']}, Score={result['score']}")
                
        return {
            "complaint_text": payload.text,
            "classification": str(result["label"]),
            "confidence_score": round(result["score"] * 100, 2)
        }
    except Exception as e:
        print("--- CRITICAL EXCEPTION IN COMPLAINTS /PREDICT ROUTE ---")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)