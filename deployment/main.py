# app/main.py
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from model import Predictor
from schemas import PredictRequest, PredictResponse, SkillOut

predictor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    predictor = Predictor()
    yield

app = FastAPI(title="Job Skill Predictor", lifespan=lifespan)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    try:
        results = predictor.predict(
            title=request.title,
            description=request.description,
            top_k=request.top_k,
            threshold=request.threshold,
        )
        return PredictResponse(
            title=request.title,
            skills=[SkillOut(**r) for r in results],
            total=len(results)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

    
