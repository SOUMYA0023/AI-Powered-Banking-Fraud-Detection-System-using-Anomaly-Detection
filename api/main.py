from typing import Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from core.inference import InferenceEngine


app = FastAPI(title="Fraud Detection API", version="1.0.0")
engine = None


class TransactionRequest(BaseModel):
    Time: float = Field(..., ge=0)
    Amount: float = Field(..., ge=0)
    features: Dict[str, float]
    model: str = "ensemble"
    threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)


@app.on_event("startup")
def startup_event():
    global engine
    engine = InferenceEngine()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: TransactionRequest):
    try:
        transaction = {"Time": payload.Time, "Amount": payload.Amount}
        transaction.update(payload.features)
        result = engine.predict(transaction, model=payload.model, threshold=payload.threshold)
        return result
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
