from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

from infere import infer  

app = FastAPI()

class InferenceRequest(BaseModel):
    features: List[float]


@app.post("/predict")
def predict(request: InferenceRequest):
    return infer(request.features)
