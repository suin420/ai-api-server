from typing import Union
from fastapi import FastAPI
import model

# 미리 학습된 모델 불러오기
models = model.models

# FastAPI 애플리케이션 생성
app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/predict/{gate}/left/{left}/right/{right}")
def predict(gate: str, left: int, right: int):
    if gate.upper() not in models:
        return {"error": "Invalid logic gate"}
    result = models[gate.upper()].predict([left, right])
    return {"gate": gate, "result": result}

@app.get("/predict/not/{value}")
def predict_not(value: int):
    result = models["NOT"].predict([value])
    return {"gate": "NOT", "result": result}

@app.get("/train/{gate}")
def train(gate: str):
    if gate.upper() not in models:
        return {"error": "Invalid logic gate"}
    models[gate.upper()].train()
    return {"gate": gate, "result": "Trained"}