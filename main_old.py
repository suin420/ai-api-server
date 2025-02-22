from typing import Union # 타입 힌트, 여러 개의 타입을 가질 수 있도록 정의
from fastapi import FastAPI
import model # model.py를 가져온다 

model = model.AndModel() # model.py에서 AndModel클래스의 인스턴스를 가져온다 

app = FastAPI()


@app.get("/") # FastAPI의 라우팅을 정의하는 데코레이터를 사용한 엔드포인트 # 루트 경로 설정
def read_root():
    return {"Hello": "World"}

# endpoint - item_id 경로 설정하면 서버 요청 가능 
@app.get("/items/{item_id}") # FastAPI의 라우팅을 정의하는 데코레이터를 사용한 엔드포인트 # 동적 경로 설정 
def read_item(item_id: int, q: Union[str, None] = None): # 타입 힌트(int, Union)
    return {"item_id": item_id, "q": q}

@app.get("/predict/left/{left}/right/{right}") 
def read_item(left: int, right: int):
    result = model.predict([left, right])
    return {"result": result}

@app.get("/train")
def train():
    model.train()
    return {"result": "OK"}

# fastapi dev main.py # fastapi: 실행 # dev: 개발 모드로 서버 실행 
# 8000 port: FastAPI/Uvicorn 개발 서버 (기본)