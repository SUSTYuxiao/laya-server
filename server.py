"""FastAPI 服务：以 HTTP 方式暴露 laya-core 的 predict 能力。

启动：uv run server.py            （默认 0.0.0.0:15666）
     LAYA_MODEL=<repo> LAYA_PORT=9000 uv run server.py
"""

import os
from contextlib import asynccontextmanager
from typing import Any, Literal

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import laya_core
from jev_adapter import to_jev_answers


class Question(BaseModel):
    """单问题定义；criteria 结构依赖 type，细校验下沉到 core 层统一报错。"""

    type: Literal["choice", "score", "noul"]
    instructions: str
    criteria: Any = None


class PredictRequest(BaseModel):
    text: str
    questions: dict[str, Question]


class PredictResponse(BaseModel):
    model: str
    answers: dict[str, Any]
    usage: dict[str, int]


class JevRequest(BaseModel):
    """JEV 风格请求（对齐 typesafe.ai /v1/systemone）：state + 可选 model + questions。"""

    state: str
    model: str | None = None  # 兼容 JEV 客户端 SDK 传入的模型名；本地服务固定用 LAYA_MODEL
    questions: dict[str, Question]


class JevResponse(BaseModel):
    model: str
    answers: dict[str, Any]
    usage: dict[str, int]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动即加载模型（首拉快照耗时长，放在启动期而非首个请求），失败则拒绝启动
    laya_core.get_core()
    yield


app = FastAPI(title="laya-server", version="0.1.0", lifespan=lifespan)


@app.get("/v1/health")
def health():
    return {"status": "ok", "model": laya_core.get_core().model_id}


@app.post("/v1/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    questions = {qid: q.model_dump() for qid, q in req.questions.items()}
    try:
        # 同步 CPU/GPU 推理；FastAPI 将 def 端点丢入线程池，不阻塞事件循环
        result = laya_core.get_core().predict(req.text, questions)
    except laya_core.LayaValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return PredictResponse(**result)


@app.post("/v1/systemone", response_model=JevResponse)
def systemone(req: JevRequest) -> JevResponse:
    """JEV 风格端点：请求/响应对齐 typesafe.ai /v1/systemone，推理走同一 core。"""
    questions = {qid: q.model_dump() for qid, q in req.questions.items()}
    try:
        result = laya_core.get_core().predict(req.state, questions)
    except laya_core.LayaValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JevResponse(
        model=result["model"],
        answers=to_jev_answers(result["answers"]),
        usage=result["usage"],
    )


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=os.environ.get("LAYA_HOST", "0.0.0.0"),
        port=int(os.environ.get("LAYA_PORT", "15666")),
    )
