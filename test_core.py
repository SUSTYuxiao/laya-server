"""laya-core + server 全链路测试。

用法：uv run test_core.py
覆盖：
  1. core 层直接调用（choice / noul / score 三种问题类型）
  2. FastAPI /v1/health
  3. FastAPI /v1/predict 正常请求
  4. FastAPI /v1/predict 非法 questions -> 400
  5. FastAPI /v1/systemone JEV 风格请求/响应
  6. jev_adapter 纯函数转换（不依赖模型加载）
"""

import json

from fastapi.testclient import TestClient

import laya_core
from jev_adapter import to_jev_answers, to_jev_response
from server import app

TEXT = "I was billed twice. Please refund the duplicate today."

QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "Which department should handle this request?",
        "criteria": ["billing", "technical", "sales"],
    },
    "refund": {
        "type": "noul",
        "instructions": "Does the customer ask for money back?",
    },
    "frustration": {
        "type": "score",
        "instructions": "How frustrated does the customer sound?",
        "criteria": ["calm and neutral", "concerned but civil", "clearly annoyed", "very angry"],
    },
}


def test_core_predict():
    print("== 1. core 直接调用（进程级单例，本进程首次触发模型加载）==")
    result = laya_core.predict(TEXT, QUESTIONS)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    answers = result["answers"]
    assert answers["department"]["type"] == "choice"
    assert answers["department"]["choice"] in {"billing", "technical", "sales"}
    assert 0.0 <= answers["refund"]["noul"] <= 1.0
    assert 0.0 <= answers["frustration"]["score"] <= 3.0
    assert result["usage"]["input_tokens"] > 0
    print(f"   department -> {answers['department']['choice']}")
    print(f"   refund     -> {answers['refund']['noul']} (P(true))")
    print(f"   frustration-> {answers['frustration']['score']} / 3\n")


def test_server():
    # with 语句触发 lifespan；模型已由上面用例加载，此处复用单例
    with TestClient(app) as client:
        print("== 2. GET /v1/health ==")
        resp = client.get("/v1/health")
        assert resp.status_code == 200, resp.text
        print(f"   {resp.json()}\n")

        print("== 3. POST /v1/predict ==")
        resp = client.post("/v1/predict", json={"text": TEXT, "questions": QUESTIONS})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        print(f"   answers: {json.dumps(body['answers'], ensure_ascii=False)}")
        assert body["model"] == "laya-rl-agent"
        assert body["answers"]["refund"]["type"] == "noul"
        print()

        print("== 4. POST /v1/predict 非法输入 -> 400 ==")
        bad = {
            "text": TEXT,
            "questions": {
                "topic": {"type": "choice", "instructions": "pick one", "criteria": []},
            },
        }
        resp = client.post("/v1/predict", json=bad)
        assert resp.status_code == 400, resp.text
        print(f"   status={resp.status_code} detail={resp.json()['detail']}\n")

        print("== 5. POST /v1/systemone（JEV 风格，例取自 typesafe.ai quickstart）==")
        jev_req = {
            "state": "Hi, I've been trying to connect my Stripe account for 3 days "
            "and the integration keeps failing. I'm losing sales. Please help ASAP.",
            "model": "jev-latest",  # 本地服务忽略该字段，始终用 LAYA_MODEL
            "questions": {
                "department": {
                    "type": "choice",
                    "instructions": "Which team should handle this",
                    "criteria": {
                        "billing": "Payment or subscription issues",
                        "technical": "Bugs or integration problems",
                        "sales": "Pricing or account questions",
                    },
                },
                "frustration": {
                    "type": "score",
                    "instructions": "How frustrated the customer appears",
                    "criteria": [
                        "Calm, just stating facts",
                        "Frustrated but civil",
                        "Very angry, strong language",
                    ],
                },
                "is_urgent": {
                    "type": "noul",
                    "instructions": "The message conveys urgency or time-sensitivity",
                },
            },
        }
        resp = client.post("/v1/systemone", json=jev_req)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        print(f"   {json.dumps(body, ensure_ascii=False, indent=2)}\n")
        answers = body["answers"]
        assert body["model"] == "laya-rl-agent"
        # JEV 契约：无 action 字段
        assert all("action" not in a for a in answers.values())
        # noul 只返回 type + noul
        assert set(answers["is_urgent"]) == {"type", "noul"}
        assert 0.0 <= answers["is_urgent"]["noul"] <= 1.0
        # choice / score 带 confidence + probabilities
        assert answers["department"]["choice"] in {"billing", "technical", "sales"}
        assert "confidence" in answers["department"] and "probabilities" in answers["department"]
        assert 0.0 <= answers["frustration"]["score"] <= 2.0
        assert "legend" in answers["frustration"]


def test_jev_adapter():
    print("== 6. jev_adapter 纯函数转换（构造 laya 格式输入，不加载模型）==")
    laya_result = {
        "model": "laya-rl-agent",
        "answers": {
            "department": {
                "type": "choice",
                "choice": "technical",
                "probabilities": {"billing": 0.15, "technical": 0.85, "sales": 0.0},
                "confidence": 0.78,
                "action": {"act_probability": 1.0},
            },
            "is_urgent": {
                "type": "noul",
                "noul": 0.9,
                "confidence": 0.9,
                "action": {"act_probability": 1.0},
            },
        },
        "usage": {"input_tokens": 100, "output_tokens": 0},
    }
    jev = to_jev_response(laya_result)
    assert set(jev) == {"model", "answers", "usage"}
    assert jev["answers"]["department"] == {
        "type": "choice",
        "choice": "technical",
        "confidence": 0.78,
        "probabilities": {"billing": 0.15, "technical": 0.85, "sales": 0.0},
    }
    assert jev["answers"]["is_urgent"] == {"type": "noul", "noul": 0.9}
    # 单函数同样可用
    assert to_jev_answers(laya_result["answers"]) == jev["answers"]
    print(f"   {json.dumps(jev, ensure_ascii=False)}\n")


if __name__ == "__main__":
    test_jev_adapter()  # 纯函数用例先跑，不依赖模型
    test_core_predict()
    test_server()
    print("ALL TESTS PASSED")
