"""Laya 核心封装：模型单例管理 + 输入校验 + 统一 predict 入口。

两类消费者共用本模块：
- 脚本直接调用：laya_core.predict(text, questions)
- HTTP 服务（server.py）：laya_core.get_core() 取进程级单例

模型在进程内只加载一次；首次 load 会从 HuggingFace 拉快照，之后走本地缓存。
"""

import json
import os
import threading

import laya_mlx as laya

DEFAULT_MODEL_ID = "aac6fef/laya-mlx"

VALID_QTYPES = ("choice", "score", "noul")


class LayaValidationError(ValueError):
    """questions/text 不合法。HTTP 层应转为 400，不要当内部错误处理。"""


def _validate_question(qid, definition):
    """校验单个问题定义，规则与 laya_mlx.Agent._to_internal 对齐但提前报错。"""
    if not isinstance(definition, dict):
        raise LayaValidationError(f"Question {qid!r} must be an object")

    kind = definition.get("type")
    if kind not in VALID_QTYPES:
        raise LayaValidationError(
            f"Question {qid!r}: unknown type {kind!r}; expected one of {list(VALID_QTYPES)}"
        )

    instructions = definition.get("instructions")
    if instructions is None:
        raise LayaValidationError(f"Question {qid!r} is missing instructions")
    if not isinstance(instructions, (str, list, dict)):
        raise LayaValidationError(f"Question {qid!r}: instructions must be a string or JSON value")

    criteria = definition.get("criteria")
    if kind == "choice":
        if isinstance(criteria, list):
            if not criteria or not all(isinstance(c, str) for c in criteria):
                raise LayaValidationError(f"Question {qid!r}: choice criteria must be non-empty strings")
            if len(set(criteria)) != len(criteria):
                raise LayaValidationError(f"Question {qid!r}: choice labels must be unique")
        elif not isinstance(criteria, dict) or not criteria:
            raise LayaValidationError(
                f"Question {qid!r}: choice criteria must be a non-empty list or object"
            )
    elif kind == "score":
        if not isinstance(criteria, list) or not criteria:
            raise LayaValidationError(
                f"Question {qid!r}: score criteria must be a non-empty ordered list of levels"
            )
    elif criteria is not None and not isinstance(criteria, dict):
        raise LayaValidationError(f"Question {qid!r}: noul criteria must be an object with false/true descriptions")


def validate_input(text, questions):
    if not isinstance(text, str) or not text.strip():
        raise LayaValidationError("text must be a non-empty string")
    if not isinstance(questions, dict) or not questions:
        raise LayaValidationError("questions must be a non-empty object keyed by question id")
    for qid, definition in questions.items():
        _validate_question(qid, definition)


class LayaCore:
    """持有已加载 Agent 的推理入口；权重加载后冻结，进程内复用同一个实例。"""

    def __init__(self, model_id=None, **kwargs):
        self.model_id = model_id or os.environ.get("LAYA_MODEL", DEFAULT_MODEL_ID)
        self.agent = laya.load(self.model_id, **kwargs)

    def predict(self, text, questions):
        """输入校验后透传 agent.predict，返回 {model, answers, usage} 原始结构。"""
        validate_input(text, questions)
        return self.agent.predict(text, questions)


_core = None
_core_lock = threading.Lock()


def get_core() -> LayaCore:
    """进程级单例：双检锁，首个调用方触发模型加载。"""
    global _core
    if _core is None:
        with _core_lock:
            if _core is None:
                _core = LayaCore()
    return _core


def predict(text, questions) -> dict:
    """脚本便捷入口：等价于 get_core().predict(text, questions)。"""
    return get_core().predict(text, questions)


if __name__ == "__main__":
    # 最小自测：uv run laya_core.py
    result = predict(
        "I was billed twice. Please refund the duplicate today.",
        {
            "department": {
                "type": "choice",
                "instructions": "Which department should handle this request?",
                "criteria": ["billing", "technical", "sales"],
            },
            "refund": {"type": "noul", "instructions": "Does the customer ask for money back?"},
        },
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
