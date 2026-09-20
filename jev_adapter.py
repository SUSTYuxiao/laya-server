"""JEV 风格适配器：laya 响应 -> typesafe.ai /v1/systemone 契约。

纯函数、零依赖、零状态：不 import core，只依赖「laya answers 格式」这一稳定契约
（见 use.md · 问题定义与返回结构）。因此 core 层可以换成任何输出 laya 格式的实现，
本模块与 /v1/systemone 端点均无需改动；反之 core 也永远不知道 JEV 的存在。

注意：不要在此引入中立中间格式——laya 格式就是项目的内部 IR，JEV 是它的一个视图。
"""


def to_jev_answers(answers: dict) -> dict:
    """laya answers -> JEV answers：去掉 action；noul 只留 type + noul，choice/score 带 confidence。"""
    out = {}
    for qid, ans in answers.items():
        item = {"type": ans["type"]}
        if ans["type"] == "noul":
            item["noul"] = ans["noul"]
        else:
            item["confidence"] = ans["confidence"]
            item["probabilities"] = ans["probabilities"]
            if ans["type"] == "choice":
                item["choice"] = ans["choice"]
            else:
                item["score"] = ans["score"]
                item["legend"] = ans["legend"]
        out[qid] = item
    return out


def to_jev_response(result: dict) -> dict:
    """完整 laya 响应 {model, answers, usage} -> JEV 响应同构结构。"""
    return {
        "model": result["model"],
        "answers": to_jev_answers(result["answers"]),
        "usage": result["usage"],
    }
