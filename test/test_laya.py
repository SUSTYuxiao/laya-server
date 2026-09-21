"""laya-mlx 本地推理测试：分类 + 布尔判断两类问题。

首次运行会从 HuggingFace 下载 aac6fef/laya-mlx 模型快照，之后走本地缓存。
运行：uv run python -m test.test_laya
"""

import json

import laya_mlx as laya

MODEL_ID = "aac6fef/laya-mlx"


def main() -> None:
    # 首次 load 触发 snapshot_download；后续运行直接命中本地缓存
    agent = laya.load(MODEL_ID)

    result = agent.predict(
        "I was billed twice. Please refund the duplicate today.",
        {
            "department": {
                "type": "choice",
                "instructions": "Which department should handle this request?",
                "criteria": ["billing", "technical", "sales"],
            },
            "refund": {
                "type": "noul",
                "instructions": "Does the customer ask for money back?",
            },
        },
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))

    answers = result["answers"]
    print(f"\ndepartment -> {answers['department']['choice']}")
    print(f"refund     -> {answers['refund']['noul']} (P(true))")


if __name__ == "__main__":
    main()
