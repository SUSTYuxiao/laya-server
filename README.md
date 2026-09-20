# laya-server

Laya 决策模型的本地推理封装与 HTTP 服务。基于 [laya-mlx](https://pypi.org/project/laya-mlx/)（Apple Silicon / MLX），输入一段文本 + 一组问题，返回结构化判定：**分类（choice）/ 布尔（noul）/ 打分（score）**。全程本地推理，不调外部 LLM API。

## 目录结构

```
laya-server/
├── laya_core.py    # core 层：模型单例 + 输入校验 + predict 入口（内嵌用法见 core-use.md）
├── jev_adapter.py  # JEV 适配器：laya 格式 -> typesafe.ai 契约（纯函数，不依赖 core）
├── server.py       # server 层：FastAPI，/v1/predict（laya 原生）、/v1/systemone（JEV 风格）、/v1/health
├── test_core.py    # 全链路测试：core 直调 + adapter 纯函数 + TestClient（含 JEV 端点用例）
├── test_laya.py    # 最小示例：不经封装直接调 laya_mlx（对照用）
├── use.md          # HTTP API 用法与设计（端点、请求/响应示例、JEV 端点差异、设计说明）
├── core-use.md     # 内嵌调用指南（不需要网关，Python 应用直接 import laya_core）
├── pyproject.toml  # uv 项目，依赖 laya-mlx / fastapi / uvicorn
└── uv.lock
```

依赖方向：`laya_core`（推理，输出 laya 格式）← `server`（HTTP 编排）→ `jev_adapter`（laya 格式 → JEV 视图）。core 可替换：新实现只要输出对齐 laya 格式，两个推理端点与 adapter 零改动。

## 快速开始

```bash
uv sync                # 安装依赖
uv run test_core.py    # 全链路测试（首次拉模型快照约 1 分钟，仅一次）
uv run server.py       # 起服务 -> http://127.0.0.1:15666
```

## 一分钟上手

两种接入方式：

- **内嵌**（无网关，进程内直调）——详见 [core-use.md](core-use.md)：

```python
import laya_core

result = laya_core.predict(
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
print(result["answers"]["department"]["choice"])  # billing
print(result["answers"]["refund"]["noul"])        # 0.8217 (P(true))
```

- **HTTP**（两个推理端点：`/v1/predict` laya 原生格式、`/v1/systemone` 兼容 typesafe.ai JEV 风格，后者改个 base_url 即可复用现有 JEV 客户端）——详见 [use.md](use.md)：

```bash
curl -s http://127.0.0.1:15666/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{"state":"I was billed twice. Please refund the duplicate today.","questions":{"refund":{"type":"noul","instructions":"Does the customer ask for money back?"}}}'
```

完整的问题定义规范、返回结构、错误码与 API 设计说明见 **[use.md](use.md)**（HTTP）与 **[core-use.md](core-use.md)**（内嵌）。

## 环境变量

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `LAYA_MODEL` | `aac6fef/laya-mlx` | 模型来源：HF repo id 或本地模型目录 |
| `LAYA_HOST` | `0.0.0.0` | 服务监听地址 |
| `LAYA_PORT` | `15666` | 服务监听端口 |
