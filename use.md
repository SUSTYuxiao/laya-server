# use.md — HTTP API 用法与设计

Laya 是跑在 Apple Silicon（MLX）上的本地决策小模型：输入一段文本 + 一组问题，返回结构化判定（分类 / 布尔 / 打分），不依赖任何外部 LLM API。

本文档聚焦 **server 层**（`server.py`，FastAPI）的 HTTP 调用；不需要网关、想在 Python 应用里直接内嵌推理的，见 **[core-use.md](core-use.md)**（core 层：`laya_core.py`）。两层共用同一模型单例与问题定义格式。

## 快速开始

```bash
uv sync                          # 安装依赖
uv run python -m test.test_core # 跑通 core + server 全链路测试（首次会拉模型快照，约 1 分钟）
uv run server.py                 # 起服务，默认 0.0.0.0:15666
```

环境变量：

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `LAYA_MODEL` | `aac6fef/laya-mlx` | 模型来源：HF repo id 或本地模型目录 |
| `LAYA_HOST` | `0.0.0.0` | 服务监听地址 |
| `LAYA_PORT` | `15666` | 服务监听端口 |

---

## 一、问题定义（所有端点共用）

`predict` 的文本之外，请求体携带「问题集」：`{问题id: 问题定义}`。每个问题必填 `type` 和 `instructions`，`criteria` 的结构随 `type` 变化：

### type = "choice"（单选分类）

`criteria` 是候选列表（或 `标签: 描述` 对象，描述能提升准确度），返回得分最高的标签 + 各标签概率。

```python
{
    "type": "choice",
    "instructions": "Which department should handle this request?",
    "criteria": ["billing", "technical", "sales"],
}
```

### type = "noul"（布尔判断）

`criteria` 可省略；若给，必须是 `{"true": 描述, "false": 描述}` 帮助模型对齐语义。返回 `noul` = P(true)。

```python
{
    "type": "noul",
    "instructions": "Does the customer ask for money back?",
}
```

### type = "score"（等级打分）

`criteria` 是**有序**等级列表（从低到高），返回期望分值（0 起）+ 各等级概率 + legend。

```python
{
    "type": "score",
    "instructions": "How frustrated does the customer sound?",
    "criteria": ["calm and neutral", "concerned but civil", "clearly annoyed", "very angry"],
}
```

### 校验规则（校验失败返回 400）

- `text` / `state`：非空字符串
- `questions`：非空对象
- `choice`：criteria 非空、标签唯一
- `score`：criteria 非空有序列表
- `noul`：criteria 若给必须是对象

---

## 二、HTTP 端点（laya 原生格式）

### `GET /v1/health`

```bash
curl http://127.0.0.1:15666/v1/health
# {"status":"ok","model":"aac6fef/laya-mlx"}
```

### `POST /v1/predict`

请求体 `{"text": ..., "questions": {问题id: 问题定义}}`，问题定义规范同上。

```bash
curl -s http://127.0.0.1:15666/v1/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "I was billed twice. Please refund the duplicate today.",
    "questions": {
      "department": {
        "type": "choice",
        "instructions": "Which department should handle this request?",
        "criteria": ["billing", "technical", "sales"]
      },
      "refund": {
        "type": "noul",
        "instructions": "Does the customer ask for money back?"
      }
    }
  }'
```

响应：

```json
{
  "model": "laya-rl-agent",
  "answers": {
    "department": {
      "type": "choice",
      "choice": "billing",
      "probabilities": {"billing": 0.9386, "technical": 0.0453, "sales": 0.0161},
      "confidence": 0.7577,
      "action": {"act_probability": 1.0}
    },
    "refund": {
      "type": "noul",
      "noul": 0.8217,
      "confidence": 0.8217,
      "action": {"act_probability": 1.0}
    },
    "frustration": {
      "type": "score",
      "score": 2.53,
      "legend": {"0": "calm and neutral", "1": "concerned but civil", "2": "clearly annoyed", "3": "very angry"},
      "probabilities": {"0": 0.0171, "1": 0.0559, "2": 0.3068, "3": 0.6202},
      "confidence": 0.3582,
      "action": {"act_probability": 1.0}
    }
  },
  "usage": {"input_tokens": 127, "output_tokens": 0}
}
```

字段速查：`choice` = 胜出标签；`noul` = P(true)；`score` = 期望分值（0 ～ len(criteria)-1）；`confidence` = 该答案的置信度；`usage.output_tokens` 恒为 0（判别式模型，非生成）。

Python 客户端等价写法：

```python
import requests

resp = requests.post(
    "http://127.0.0.1:15666/v1/predict",
    json={"text": "...", "questions": {...}},
)
resp.raise_for_status()
answers = resp.json()["answers"]
```

### 错误码

| 状态码 | 场景 |
| --- | --- |
| 400 | text/questions 校验失败，`detail` 给出具体原因 |
| 422 | 请求体不符合 schema（缺字段、type 拼错等，FastAPI/pydantic 默认行为） |

---

## 三、JEV 风格端点（`POST /v1/systemone`）

对齐 [typesafe.ai /v1/systemone](https://docs.typesafe.ai/introduction/quickstart) 的请求/响应契约，推理与 `/v1/predict` 走同一个 core，仅格式不同。已有 JEV 客户端代码改个 base_url 即可切到本地。

### 请求

- `state`（对应 `/v1/predict` 的 `text`）：待分析文本
- `model`：可选，兼容 JEV SDK 的必填字段；本地服务固定使用 `LAYA_MODEL`，该字段被忽略
- `questions`：问题定义规范与 `/v1/predict` 完全相同

```bash
curl -s http://127.0.0.1:15666/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{
    "state": "Hi, I have been trying to connect my Stripe account for 3 days and the integration keeps failing. Please help ASAP.",
    "model": "jev-latest",
    "questions": {
      "department": {
        "type": "choice",
        "instructions": "Which team should handle this",
        "criteria": {
          "billing": "Payment or subscription issues",
          "technical": "Bugs or integration problems",
          "sales": "Pricing or account questions"
        }
      },
      "frustration": {
        "type": "score",
        "instructions": "How frustrated the customer appears",
        "criteria": ["Calm, just stating facts", "Frustrated but civil", "Very angry, strong language"]
      },
      "is_urgent": {
        "type": "noul",
        "instructions": "The message conveys urgency or time-sensitivity"
      }
    }
  }'
```

### 响应

```json
{
  "model": "laya-rl-agent",
  "answers": {
    "department": {
      "type": "choice",
      "choice": "technical",
      "confidence": 0.78,
      "probabilities": {"billing": 0.15, "technical": 0.85, "sales": 0.0}
    },
    "frustration": {
      "type": "score",
      "score": 1.0,
      "confidence": 1.0,
      "legend": {"0": "Calm, just stating facts", "1": "Frustrated but civil", "2": "Very angry, strong language"},
      "probabilities": {"0": 0.0, "1": 1.0, "2": 0.0}
    },
    "is_urgent": {"type": "noul", "noul": 1.0}
  },
  "usage": {"input_tokens": 392, "output_tokens": 0}
}
```

### 与 `/v1/predict` 的差异

| 差异点 | `/v1/predict`（laya 原生） | `/v1/systemone`（JEV 风格） |
| --- | --- | --- |
| 文本字段名 | `text` | `state` |
| `model` 请求字段 | 无 | 可选（被忽略，固定用 `LAYA_MODEL`） |
| answers 中的 `action` | 有（`act_probability`） | 无 |
| noul 的 `confidence` | 有 | 无（只返回 `type` + `noul`） |
| 其余（校验规则、错误码 400/422） | 相同 | 相同 |

### SDK demo

```bash
uv sync
uv run server.py
```

另开一个终端，运行 Python SDK demo：

配置优先级为：命令行参数 > 环境变量 > 默认值。默认连接 `http://127.0.0.1:15666`，默认 API key 为 `local`。

显式指定参数：

```bash
uv run sdk_demo.py --base-url http://127.0.0.1:15666 --api-key local
```

或使用环境变量兜底：

```bash
export TYPESAFE_BASE_URL=http://127.0.0.1:15666
export TYPESAFE_API_KEY=local
uv run sdk_demo.py
```

demo 位于根目录 [sdk_demo.py](sdk_demo.py)，用 `Choice` / `Noul` / `Score` 覆盖三种问题类型。`local` 只是 SDK 请求本地服务所需的占位 key；访问外部官方 API 时仍需设置真实 `TYPESAFE_API_KEY`。

---

## 四、API 设计说明

- **分层与依赖方向**：`laya_core`（推理，输出 laya 格式）← `server`（HTTP 编排）→ `jev_adapter`（laya 格式 → JEV 视图）。JEV 转换独立在 [jev_adapter.py](jev_adapter.py)：纯函数、零 import、零状态，只依赖「laya 响应格式」这一稳定契约，不依赖任何具体 core 实现；core 也永远不知道 JEV 的存在。
- **契约锚在 laya 格式，不引入中立中间格式**：laya 格式就是项目内部 IR。若未来把 core 换成其他实现（另一个推理引擎、远程服务），只需新实现输出对齐 laya 格式（差异在它内部消化），`/v1/predict`、`/v1/systemone` 与 adapter 均零改动；再垫一层中立格式只会让转换链翻倍。
- **单例加载**：模型权重加载约秒级、加载后冻结，进程内多实例没有意义。`get_core()` 用双检锁保证并发下只加载一次。
- **启动即加载**：server 在 lifespan 里加载模型，首个快照下载（约 1 分钟，一次性）发生在启动期而不是首个请求，避免线上冷请求超时；加载失败直接拒绝启动。
- **校验前置到 core 层**：schema 校验（pydantic）只管字段形状，语义校验（criteria 随 type 变化的规则）集中在 `laya_core.validate_input`，脚本调用和 HTTP 调用报错一致，HTTP 层只做 `LayaValidationError -> 400` 的翻译。
- **同步推理、异步服务**：MLX 推理是同步 GPU 操作，端点用 `def`（FastAPI 丢线程池），不阻塞事件循环。
