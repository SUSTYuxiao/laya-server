# core-use.md — 内嵌调用（core 层）

面向**不需要 HTTP 网关**的场景：Python 应用直接 `import laya_core`，在进程内完成本地推理（Apple Silicon / MLX）。需要 API 方式调用的见 [use.md](use.md)。

## 快速开始

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

print(result["answers"]["department"]["choice"])   # "billing"
print(result["answers"]["refund"]["noul"])         # 0.8217 (P(true))
```

`laya_core.predict(text, questions)` 等价于 `laya_core.get_core().predict(...)`。

## 实例管理

- **进程级单例**：`get_core()` 双检锁保证模型只加载一次，之后所有调用复用同一实例。首次调用耗时 = 权重加载（秒级）；若本地无缓存快照还会先从 HuggingFace 下载（约 1 分钟，仅一次）。
- **模型来源**：默认 `aac6fef/laya-mlx`，可用环境变量 `LAYA_MODEL` 覆盖（HF repo id 或本地模型目录均可）。
- **自定义加载参数**：直接构造 `LayaCore`，参数透传 `laya_mlx.load`：

```python
core = laya_core.LayaCore(model_id="aac6fef/laya-mlx", dtype="float32")
result = core.predict(text, questions)
```

- **并发**：`predict` 可多线程并发调用（FastAPI 的同步端点就是这个模式）。**多进程**各自加载一份模型——`uvicorn --workers N`、`multiprocessing` 场景注意内存翻倍。
- 权重加载后实例冻结；换模型或参数需要新建 `LayaCore`，不影响单例。

## 问题定义

三种问题类型，`criteria` 结构随 `type` 变化：

| type | 语义 | criteria | 返回关键字段 |
| --- | --- | --- | --- |
| `choice` | 单选分类 | 非空候选 `list[str]`，或 `标签: 描述` 对象 | `choice`（胜出标签）+ `probabilities` |
| `noul` | 布尔判断 | 可省略；若给须为 `{"true": ..., "false": ...}` | `noul`（P(true)） |
| `score` | 等级打分 | **有序**等级列表（从低到高） | `score`（期望分值，0 ～ len-1）+ `legend` |

```python
{
    "frustration": {
        "type": "score",
        "instructions": "How frustrated does the customer sound?",
        "criteria": ["calm and neutral", "concerned but civil", "clearly annoyed", "very angry"],
    }
}
```

格式与上游 `laya_mlx` 完全兼容，官方 presets 可直接传入：

```python
from laya_mlx import triage_questions

result = laya_core.predict(text, triage_questions())
```

完整规范（校验规则、choice 的 dict 形式示例等）见 [use.md · 问题定义](use.md#一问题定义所有端点共用)。

## 返回结构

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

字段速查：`choice` = 胜出标签；`noul` = P(true)；`score` = 期望分值；`confidence` = 该答案置信度；`usage.output_tokens` 恒为 0（判别式模型，非生成）。

## 错误处理

输入不合法抛 `laya_core.LayaValidationError`（`ValueError` 子类），消息带问题 id 与原因，应在调用侧捕获：

```python
try:
    result = laya_core.predict(text, questions)
except laya_core.LayaValidationError as exc:
    print(f"bad input: {exc}")
```

校验规则：`text` 非空字符串；`questions` 非空对象；`choice` 的 criteria 非空且标签唯一；`score` 的 criteria 非空有序列表；`noul` 的 criteria 若给必须是对象。模型内部错误（如非有限输出）会抛 `FloatingPointError`，按 `laya_mlx` 提示可改用 `dtype="float32"` 重试。
