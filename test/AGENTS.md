# test 目录

## 职责

- 存放手动验证脚本，覆盖 core、server 与第三方模型的对照路径。
- 必须在项目根目录运行，导入根目录模块，不包含运行期状态。

## 文件

- `__init__.py`：标记测试包，使 `python -m test.*` 从项目根解析导入。
- `test_core.py`：全链路手动测试，覆盖 core 直调、adapter、FastAPI TestClient 与 JEV 端点。
- `test_laya.py`：直接调用 `laya_mlx` 的最小对照脚本，用于校验项目封装行为。

## 运行

```bash
uv run python -m test.test_core
uv run python -m test.test_laya
```
