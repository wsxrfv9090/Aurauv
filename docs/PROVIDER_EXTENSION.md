# Aurauv Provider 扩展指南

## 1. 不要把包特有逻辑放进核心

核心只负责：

- uv 命令边界；
- workspace / submodule 环境所有权；
- route 选择与状态；
- fallback 授权；
- Python 与 uv 生命周期；
- provider 调度。

包名、平台条件、wheel/index、系统前提和运行时验证属于 provider。

## 2. Provider 接口

继承：

```python
from aurauv.providers.base import Provider
```

必须认真实现：

- `validate_contract()`：只读验证 pyproject/uv 声明，不能修改 lock；
- `detect()`：根据机器返回 route option；
- `preflight(option)`：环境变更前验证系统条件；
- `infer_installed(python)`：从现有环境推断 option；
- `verify(python, option, cwd)`：安装后验证真实运行时；
- `protected_packages()`：定向 upgrade/repair 的 package family；
- `current_target_install_args(option)`：Colab/current interpreter 使用的额外 `uv pip install` 参数。

然后在 `src/aurauv/providers/registry.py` 注册。

## 3. 新 route 还是挂接旧 route

如果新包与 accelerator 必须严格同选，例如某包的 CUDA wheel 必须和 PyTorch CUDA 一致，可以挂接 `accelerator` route。

如果它是独立维度，例如 Linux 上启用 Triton、其他平台禁用，应建立独立 route：

```toml
[tool.aurauv.routes.triton]
default = "auto"
detector = "triton"
providers = ["triton"]
fallbacks = { enabled = "disabled" }

[tool.aurauv.routes.triton.options.enabled]
extras = ["triton-enabled"]

[tool.aurauv.routes.triton.options.disabled]
extras = ["triton-disabled"]
```

不要把所有维度塞入 `cpu/mps/cuda/linux-cuda/linux-cpu/...` 的笛卡尔积。

## 4. GUI/headless 互斥 wheel

未来若处理两个 distribution 互斥、但 import name 相同的包：

1. route option 分别映射两个 extras；
2. `tool.uv.conflicts` 声明 extras 互斥；
3. provider 验证两个 distribution 不会同时存在；
4. 根据运行场景选择 full/headless；
5. 验证公共 import API，而不是仅检查安装元数据；
6. Colab/current target 要明确替换语义，不能并装。

在确定具体包之前，不应加入猜测性的 hard-coded rule。

## 5. 测试最低要求

每个 provider 至少应包含：

- 正确合同通过；
- 不完整/冲突合同失败；
- 自动检测；
- 显式 option；
- preflight 失败不修改环境；
- fallback 无授权失败、有授权成功；
- 安装后 backend/version/runtime 验证；
- state machine 在机器签名变化后重新检测；
- current interpreter 路径。
