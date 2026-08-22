# 新增 provider 指南

## 1. 什么时候应该新增 provider

满足任一条件时值得使用 provider：

- 同一逻辑依赖在不同机器上需要不同 distribution/index/extra；
- 安装前需要驱动、OS、compiler、ABI 或 runtime 检查；
- import 成功仍不足以证明功能可用；
- 一组包需要作为同一个升级与修复单元；
- 多个 distribution 互斥但共享 import name。

普通纯 Python 条件依赖优先使用 PEP 508 markers 和 uv，不应为了“看起来统一”写 provider。

## 2. 实现接口

在 `src/aurauv/providers/` 新建模块并继承：

```python
from aurauv.providers.base import Provider

class ExampleProvider(Provider):
    def validate_contract(self) -> dict:
        ...

    def detect(self):
        ...

    def infer_installed(self, python):
        ...

    def machine_fingerprint(self) -> dict:
        ...

    def preflight(self, option: str) -> dict:
        ...

    def verify(self, python, *, option: str, cwd):
        ...

    def protected_packages(self) -> tuple[str, ...]:
        ...
```

在 `providers/registry.py` 注册 provider type。

## 3. validate_contract

必须尽早拒绝不一致的 pyproject，例如：

- route option 缺少对应 extra；
- routed package 又出现在 unconditional dependencies；
- index 名不存在或不是 explicit；
- root forwarding extra 没有引用 member provider extra；
- mutually exclusive extras 未进入同一 `tool.uv.conflicts` 集合；
- import mapping 不完整。

不要在 provider 中修改 pyproject；配置错误应由用户显式编辑。

## 4. detect 与 infer_installed

优先级中 `infer_installed` 高于新机器检测，用于保留已有环境的明确选择。

`detect` 应：

- 快速；
- 只读；
- 不下载；
- 不安装系统组件；
- 返回可解释 reason/details。

## 5. machine_fingerprint

只放会使 route 选择失效的低成本事实。例如：

```text
OS / architecture
GPU 查询结果
driver/runtime 版本
关键系统库模式
```

不要放每次都会变化的时间、临时目录或无关设备信息，否则状态无法复用。

## 6. preflight

preflight 在 uv 修改环境之前执行。它负责回答：

```text
这个 option 在当前机器上是否有资格尝试？
```

例如 Triton provider 可检查 Linux、支持的 accelerator 和编译前提；但不应在这里安装 toolkit。

## 7. verify

verify必须在目标解释器中运行，验证“实际安装结果”，而不是再次查看配置。

互斥 GUI/headless 包族建议验证：

- 目标 distribution 存在；
- 另一互斥 distribution 不存在；
- import name 可用；
- 最小功能 probe 通过；
- distribution/version 与 option 匹配。

## 8. fallback 安全性

只有确实可以通过重新 routed sync 恢复的错误，才标记 `fallback_safe=True`。

例如：

```text
CUDA runtime 不可用 → CPU fallback 通常安全
配置损坏/lock 不一致 → 不应伪装成 fallback
未知 ABI crash → 默认 fail closed
```

## 9. 测试最低要求

每个新 provider 至少需要：

- 配置合同测试；
- 自动检测测试；
- 显式 option preflight；
- 未授权 fallback 失败；
- 授权 fallback 成功；
- 安装后验证成功/失败；
- provider-specific machine fingerprint 导致 state 失效；
- 代码中不存在项目唯一版本点的复制常量；
- 一个真实 uv metadata/lock 集成测试。
