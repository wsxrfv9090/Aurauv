# Aurauv 架构与不变量

## 1. 设计目标

Aurauv不是第二个包管理器，而是 `uv` 之前和之后的状态机：

```text
before uv: 发现所有权、选择 route、检查系统前提
uv:        解析、锁定、同步、运行
post uv:   验证 provider、持久化成功状态
```

只要某件事能由 `pyproject.toml + uv.lock` 表达，就继续交给 uv；只有机器选择、系统前提和运行时证据留在 provider。

## 2. 核心不变量

### 2.1 uv 是依赖真相来源

- 不手改 `uv.lock`；
- 不在项目 `.venv` 中做 lock 外的 `pip install` 修补；
- 不自行解析 Python requirement；
- route 只映射到已经由项目声明的 extras。

### 2.2 一个环境只有一个 owner

`environment_owner` 决定：

```text
pyproject 配置来源
根 uv.lock
默认 .venv
Aurauv state 目录
root extras
provider/member 合同
```

member 可以有 standalone lock，但不能在 workspace 内拥有第二份运行环境。

### 2.3 成功后才持久化

state 只在以下步骤全部成功后写入：

```text
uv command success
provider verification success
project import verification success
```

预检失败、uv 失败、runtime 失败或用户拒绝 fallback 都不会把失败选择记录成稳定状态。

### 2.4 fallback 必须授权

fallback来源只能是：

- `--aura-fallback ROUTE=OPTION`；
- `--aura-fallback cpu`（accelerator 简写）；
- 交互确认；
- `--aura-yes`。

配置中的 `fallbacks` 只说明“允许退到哪里”，不等于自动授权。

### 2.5 路由参数与 uv 参数分离

Aurauv只解析放在 uv 子命令之前的 `--aura-*`。一旦识别 uv 子命令，后续参数保留给
uv 或子程序；0.2.0 已知的 child/uv 参数表漂移单独记录在 README，本次功能新增
不改变这些转发语义。

## 3. 分层

```text
cli.py
  └── invocation.py       namespaced 参数与 uv 参数边界
      └── engine.py       命令状态机
          ├── topology.py workspace/submodule 所有权
          ├── config.py   [tool.aurauv] 合同
          ├── routes.py   option 选择与 fallback
          ├── runtime.py  uv/Python 授权
          ├── state.py    per-environment 持久状态
          ├── member_lock.py standalone member lock 协议
          ├── transaction.py add/remove metadata checkpoint
          └── providers/
              ├── base.py
              ├── exclusive_distribution.py
              ├── pytorch.py
              └── pytorch_companion.py
```

## 4. route 与 provider 的职责边界

Route描述业务选择：

```text
name
options
option -> root extras
default/detector
fallback graph
有序且无重复的 provider list
```

Provider描述某一包族如何实现和验证选择：

```text
provider project
包名与 import name
option -> provider extra
option -> index
自动检测
机器指纹
preflight
runtime verify
upgrade package family
current-interpreter 安装参数
```

核心不知道 `torch`、`cu132`、`nvidia-smi` 或 MPS；这些只存在于 PyTorch provider。
`pytorch-companion` 可在同 route 的基础 provider 后验证实际安装的 `torchaudio`、
`torchcodec`，缺席时 no-op，出现时把 distribution/module/backend 证据写入 state。
`exclusive-distribution` 由项目配置一个完整互斥 family、option → distribution 映射、
共享 module 和可选能力属性；同步后必须只发现所选 distribution，且 module probe
通过。它不改写依赖元数据，也不把冲突静默修成另一个 option。

## 5. 成员 lock 协议

当前 `metadata` strategy 适用于：

- member 本身不是 workspace root；
- 没有 workspace/path source；
- lock 所需动态文件可由 `metadata-files` 明确复制。

未来可新增：

```text
copy-tree      复制白名单目录
command        调用 member 自己声明的 lock hook
manifest       读取独立上下文 manifest
none           不发布 standalone lock
```

每种 strategy 都必须保证：

- 在 workspace 外解析；
- 不创建 member `.venv`；
- check-only 零写入；
- 更新时原子替换；
- 不触碰根 lock。

## 6. uv command 策略

| uv 命令 | Aurauv 行为 |
|---|---|
| `sync` | 注入 route extras；根 owner sync；验证；写状态 |
| `add/remove` | checkpoint 后以 `--no-sync` 修改 metadata，再 routed sync；后续失败则恢复 metadata/lock/state，并在原 root lock 存在时补偿同步环境；环境无法证明恢复时失效 state |
| `run` | routed inexact pre-sync，再 `run --no-sync` |
| `lock` | 原样根 lock，随后检查/维护 member standalone locks |
| `export/tree` | 注入 route extras并转发 |
| 其他命令 | 原样 passthrough |

`--aura-no-route` 对所有 uv 命令直接 passthrough。

`--locked` / `--frozen` 继续传给 uv，但 execution capabilities 禁止 Aurauv 自身
update/install/member refresh/provider repair/resync/state write。正常命令的原有能力不变。

## 7. 当前 interpreter 例外

Colab/kernel 不使用 `.venv` exact sync：

```text
uv export selected lock branch
uv pip install into sys.executable
editable install local packages without dependencies
provider verification
```

这是显式 target，不是普通项目环境的隐藏分支。
