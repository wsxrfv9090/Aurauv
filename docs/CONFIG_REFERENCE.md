# `[tool.aurauv]` 配置参考

当前 schema：

```toml
[tool.aurauv]
schema-version = 1
```

## 1. 根配置

```toml
[tool.aurauv]
schema-version = 1
project = "My Project"
minimum-uv = "0.10.0"
python-request = "3.13"
python-install-policy = "ask"
uv-update-policy = "ask"
state-directory = ".aurauv"
allow-unmanaged-submodule = false
```

| 键 | 默认值 | 含义 |
|---|---|---|
| `schema-version` | 必填 | 当前只能是 `1` |
| `project` | `[project].name` | 日志中的可读名称 |
| `minimum-uv` | `0.10.0` | Aurauv允许执行项目命令的最低 uv 版本 |
| `python-request` | 自动从项目合同推导 | 需要安装 Python 时传给 `uv python install` 的请求 |
| `python-install-policy` | `ask` | `ask / always / never` |
| `uv-update-policy` | `ask` | `ask / always / never` |
| `state-directory` | `.aurauv` | owner 内的相对状态目录 |
| `allow-unmanaged-submodule` | `false` | 是否允许绕过 submodule ownership 防护；通常不要开启 |

`state-directory` 不得是绝对路径，也不得包含 `..`。

## 2. 项目 import 验证

```toml
[tool.aurauv.verify]
imports = ["model_core"]
```

同步后使用环境 Python 导入这些模块。只写当前快照中确定存在的 package；不要因为项目名称看起来像 package 就盲目加入。

## 3. Route

```toml
[tool.aurauv.routes.accelerator]
default = "auto"
detector = "pytorch"
providers = ["pytorch"]
fallbacks = { cuda = "cpu", mps = "cpu" }

[tool.aurauv.routes.accelerator.options.cpu]
extras = ["cpu"]
description = "Official CPU wheels"
```

| 键 | 含义 |
|---|---|
| `default` | 固定 option，或 `auto` |
| `detector` | `auto` 时负责检测的 provider 名称 |
| `providers` | 对该 route 执行 preflight/verify 的 provider 列表 |
| `fallbacks` | 允许建议的 source → target；不代表自动授权 |
| `options.<name>.extras` | 选中该 option 时注入 owner 项目的 extras |
| `options.<name>.description` | 可读说明 |

每个 `extras` 必须已经存在于 environment owner 的 `[project.optional-dependencies]`。

## 4. PyTorch provider

```toml
[tool.aurauv.providers.pytorch]
type = "pytorch"
project = "."
route = "accelerator"
packages = ["torch", "torchvision"]
extras = { cpu = "cpu", mps = "mps", cuda = "cuda" }
indexes = { cpu = "pytorch-cpu", mps = "pytorch-cpu", cuda = "pytorch-cuda" }
imports = { torch = "torch", torchvision = "torchvision" }
mps-requires-available = true
```

| 键 | 含义 |
|---|---|
| `type` | 内置 provider 类型；当前为 `pytorch` |
| `project` | 真正声明 PyTorch extras/index 的项目，相对 owner |
| `route` | provider 绑定的 route |
| `packages` | 同步、upgrade、repair 的 distribution family |
| `extras` | provider option → provider project extra |
| `indexes` | provider option → `tool.uv.index` 名称 |
| `imports` | distribution → import module；默认用 distribution 名转换 |
| `mps-requires-available` | `true` 时要求 MPS built 且当前 runtime available |

workspace root 的 route extras必须把 member extras转发到根，例如：

```toml
[project.optional-dependencies]
cpu = ["model-core[cpu]"]
mps = ["model-core[mps]"]
cuda = ["model-core[cuda]"]
```

## 5. Member 合同

```toml
[tool.aurauv.members.model_core]
path = "modules/model-core"
distribution = "model-core"
standalone-lock = "metadata"
metadata-files = ["pyproject.toml", "uv.lock", "README.md"]
```

| 键 | 含义 |
|---|---|
| `path` | 相对 environment owner 的 member 路径 |
| `distribution` | member 的 `[project].name`，规范化后必须一致 |
| `standalone-lock` | `metadata` 或 `none` |
| `metadata-files` | 在 workspace 外复现 standalone lock 所需的显式文件 |

`metadata` 不是任意项目复制器。遇到 path/workspace source 或 nested workspace 会停止。

## 6. CLI override

```bash
aurauv --aura-route accelerator=cuda sync
aurauv --aura-device cuda sync
aurauv --aura-fallback accelerator=cpu sync
aurauv --aura-fallback cpu sync
aurauv --aura-refresh sync
aurauv --aura-target current aura bootstrap
```

`--aura-device` 与无 route 名的 `--aura-fallback` 是 `accelerator` 的便利别名。
