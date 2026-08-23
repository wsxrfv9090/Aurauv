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
providers = ["pytorch", "pytorch-companions"]
fallbacks = { cuda = "cpu", mps = "cpu" }

[tool.aurauv.routes.accelerator.options.cpu]
extras = ["cpu"]
description = "Official CPU wheels"
```

| 键 | 含义 |
|---|---|
| `default` | 固定 option，或 `auto` |
| `detector` | `auto` 时负责检测的 provider 名称 |
| `providers` | 按声明顺序对该 route 执行 preflight/verify 的 provider 列表 |
| `fallbacks` | 允许建议的 source → target；不代表自动授权 |
| `options.<name>.extras` | 选中该 option 时注入 owner 项目的 extras |
| `options.<name>.description` | 可读说明 |

每个 `extras` 必须已经存在于 environment owner 的 `[project.optional-dependencies]`。
每个 provider 必须恰好列在其绑定 route 中一次；detector 必须先于依赖它的
companion provider。

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
| `type` | 基础 provider 类型；这里为 `pytorch` |
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

## 5. PyTorch companion provider

```toml
[tool.aurauv.providers.pytorch-companions]
type = "pytorch-companion"
project = "."
route = "accelerator"
base-provider = "pytorch"
packages = ["torchaudio", "torchcodec"]
imports = { torchaudio = "torchaudio", torchcodec = "torchcodec" }
```

| 键 | 含义 |
|---|---|
| `type` | 固定为 `pytorch-companion` |
| `project` | 必须是 environment owner（`.`） |
| `route` | 与基础 PyTorch provider 相同的 route |
| `base-provider` | 同 route 中先执行的 `pytorch` provider |
| `packages` | 需要在实际安装时纳入二进制兼容验证的 distribution |
| `imports` | distribution → import module，必须完整覆盖 `packages` |

companion provider 不新增依赖、不选择 route，也不自行解释 `uv.lock` 的全部 marker
分支。它在 routed sync 或 `uv sync --check` 后检查 managed interpreter：

- distribution 未安装：记录为 inactive，不报错；
- distribution 已安装：导入 module，记录 distribution/module version；
- module version 带 `+cuNNN`/`+cpu`/ROCm backend 标签：与所选 route 和
  `torch.version.cuda` 对比；
- import 或 backend 不匹配：在 state 写入前失败，并只允许对实际 active 包执行
  一次 focused reinstall；不会自行授权 fallback。

成功 state 的 `provider_results.pytorch-companions` 会明确包含 configured、active、
inactive packages、module backend 和 expected backend。

## 6. Member 合同

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

## 7. CLI override

```bash
aurauv --aura-route accelerator=cuda sync
aurauv --aura-device cuda sync
aurauv --aura-fallback accelerator=cpu sync
aurauv --aura-fallback cpu sync
aurauv --aura-refresh sync
aurauv --aura-target current aura bootstrap
```

`--aura-device` 与无 route 名的 `--aura-fallback` 是 `accelerator` 的便利别名。
