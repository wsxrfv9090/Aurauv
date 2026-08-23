# Aurauv 架构与不变量

## 1. 定位

Aurauv 是 uv 的 orchestration layer（编排层），不是第二个 resolver（解析器）或 installer（安装器）。

uv 继续拥有：

- `pyproject.toml` 依赖声明；
- `uv.lock` 解析结果；
- Python 获取与选择；
- `.venv` 创建和同步；
- workspace package 选择；
- `add/remove/run/export/tree` 的原生命令语义。

Aurauv只拥有：

- environment ownership（环境所有权）判断；
- route option（路由选项）选择；
- provider 安装前检查与安装后验证；
- 明确授权的 fallback；
- 路由状态持久化；
- workspace member 的 standalone-lock 协议；
- 在不破坏路由的前提下组合 uv 命令。

## 2. 核心不变量

### 2.1 唯一环境所有者

每次项目型调用必须先得到：

```text
invocation_root  = 用户正在操作的项目
owner_root       = 拥有共享 .venv、根 uv.lock 和 Aurauv state 的项目
effective_cwd    = uv/子程序实际看到的工作目录
```

三者不能混为一谈。

- standalone：`invocation_root == owner_root`；
- workspace member：`invocation_root != owner_root`；
- member 同时为 inner workspace root：外层显式 workspace 仍是 `owner_root`；
- 未受管 Git submodule：默认停止，不创建局部 `.venv`。

### 2.2 uv 是唯一依赖真相

Aurauv不得使用下列模式修复项目环境：

```text
uv sync
pip install 另一版 torch
```

项目环境必须能由：

```text
pyproject.toml + uv.lock + routed uv sync
```

复现。`uv pip install` 只允许用于 current-interpreter/Colab 这种明确不能 exact-sync 的宿主环境，并且依赖仍从同一个 lock 导出。

### 2.3 每个 route 恰好一个 option

route 表示一个互斥维度，例如：

```text
accelerator = cpu | mps | cuda
```

多个 route 可以正交组合：

```text
accelerator = cuda
triton      = enabled
vision      = headless
```

核心不得把多个维度预先展开成笛卡尔积 profile。

### 2.4 fallback 不等于配置默认值

`fallbacks = { cuda = "cpu" }` 只声明“可建议的安全目标”。实际 fallback 仍需满足至少一种授权：

- 用户显式传 `--aura-fallback`；
- 用户交互确认；
- 用户显式传 `--aura-yes`。

非交互环境不能仅因配置存在 fallback 就静默降级。

### 2.5 状态只记录成功事实

state 只有在以下步骤全部成功后写入：

1. uv 操作成功；
2. provider runtime verification 成功；
3. 配置的项目 imports 成功。

state 不是 lock，不参与依赖解析。机器签名、配置摘要或环境路径改变后，state 必须失效并重新检测。

## 3. 调用流水线

```text
CLI/launcher
  ↓
parse only --aura-* before uv command
  ↓
find uv (zero mutation on failure)
  ↓
discover invocation project and outer owner
  ↓
load owner [tool.aurauv]
  ↓
validate uv version and project Python
  ↓
build providers and validate pyproject contracts
  ↓
load valid persisted route state
  ↓
select route options
  ↓
provider preflight
  ↓
compose native uv command
  ↓
run uv
  ↓
provider runtime verification
  ↓
write state
```

## 4. 命令分层

### 4.1 原样转发

未由核心接管的 uv 子命令直接执行：

```text
python / tool / pip / build / publish / cache / self / ...
```

`--aura-no-route` 强制所有命令原样转发。

### 4.2 routed commands

- `sync`：注入 route extras；
- `export` / `tree`：注入 route extras，让输出对应当前机器；
- `run`：先 routed inexact sync，再 `uv run --no-sync`；
- `add/remove`：先 `--no-sync` 修改元数据，再 routed sync；
- `lock`：uv 维护 root lock，Aurauv再按合同维护 member standalone lock。

### 4.3 只读边界

`--dry-run`、`--check`、`--locked`、`--frozen` 等不能触发：

- state 写入；
- member lock 隐式刷新；
- Python 安装；
- uv self-update；
- provider repair/resync。

`--locked` / `--frozen` 不改变 uv 自身是否同步环境的定义；该边界限制的是 Aurauv
额外副作用。实现使用显式 execution capabilities，而不是把 lock 约束等同于
“uv 什么都不写”。

## 5. Provider 边界

Provider拥有包族知识：

```python
validate_contract()
detect()
infer_installed()
machine_fingerprint()
preflight(option)
verify(python, option)
protected_packages()
current_target_install_args(option)
```

Provider不应决定 workspace owner、解析通用 CLI 或直接写 state。

## 6. PyTorch 当前实现

PyTorch provider 校验：

- `cpu/mps/cuda` extras；
- extras 互斥；
- `tool.uv.sources` 按 extra 指向 explicit index；
- CPU/MPS 使用 `/whl/cpu`；
- CUDA 使用 `/whl/cuNNN` 或 `/whl/cuNNNN`；
- 根 workspace extras 正确转发 member extras；
- CUDA 预期 runtime 从唯一 index URL 推导；
- 安装后验证 distribution、import、CUDA/MPS runtime。

同 route 可在基础 `pytorch` provider 后挂接 `pytorch-companion`。companion 不检测
机器、不新增 extra、不替 uv 解释 lock marker；它在 sync/check 后按 managed
interpreter 中实际存在的 distribution 激活。`torchaudio`、`torchcodec` 等出现时，
其 distribution/module version、backend 与 import 结果进入 provider state；缺席时
明确记录 inactive。基础 provider 与 companion 的执行顺序来自 route.providers。

## 7. Member standalone lock

workspace 根 lock 与 member 独立 clone lock 是两个合同。

`metadata` strategy：

1. 显式复制 member 声明的 metadata files；
2. 在 workspace 外运行 uv lock；
3. 不创建 member `.venv`；
4. 只在内容变化时原子替换 member `uv.lock`；
5. 遇到 path/workspace sources 或 nested workspace 时拒绝猜测。

## 8. add/remove 事务边界

Aurauv在 `uv add/remove --no-sync` 前以内存 checkpoint 记录 owner、invocation root、
workspace members 的 `pyproject.toml`/`uv.lock` 以及目标 state。后续 member lock、
root sync、provider/project verification 或 state 写入失败时：

1. 原子恢复精确文件内容和 mode；
2. 删除本事务新建的 tracked metadata 文件；
3. 如果环境可能已变化且原 root lock 存在，按原 route extras执行补偿
   `uv sync --locked`；
4. 无原 root lock 或补偿同步失败时，只恢复可证明的元数据、失效无法再证明的
   state，并明确报告环境恢复边界。

事务不复制 `.venv`，不创建持久 backup，也不替代 uv lock。命令执行期间并发手工
修改同一 metadata 文件不属于支持的事务模型。

## 9. 扩展原则

新增 Triton 或 GUI/headless provider 时，优先新增 route/provider，不修改：

- topology；
- state；
- runtime authorization；
- generic uv invocation；
- member-lock core。

只有通用语义确实变化时才修改 engine。
