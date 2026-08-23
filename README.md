# Aurauv

> **Aura around uv：在不替代 uv 的前提下，为机器相关依赖、项目拓扑和共享环境增加一层可验证的路由。**

Aurauv 是面向 ML 项目的 `uv` 包装层。它保留 `uv` 负责的依赖解析、锁文件、环境同步和 Python 管理，只补上 `uv` 本身不会替项目决定的部分：

- 这台机器应该选择 CPU、Apple MPS 还是 NVIDIA CUDA；
- 某个路由选择应该映射到哪些 optional extras（可选依赖组）；
- 项目位于独立 clone、workspace 根、workspace member，还是 Git submodule；
- 哪个目录拥有唯一的 `.venv` 和根 `uv.lock`；
- 可独立 clone 的 workspace member 如何维护自己的 standalone `uv.lock`；
- 安装后的 PyTorch wheel、CUDA runtime 和 MPS runtime 是否真的可用；
- 路由选择如何针对“项目 + 实际环境 + 当前机器”持久化。

当前内置 provider（提供者）只有 **PyTorch**。核心已经按通用 `route → option → provider` 模型设计，未来可以新增 Triton、ROCm、XPU、GUI/headless 互斥包族等，而不需要重写 workspace、状态、确认和 uv 转发逻辑。

---

## 1. 使用方式

### 1.1 最舒服的日常入口

从 Aurauv 仓库安装为 uv tool（工具环境与目标项目环境彼此独立）：

```bash
uv tool install .
```

也可以直接从 GitHub 安装：

```bash
uv tool install git+https://github.com/wsxrfv9090/Aurauv.git
```

安装 Aurauv CLI 后，把原来的：

```bash
uv sync
uv add polars
uv remove polars
uv run pytest
uv lock --check
```

替换为：

```bash
aurauv sync
aurauv add polars
aurauv remove polars
aurauv run pytest
aurauv lock --check
```

除 `--aura-*` 参数外，其余参数会尽可能保持 uv 原义。非项目型或尚未由 Aurauv 接管的 uv 子命令会原样转发，例如：

```bash
aurauv --version
aurauv python list
aurauv tool list
aurauv pip list
aurauv build
```

明确绕过路由、完全按 uv 执行：

```bash
aurauv --aura-no-route sync
```

### 1.2 不安装 CLI，直接随项目携带

`templates/*/deployment/` 都包含一个标准库-only 的自包含文件：

```text
deployment/aurauv.pyz
```

项目中可直接运行：

```bash
python3 deployment/aurauv.pyz sync
```

为了更方便，还提供薄启动器：

```bash
bash deployment/setup.sh
python deployment/setup.py
```

默认无参数等价于：

```bash
aurauv sync
```

兼容简写：

```bash
bash deployment/setup.sh cpu
bash deployment/setup.sh mps
bash deployment/setup.sh cuda
bash deployment/setup.sh cuda --fallback cpu
```

等价于：

```bash
aurauv --aura-device cpu sync
aurauv --aura-device mps sync
aurauv --aura-device cuda sync
aurauv --aura-device cuda --aura-fallback cpu sync
```

`setup.py` 只要求启动它的 Python 为 **3.11+**。项目本身可以要求 Python 3.13+；Aurauv 启动后再让 uv 查找兼容 Python，并且只有获得授权时才调用 `uv python install`。

把最终 runtime 写入任意目标项目：

```bash
python tools/vendor.py /path/to/project --with-colab-bootstrap
```

这会生成 `deployment/aurauv.pyz` 和三个薄启动器，但不会自动改写目标 `pyproject.toml`。配置应从 `templates/` 合并，避免覆盖项目已有依赖。`bootstrap/` 还提供一套不安装 wheel 也能直接运行的通用 CLI bundle。

---

## 2. 首次 setup 的执行顺序

对受管项目执行 `aurauv sync` 时，顺序固定为：

```text
检查 uv 是否存在
  ↓
检查 uv 最低版本
  ↓
发现当前项目、workspace 和 Git submodule 拓扑
  ↓
确定唯一 environment owner（环境所有者）
  ↓
读取 owner 的 [tool.aurauv]
  ↓
查找满足项目合同的 Python
  ↓
必要时询问是否允许 uv 安装 Python
  ↓
读取当前环境的持久路由状态
  ↓
状态无效时进行机器识别
  ↓
provider 安装前检查
  ↓
为 uv 注入选定 route 对应的 extras
  ↓
uv lock/sync
  ↓
provider 运行时验证
  ↓
全部成功后写入状态
```

### uv 不存在

Aurauv 立即报错并终止，不会：

- 创建 `.venv`；
- 修改 `pyproject.toml`；
- 创建或修改 `uv.lock`；
- 安装 Python；
- 修改当前 Colab/kernel 环境。

Aurauv 不会偷偷执行 `pip install uv`。应先使用 Astral 官方方式安装 uv。

### uv 版本过旧

默认策略是 `ask`：交互终端中询问是否运行：

```bash
uv self update
```

非交互环境不会默认同意。也可显式指定：

```bash
aurauv --aura-update-uv sync
aurauv --aura-no-update-uv sync
```

若 uv 由系统包管理器管理，`uv self update` 可能拒绝更新；Aurauv 会保留原错误并要求通过原安装渠道升级。

### Python 不满足项目要求

Aurauv先以 `--no-python-downloads` 查找已有兼容解释器。找不到时，默认询问是否允许：

```bash
uv python install <python-request>
```

显式控制：

```bash
aurauv --aura-install-python sync
aurauv --aura-no-install-python sync
```

---

## 3. 四种项目拓扑

Aurauv 的第一原则是：**先确定环境所有权，再做包路由。**

| 当前目录状态 | environment owner | `.venv` | lock 行为 |
|---|---|---|---|
| 独立 clone | 当前项目 | 当前项目 `.venv` | 当前项目 `uv.lock` |
| workspace root | 根项目 | 根 `.venv` | 根 `uv.lock` |
| workspace member / 受管 submodule | 外层 workspace root | 只使用根 `.venv` | 根 lock + 可选 member standalone lock |
| 当前项目既是外层 member 又是内层 root | **外层 workspace root** | 只使用外层根 `.venv` | 按 member 处理 |

### 3.1 独立 clone

例如单独 clone 一个 ML 项目：

```text
sample-ml/
├── pyproject.toml
├── uv.lock
└── deployment/
```

Aurauv把该项目视为环境所有者，管理：

```text
sample-ml/.venv
sample-ml/uv.lock
sample-ml/.aurauv/state/*.json
```

### 3.2 workspace root

例如一个包含 ML member 的 workspace：

```text
workspace-app/
├── pyproject.toml
├── uv.lock
├── .venv/
└── modules/model-core/
```

Aurauv明确提示发现 workspace members / Git submodules，并只使用：

```text
workspace-app/.venv
workspace-app/uv.lock
```

### 3.3 workspace member / Git submodule

从这里运行也可以：

```bash
cd workspace-app/modules/model-core
aurauv sync
```

Aurauv会把 invocation root（调用项目）识别为 member，把 environment owner 识别为外层 workspace root：

```text
不创建 modules/model-core/.venv
同步 workspace-app/.venv
使用 workspace-app/uv.lock
按声明维护 member 独立 clone 的 uv.lock
命令工作目录仍保持在 modules/model-core
```

### 3.4 member 同时也是内层 workspace root

如果一个项目既包含自己的 members，又被外层 workspace 收纳，**外层 member 身份优先**。这避免进入内层目录后静默创建第二份 `.venv`。

### 3.5 未受管 Git submodule

如果目录确实是 Git submodule，但没有被外层 uv workspace 和 `[tool.aurauv.members]` 正确接管，Aurauv默认 fail-closed（封闭失败）：

```text
不猜测环境所有者
不创建 member-local .venv
不把两个项目静默混成一个环境
```

把该 Python submodule 加入根 `tool.uv.workspace.members` 和 `tool.aurauv.members` 后再运行。

---

## 4. 路由模型：route / option / provider

Aurauv 不把未来所有问题都硬编码成 CPU/MPS/CUDA。

```text
route（互斥维度）
  ├── option A
  ├── option B
  └── option C

provider（包族适配器）
  ├── 校验 pyproject 合同
  ├── 自动检测
  ├── 安装前检查
  ├── 安装后验证
  └── 返回状态指纹
```

本版只有：

```text
route: accelerator
options: cpu / mps / cuda
provider: pytorch
```

未来可以独立加入：

```text
route: triton
options: enabled / disabled

route: vision-runtime
options: gui / headless
```

不同 route 正交；同一 route 内 option 互斥。这样不会把 accelerator、Triton、GUI/headless 全部揉成一个不断膨胀的巨型 profile。

你提到的 ComfyUI 场景很可能属于“同一 import name、不同且互斥 distribution”的 GUI/headless 包族；Aurauv当前没有猜测或实现它，但 provider 边界已经能承载：包冲突校验、平台检测、选项映射和 import/runtime 验证。

---

## 5. PyTorch provider

### 5.1 项目依赖仍由 pyproject + uv.lock 完整管理

Aurauv不使用“先 uv sync，再额外 pip install CUDA Torch”的修补方式。PyTorch仍通过 extras、`tool.uv.sources` 和 explicit indexes 进入 uv lock。

典型 provider 项目：

```toml
[project.optional-dependencies]
cpu = ["torch", "torchvision"]
mps = ["torch", "torchvision"]
cuda = ["torch", "torchvision"]

[tool.uv]
conflicts = [[
  { extra = "cpu" },
  { extra = "mps" },
  { extra = "cuda" },
]]

[tool.uv.sources]
torch = [
  { index = "pytorch-cpu", extra = "cpu" },
  { index = "pytorch-cpu", extra = "mps" },
  { index = "pytorch-cuda", extra = "cuda" },
]
torchvision = [
  { index = "pytorch-cpu", extra = "cpu" },
  { index = "pytorch-cpu", extra = "mps" },
  { index = "pytorch-cuda", extra = "cuda" },
]

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[[tool.uv.index]]
name = "pytorch-cuda"
url = "https://download.pytorch.org/whl/cu132"
explicit = true
```

### 5.2 CUDA backend 只有一个手工更新点

Aurauv代码中没有 `cu132` 常量。它从 provider 项目的 index URL 推导：

```text
.../whl/cu132 → wheel backend cu132 → torch.version.cuda 应为 13.2
```

未来更新 CUDA wheel backend，只改 provider 项目 `pyproject.toml` 中的 URL，然后：

```bash
aurauv --aura-device cuda aura upgrade pytorch
```

### 5.3 自动检测

默认优先级：

```text
显式 --aura-route / --aura-device
  ↓
显式 uv --extra
  ↓
当前环境的有效持久状态
  ↓
当前环境中已有可识别 provider
  ↓
route 固定 default
  ↓
provider 自动检测
```

PyTorch 自动检测：

- Apple Silicon macOS → `mps`；
- 非 macOS 且 `nvidia-smi` 能真实查询 GPU → `cuda`；
- 其他情况 → `cpu`。

仅仅“PATH 中存在 `nvidia-smi` 文件”不算 CUDA 可用。

### 5.4 安装后验证

- `torch` / `torchvision` distribution 存在且可 import；
- CPU/MPS 不得残留 CUDA wheel；
- MPS wheel 必须由 MPS build，默认还要求 `mps.is_available()`；
- CUDA wheel 的 `torch.version.cuda` 必须匹配 index URL；
- CUDA 必须满足 `torch.cuda.is_available()`；
- 项目配置的额外 imports 必须成功。

如果 uv 元数据显示同步，但 PyTorch 文件损坏，Aurauv允许一次受控的 provider package 定向重装；不会绕开 uv lock。

---

## 6. fallback 绝不静默

例如 CPU-only 机器上显式请求：

```bash
aurauv --aura-device cuda sync
```

默认失败，不会偷偷降为 CPU。

允许交互确认：

```bash
aurauv --aura-device cuda sync
# preflight/runtime 失败后询问是否 fallback
```

明确授权：

```bash
aurauv --aura-device cuda --aura-fallback cpu sync
```

非交互自动化：

```bash
aurauv \
  --aura-device cuda \
  --aura-fallback cpu \
  --aura-no-input \
  sync
```

`--aura-yes` 会同意配置中可用的 fallback 与安装/更新询问，应只在你确实接受这些动作时使用。

---

## 7. 状态持久化

默认状态路径：

```text
<environment-owner>/.aurauv/state/<environment-id>.json
```

它不是依赖锁，也不代替 `uv.lock`。它只保存某个实际环境上一次成功验证的 route 选择和证据。

状态按以下信息失效：

- environment owner；
- 实际环境路径；
- `[tool.aurauv]` 配置摘要；
- OS / architecture；
- provider 机器指纹（例如 NVIDIA 查询结果）；
- route option 是否仍存在。

强制重新检测：

```bash
aurauv --aura-refresh sync
```

建议加入 `.gitignore`：

```gitignore
.aurauv/
```

---

## 8. add / remove / sync / run 的语义

### `sync`

Aurauv只增加当前 route 所需的 `--extra` 和正确的 root `--project`，然后让 uv 正常同步。

### `add` / `remove`

裸 `uv add/remove` 会自动同步，但不会替项目永久记住某个 optional extra。Aurauv采用：

```text
uv add/remove --no-sync
  ↓
按原命令更新 pyproject / lock
  ↓
维护需要独立 clone 的 member lock
  ↓
执行包含持久 route 的 uv sync
  ↓
验证并写状态
```

Aurauv在第一步之前以内存 checkpoint 记录 root/member 的 `pyproject.toml`、
`uv.lock` 和目标 state。后续阶段失败时原子恢复这些文件；如果 routed sync
可能已经修改环境且原 root lock 存在，再按原 lock 补偿同步。该过程不创建
持久 `.bak/.old` 文件。若事务开始时没有 root lock，Aurauv会恢复元数据并明确
报告环境无法安全重建，同时失效无法再证明对应当前环境的 state，而不会猜测
旧解析结果。补偿同步失败时也遵守同一 state 失效规则。

用户显式给出 `--no-sync` 时，Aurauv尊重该语义，不追加同步。

### `run`

Aurauv先确保 routed environment 可用，然后以：

```text
uv run --no-sync ...
```

执行子命令，避免 uv 在第二阶段再次用未路由的默认条件同步。

Aurauv以首个子程序 positional argument 作为参数边界。0.1.1 的已知 uv 参数
漂移见下文；在这些边界内不要假定所有未来 uv 选项都已被 wrapper 理解。

### `--dry-run` / `--check` / `--locked` / `--frozen` / `--offline`

这些约束会继续原样传给 uv。`--locked` / `--frozen` 仍允许 uv 按自身语义同步
环境，但不会触发 Aurauv 自身的 uv update、Python install、member lock refresh、
provider repair/resync 或 state 写入。`--check` / `--dry-run` 同样遵守该副作用边界。

### `--all-extras`

受管 route 使用 mutually exclusive extras（互斥 extras）时，Aurauv拒绝 `--all-extras`，因为它会破坏“每个 route 恰好一个 option”的合同。

### 0.1.1 已知 uv 参数漂移（本版本仅声明）

Aurauv为 `add/remove/run` 重建 routed sync 时维护显式参数表。当前已知以下 uv
0.12.5 语义尚未完整建模，本版本不修改它们：

- `uv run` 子程序参数若恰好包含 `--isolated`、`--no-project`、`--script` 或
  `--gui-script`，可能被提前识别为 uv run 选项；
- `uv add --optional NAME PACKAGE` 会把依赖写入 optional extra，但 follow-up
  routed sync 尚不会自动转换为 `sync --extra NAME`；
- follow-up sync 尚未复制 `--no-editable-package`、`--no-install-project`、
  `--no-install-workspace`、`--no-install-local`、`--no-install-package`、
  `--upgrade-group`、`--prerelease-package`、`--system-certs` 和项目型 `--script`。

需要这些参数的精确原生语义时，应使用 `--aura-no-route` 明确直通 uv，并由调用者
另行执行 reviewed routed sync；不要把直通视为已经保留了 Aurauv route。

---

## 9. workspace member 的 standalone lock

uv workspace 的运行环境使用根 `uv.lock`；但可独立 clone 的 member 仍可以提交自己的 standalone `uv.lock`。

Aurauv当前提供显式策略：

```toml
[tool.aurauv.members.model_core]
path = "modules/model-core"
distribution = "model-core"
standalone-lock = "metadata"
metadata-files = ["pyproject.toml", "uv.lock", "README.md"]
```

`metadata` 策略会把声明的元数据复制到 workspace 外临时目录，再让 uv 检查或生成 standalone lock，最后原子替换 member 的 `uv.lock`。它不会创建 member `.venv`。

该策略会拒绝：

- member 自己还是嵌套 workspace root；
- member 使用无法在临时上下文复现的 workspace/path source；
- metadata-files 不完整。

这不是全盘扫描，而是显式协议。未来复杂 member 可以新增另一种 standalone-lock strategy，而不是让核心猜测要复制哪些文件。

管理命令：

```bash
aurauv aura lock-members --check
aurauv aura lock-members
aurauv aura doctor
```

---

## 10. Colab / 当前解释器

运行中的 notebook kernel 不能“切换进入”项目 `.venv`，也不应被 exact sync 清掉预装包。因此 Aurauv提供 current target（当前解释器目标）：

```bash
aurauv --aura-target current aura bootstrap
```

它会：

1. 验证当前解释器满足项目 Python 合同；
2. 从同一个 `uv.lock` 导出选中 routes 的依赖；
3. 使用 `uv pip install --python <current>` 非精确安装；
4. 以 editable/no-deps 安装根项目与可安装 members；
5. 执行 provider 和项目 import 验证；
6. 不删除 notebook host 的其他包。

示例 `colab/bootstrap.py` 已包含在模板中。该入口同样要求系统已有 uv；缺少 uv 时零修改退出。

---

## 11. pyproject 配置概览

最小 standalone 结构：

```toml
[tool.aurauv]
schema-version = 1
project = "My ML Project"
minimum-uv = "0.10.0"
python-request = "3.13"
python-install-policy = "ask"
uv-update-policy = "ask"
state-directory = ".aurauv"
allow-unmanaged-submodule = false

[tool.aurauv.verify]
imports = ["my_package"]

[tool.aurauv.routes.accelerator]
default = "auto"
detector = "pytorch"
providers = ["pytorch", "pytorch-companions"]
fallbacks = { cuda = "cpu", mps = "cpu" }

[tool.aurauv.routes.accelerator.options.cpu]
extras = ["cpu"]

[tool.aurauv.routes.accelerator.options.mps]
extras = ["mps"]

[tool.aurauv.routes.accelerator.options.cuda]
extras = ["cuda"]

[tool.aurauv.providers.pytorch]
type = "pytorch"
project = "."
route = "accelerator"
packages = ["torch", "torchvision"]
extras = { cpu = "cpu", mps = "mps", cuda = "cuda" }
indexes = { cpu = "pytorch-cpu", mps = "pytorch-cpu", cuda = "pytorch-cuda" }
mps-requires-available = true

[tool.aurauv.providers.pytorch-companions]
type = "pytorch-companion"
project = "."
route = "accelerator"
base-provider = "pytorch"
packages = ["torchaudio", "torchcodec"]
imports = { torchaudio = "torchaudio", torchcodec = "torchcodec" }
```

完整 standalone、workspace root 和 workspace member 模板见：

```text
templates/
```

可复制的最小配置片段见：

```text
examples/
```

---

## 12. 未来新增 provider 的边界

新增 Triton 或 GUI/headless 包族时，不应改动 `engine.py` 的 workspace 和状态机。应新增 provider，实现：

```python
class NewProvider(Provider):
    def validate_contract(self): ...
    def detect(self): ...
    def machine_fingerprint(self): ...
    def preflight(self, option): ...
    def verify(self, python, *, option, cwd): ...
    def protected_packages(self): ...
```

然后在 `providers/registry.py` 注册类型，并在 `pyproject.toml` 中新增 route/provider 配置。

例如未来 Triton 可以有：

```text
route = triton
options = enabled / disabled
detect = Linux + supported accelerator
preflight = OS、Python、Torch backend、编译前提
verify = import triton + 最小 runtime probe
```

GUI/headless provider 可以验证：

```text
两个 distribution 不得同时出现
二者 import name 可能相同
选择必须由平台/用途或显式配置决定
安装后验证实际 distribution，而不只验证 import 成功
```

---

## 13. 管理命令

```bash
aurauv aura status
aurauv --aura-json aura status
aurauv aura routes
aurauv aura doctor
aurauv aura lock-members --check
aurauv aura lock-members
aurauv --aura-device cuda aura upgrade pytorch
aurauv --aura-target current aura bootstrap
aurauv aura version
```

查看完整帮助：

```bash
aurauv aura help
```

---

## 14. 明确边界

Aurauv当前不会：

- 安装 NVIDIA 驱动、CUDA Toolkit、编译器或系统动态库；
- 把 `nvcc` 当作普通 PyTorch wheel 的必要条件；
- 静默 fallback；
- 静默创建 submodule-local `.venv`；
- 通过额外 `pip install` 绕开 uv lock 修补项目环境；
- 自动猜测未来 Triton、OpenCV GUI/headless、ROCm 或 XPU 合同；
- 保证尚未在真实硬件执行过的 CUDA/MPS runtime 可用。

本仓库的测试覆盖参数边界、拓扑、member lock、授权模型、状态、offline uv 集成和 zipapp。真实 RTX 4070、Apple MPS、Windows launcher 和 Colab kernel 仍需在对应设备做首次验收。

---

## 15. 仓库结构

```text
Aurauv/
├── src/aurauv/                 # 标准库-only 核心
│   ├── engine.py               # uv 编排和命令语义
│   ├── topology.py             # workspace/submodule 所有权
│   ├── routes.py               # route 选择与 fallback
│   ├── member_lock.py          # standalone member lock 协议
│   └── providers/pytorch.py    # 当前唯一 provider
├── templates/                  # standalone/root/member 与 Colab 部署模板
├── docs/                       # 配置和行为参考
├── examples/                   # 通用 standalone/workspace 配置片段
├── bootstrap/                  # 免安装 wheel 的通用 CLI bundle
├── scripts/build_zipapp.py     # 生成自包含 aurauv.pyz
└── tests/                      # 单元与真实 uv 集成测试
```

## License

MIT License。
