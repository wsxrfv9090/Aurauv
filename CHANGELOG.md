# Changelog

## Unreleased

- 修复短参数（含 `-p3.13`、`-qP PACKAGE`、`-nfPATH` 等组合）的预同步转发，保留 Python、索引和构建参数。
- 识别 preview、Python preference/fetch 与 trusted-host 等全局参数及别名，避免误把参数值识别为 uv 命令。
- uv 的 help/version 请求直接转发，不再创建环境、lockfile 或 Aurauv state；`run -` 和短 `-s` 组合正确区分子程序边界。
- 预同步保留参数顺序、反向开关和相对路径的原始工作目录，`--no-active` 正确覆盖 `--active`。
- 支持当前 uv 的 `run --no-editable-package`、`--upgrade-group`、`--prerelease-package` 和 `--system-certs`，预同步保留这些选项。
- `workspace` 和未来 uv 子命令的参数原样转发，不再误吞子命令后的 `--aura-*`。
- 修复 `run -m` / `-s` 选项边界，子程序参数不再影响路由、repair 或 fallback 同步。
- 在 uv 0.12.21 上验证；不提高 minimum-uv，也不自动升级用户的 uv。

## 0.2.0 — 2026-09-02

- 新增 `exclusive-distribution` provider，管理共享同一 import name 的互斥发行包族。
- 验证 route option 对应的实际 distribution、包族唯一性、共享 module import 与可选能力属性。
- 新增真实离线 uv route 切换、current-interpreter bootstrap、状态失效和 fail-closed 回归测试。

## 0.1.2 — 2026-08-24

- 修复 Windows BAT 启动器把 Python 版本比较中的 `^` 传入解释器而误报缺少 Python。
- 增加 BAT Python 版本检测表达式的发行物回归约束。

## 0.1.1 — 2026-08-23

- 命令与失败诊断中的 token、密码、URL userinfo 和敏感查询参数脱敏。
- 新增同 accelerator route 的可选 PyTorch companion provider。
- 实际安装的 `torchaudio`、`torchcodec` 可进入 runtime verification 与 state。
- `--locked` / `--frozen` 禁止 Aurauv 自身更新、安装、repair、resync 和 state 写入。
- provider 必须且只能按声明顺序挂接到所属 route。
- `add/remove` 多阶段失败时恢复项目元数据、lock、state，并在可行时补偿同步环境。

## 0.1.0 — 2026-08-21

- 首次可交付版本。
- 标准库-only uv wrapper 与 zipapp。
- standalone/workspace root/workspace member/Git submodule 拓扑。
- 外层 member 身份优先，包括多层嵌套 workspace 所有权。
- 通用 route/option/provider 模型。
- PyTorch CPU/MPS/CUDA provider。
- 明确授权的 preflight/runtime fallback。
- 每环境机器指纹与路由状态。
- member standalone metadata lock 协议。
- routed sync/add/remove/run/export/tree。
- current-interpreter/Colab bootstrap。
- 通用 standalone/workspace 模板与中文使用文档。
