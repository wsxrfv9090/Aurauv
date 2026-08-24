# Changelog

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
