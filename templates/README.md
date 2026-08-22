# Aurauv 项目模板

- `standalone/`：单独 clone 的 ML 项目；当前项目拥有 `.venv`、`uv.lock` 和 PyTorch provider。
- `workspace-root/`：包含 members/submodules 的根项目；根拥有唯一 `.venv` 和根 lock，provider 可以位于某个 member。
- `workspace-member/`：可独立 clone 的 member；独立时拥有自己的环境，位于外层 workspace 时服从外层 owner。

每个模板都包含：

```text
deployment/setup.py
deployment/setup.sh
deployment/setup.bat
deployment/aurauv.pyz   # 构建交付时生成
colab/bootstrap.py
pyproject-aurauv-snippet.toml
```

将 snippet 合并到项目现有 `pyproject.toml`，不要把它当成完整项目文件直接覆盖。
