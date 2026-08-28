# 更新日志 / Changelog

## v1.10.1

- 前端工作台：「镜头设置」与「生成视频与版本」面板改为可折叠（`details/summary`），默认展开；切换镜头或收起/展开时保持状态并自适应布局（`web/ref2va-unified-director-studio-v190.js`）。
- 版本号统一为 1.10.1：后端 `REF2VA_DIRECTOR_BACKEND_VERSION`、前端 `FRONTEND_VERSION`。
- 正式工作流更名为 `梦镜 DreamShot 工作流_v1.10.1.json`（内容与 v1.10.0 完全一致，仅文件名变化）。
- GitHub 仓库结构标准化：
  - 插件目录由 `ComfyUI-Ref2VA-Gates` 整理为仓库根目录（节点类名与节点 ID 均未改动，旧工作流兼容）。
  - 新增 README.md / README_EN.md / requirements.txt / pyproject.toml / .gitignore / LICENSE（占位）/ CHANGELOG.md。
  - 工作流整理至 `workflows/DreamShot_v1.10.1.json`；教程整理至 `docs/使用教程.md`。
  - 清理开发产物：`__pycache__/`、`*.pyc`、`*.bak` 历史备份（共 12 个）。

## v1.10.0

- 以 ZIP 安装包形式发布的版本，包含：Ref2VA-Gates 扩展、DreamShot 工作台脚本、可导入工作流、中文安装与使用教程。
- （更早版本的逐条更新记录未随原包保留，自 v1.10.1 起在本文件中维护。）
