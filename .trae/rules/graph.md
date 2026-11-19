---
alwaysApply: false
description: 当需要为笔记绘制图表（如流程图、架构图）或意图使用 mermaid 时触发，统一使用 archify 生成并规范图表文件存储。
---
1. 统一使用 **archify** 进行绘图，不要直接使用 mermaid；如收到 Mermaid 输入，先转换为 archify JSON。
2. 根据图表语义选择 `architecture`、`workflow`、`sequence`、`dataflow` 或 `lifecycle` 类型。
3. 使用 archify 的 `finalize` 流程生成并校验独立 HTML 图表，HTML 内含 SVG；不得将未通过校验的产物视为完成。
4. 将生成的 HTML 及其候选 JSON 保存在相关笔记同级目录下的 `assets` 文件夹中；如笔记需要静态图表，再从已校验的 HTML 导出 SVG，并一并保存到该目录。
