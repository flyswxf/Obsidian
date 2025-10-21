# UML 分支说明

这个分支专用于共享我的 Obsidian 笔记中的 `编程/UML` 内容，已明确排除以下子目录：
- `编程/UML/领域模型`
- `编程/UML/原型图`

分支中文件布局：
- 根目录下仅包含 `编程/UML` 以及本说明文件 `README.md`

## 如何获取本分支
如果你只想获取 UML 笔记内容，建议仅克隆该分支：

- 标准克隆该分支：
  ```bash
  git clone --branch UML --single-branch <仓库URL>
  ```
- 只取最新快照（更快）：
  ```bash
  git clone --branch UML --single-branch --depth 1 <仓库URL>
  ```

克隆完成后，Obsidian 可直接将克隆下来的文件夹作为新的库（vault）打开，或把 `编程/UML` 目录添加到你现有库中。

## 更新策略
- 此分支仅维护 `编程/UML` 相关内容，其他笔记不在此分支。
- 当主仓库更新 UML 笔记时，会在该分支中同步。

## 备注
- 若需要完整仓库或其他笔记，请切换至 `main` 分支或对应功能分支。
