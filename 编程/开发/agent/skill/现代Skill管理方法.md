## 核心思路

现代 Skill 管理逐渐采用类似 npm 的包管理方式：Skill 以 Git 仓库中的目录或技能包形式发布，通过统一 CLI 下载、安装到不同 Agent 的标准目录，并支持查看、更新和移除。

典型入口是：

```bash
npx skills add <owner>/<repo>
```

以 `skills.sh` 生态为例，`skills` CLI 负责连接 Skill 来源与本地 Agent：

$$Skill\ Ecosystem = Registry + Package + CLI + Agent\ Adapter$$

- **Registry**：提供搜索、展示和发现能力，例如 `skills.sh`。
- **Package**：通常是包含 `SKILL.md` 的目录，也可以附带脚本、参考资料和模板。
- **CLI**：负责下载、筛选、安装、更新和移除。
- **Agent Adapter**：将 Skill 放到 Claude Code、Codex、Cursor、OpenCode 等 Agent 能识别的位置。

## 基本命令

### 安装 Skill

```bash
# 从 GitHub 仓库安装
npx skills add vercel-labs/agent-skills

# 安装指定 Skill
npx skills add vercel-labs/agent-skills --skill frontend-design

# 安装到指定 Agent
npx skills add vercel-labs/agent-skills --agent claude-code

# 安装到全局目录
npx skills add vercel-labs/agent-skills --global
```

默认情况下，Skill 通常安装到当前项目范围，使配置可以随项目共享。使用 `--global` 后，Skill 对当前用户的多个项目可用。

### 查看和筛选

```bash
# 查看仓库中可用的 Skill
npx skills add vercel-labs/agent-skills --list

# 查看本机已安装的 Skill
npx skills list
```

安装整个 Skill 集合前，先使用 `--list` 查看目录，再通过 `--skill` 选择需要的条目，可以减少无关规则对上下文和 Agent 行为的影响。

### 更新和移除

```bash
# 检查可更新内容
npx skills check

# 更新已安装 Skill
npx skills update

# 移除 Skill
npx skills remove <skill-name>
```

这使 Skill 不再是手动复制的 Markdown 文件，而成为可追踪、可更新、可回退的本地依赖。

### 非交互安装

在项目初始化或 CI 环境中，可以跳过确认提示：

```bash
npx skills add owner/repo \
  --skill skill-name \
  --agent codex \
  --global \
  --yes
```

使用非交互模式前，应固定来源、目标 Agent 和 Skill 名称，避免自动安装不必要的内容。

## Skill 来源

`npx skills add` 通常支持多种来源：

```bash
# GitHub 简写
npx skills add owner/repo

# GitHub 完整地址
npx skills add https://github.com/owner/repo

# 仓库中的指定目录
npx skills add https://github.com/owner/repo/tree/main/skills/example

# GitLab 或其他 Git 地址
npx skills add https://git.example.com/org/repo

# 本地 Skill 目录
npx skills add ./my-local-skills
```

私有仓库仍然使用 Git 的现有认证方式，例如 SSH、Git Credential Manager 或 GitHub CLI。认证凭据不应写入 Skill 文件或命令参数。

## Skill 包的结构

推荐的仓库结构如下：

```text
agent-skills/
└── skills/
    ├── frontend-design/
    │   ├── SKILL.md
    │   ├── references/
    │   ├── scripts/
    │   └── assets/
    └── code-review/
        └── SKILL.md
```

`SKILL.md` 是技能入口，通常包含 YAML Frontmatter：

```markdown
---
name: code-review
description: 审查代码变更，识别正确性、安全性和可维护性问题
---

## 执行流程

读取变更，分析风险，运行必要测试，并按严重程度输出问题。
```

其中 `name` 和 `description` 用于发现和匹配，正文用于指导 Agent 执行。脚本、参考资料和模板只在任务需要时加载，符合渐进式披露原则。

## 多 Agent 适配

现代 CLI 的重要价值是统一安装入口。相同的 Skill 内容可以被安装到不同 Agent 的目录：

```bash
npx skills add owner/repo \
  --skill code-review \
  --agent claude-code \
  --agent codex \
  --agent cursor
```

这解决了不同 Agent 目录不同、安装方式不同的问题。Skill 本身保持跨 Agent，CLI 负责处理目标目录、符号链接或文件复制等适配细节。

如果需要明确复制文件而不是建立链接，可使用：

```bash
npx skills add owner/repo --copy
```

## 项目级与全局级

| 类型 | 典型用途 | 特点 |
|---|---|---|
| 项目级 | 项目专用规范、框架约束、团队流程 | 随项目配置共享，影响范围较小 |
| 全局级 | 通用代码审查、通用调试和个人工作流 | 多个项目可复用，维护成本较低 |

推荐将强依赖项目上下文的 Skill 放在项目级，将通用能力放在全局级。项目级 Skill 与全局 Skill 同名时，应确认 Agent 的优先级规则，避免加载了错误版本。

## 现代管理流程

1. **发现**：在 `skills.sh` 或 Git 仓库中查找 Skill。
2. **审查**：阅读 `SKILL.md`、脚本和依赖，确认来源可信。
3. **试装**：先安装到单个 Agent 或测试项目。
4. **验证**：检查触发条件、工具调用、输出格式和副作用。
5. **推广**：确认稳定后安装到团队项目或全局环境。
6. **更新**：定期运行 `npx skills check` 和 `npx skills update`。
7. **清理**：移除不再使用、重复或存在风险的 Skill。

## 安全注意事项

Skill 不只是提示词包，可能包含脚本、命令和工具调用说明。安装前应重点检查：

- `SKILL.md` 是否要求读取或上传敏感信息；
- `scripts/` 是否执行删除、部署、网络下载等高风险命令；
- 是否存在隐藏的外部依赖或不必要的权限；
- 来源仓库是否可信，提交记录是否稳定；
- 更新是否可能改变原有行为；
- 安装范围是项目级还是全局级。

不要因为 Skill 出现在排行榜中就直接信任。排行榜适合发现，不等于安全审计。高风险 Skill 应在隔离环境中试运行，并保留版本和回滚方式。

## 遥测与隐私

部分 Skill CLI 默认收集匿名安装统计，用于技能排行和生态分析。对隐私敏感的环境，应先查看 CLI 文档，并按需要关闭遥测：

```bash
DISABLE_TELEMETRY=1 npx skills add owner/repo
```

企业环境还应确认依赖下载、仓库认证和日志记录是否符合内部安全要求。

## 手动安装的局限

无法使用 Node.js 或 CLI 时，可以手动复制包含 `SKILL.md` 的目录到 Agent 的技能目录。但手动复制通常存在以下问题：

- 无法被 `npx skills list` 正确追踪；
- 不参与 `npx skills check` 和 `npx skills update`；
- 容易出现多份副本和版本漂移；
- 不同 Agent 的目录和加载规则需要自行处理。

因此，手动复制适合作为临时回退方案，长期管理优先使用统一 CLI。

## 与传统方式的区别

| 方式 | 管理特点 |
|---|---|
| 直接复制 `SKILL.md` | 简单，但难以更新、追踪和跨 Agent 同步 |
| Git 仓库手动克隆 | 可版本控制，但安装路径和筛选过程需要人工处理 |
| `npx skills` CLI | 统一发现、安装、适配、更新和移除 |
| 企业内部 Skill Registry | 在 CLI 基础上增加权限、审计、审批和私有分发 |

因此，`npx skills` 代表的是一种“Skill 包管理器”思路：Skill 由仓库发布，CLI 负责分发，Agent 负责加载，项目或团队负责审核。

## 参考资料

- [skills.sh CLI 文档](https://skills.sh/docs/cli)
- [Vercel Skills CLI 仓库](https://github.com/vercel-labs/skills)
- [Agent Skills 规范](https://agentskills.io/)
