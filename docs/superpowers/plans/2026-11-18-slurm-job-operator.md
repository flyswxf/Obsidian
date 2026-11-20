---
tags:
  - agent-skill
  - implementation-plan
  - slurm
---

# Slurm Job Operator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 创建一个项目级技能，指导 Agent 安全地准备、提交、监控、诊断和验证 Slurm 作业。

**Architecture:** 使用单个 `SKILL.md` 保存触发描述和强制操作规程。技能复用项目现有集群安全规则，不附加容易与具体项目环境冲突的通用作业模板。

**Tech Stack:** Markdown、YAML frontmatter、Slurm、Environment Modules、Conda

---

### Task 1: 创建并验证项目级技能

**Files:**
- Create: `.trae/skills/slurm-job-operator/SKILL.md`
- Reference: `.trae/rules/cluster-safety.md`
- Reference: `docs/superpowers/specs/2026-11-18-slurm-job-operator-design.md`

- [x] **Step 1: 确认目标目录不存在**

运行：

```powershell
Test-Path "d:\Obsidian\.trae\skills\slurm-job-operator"
```

预期：输出 `False`。若输出 `True`，停止创建并报告冲突，不覆盖既有技能。

- [x] **Step 2: 创建技能目录**

运行：

```powershell
New-Item -ItemType Directory -Force "d:\Obsidian\.trae\skills\slurm-job-operator"
```

预期：目录创建成功。

- [x] **Step 3: 写入技能**

使用 `apply_patch` 创建 `.trae/skills/slurm-job-operator/SKILL.md`。frontmatter 必须为：

```yaml
---
name: slurm-job-operator
description: Safely prepare, submit, monitor, diagnose, and validate Slurm jobs. Use for sbatch, GPU or CPU cluster jobs, remote Conda workloads, and failed-job recovery. Do not use for local-only commands.
---
```

正文必须覆盖：

- 提交前只读检查和小样本优先。
- 独立作业名、日志和输出目录。
- `/etc/profile.d/modules.sh` 初始化。
- Conda 绝对路径与 `conda run`。
- `bash -n` 静态检查、`sbatch` 提交和 Job ID 记录。
- `squeue` 与 `sacct` 的状态判断。
- 活动作业禁止重复提交。
- 失败时先检查日志和产物，再只修复一个根因。
- 成功状态后仍需验证产物数量、可读性和关键内容。

- [x] **Step 4: 验证文件格式**

运行 PowerShell 检查，确认：

```text
目录名 = slurm-job-operator
name = slurm-job-operator
description 非空且不含冒号空格、尖括号或 TODO
正文非空
```

预期：所有检查通过。

- [x] **Step 5: 检查关键规则覆盖**

运行：

```powershell
rg -n "modules\.sh|conda run|bash -n|sbatch|Job ID|squeue|sacct|PENDING|RUNNING|COMPLETING|产物" "d:\Obsidian\.trae\skills\slurm-job-operator\SKILL.md"
```

预期：每项关键规则至少命中一次。

- [x] **Step 6: 报告结果**

报告技能绝对路径、最终 description 和验证结果。不创建 Git commit，除非用户另行明确要求。
