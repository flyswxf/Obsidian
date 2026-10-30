## 目录

- [[#1. 岗位概述]]
- [[#2. 知识体系全景图（知识树）]]
- [[#3. 分领域详细知识点梳理]]
  - [[#3.1 大模型评测基础与Benchmark]]
  - [[#3.2 Prompt工程与Rubrics设计]]
  - [[#3.3 评测归因分析与Agent评测]]
  - [[#3.4 评测数据构建与裁判模型应用]]
  - [[#3.5 评测体系设计与系统架构]]
- [[#4. 分领域学习资料规划]]
  - [[#4.1 大模型评测基础与Benchmark — 学习资料]]
  - [[#4.2 Prompt工程与Rubrics设计 — 学习资料]]
  - [[#4.3 评测归因分析与Agent评测 — 学习资料]]
  - [[#4.4 评测数据构建与裁判模型应用 — 学习资料]]
  - [[#4.5 评测体系设计与系统架构 — 学习资料]]
  - [[#4.6 编程与工程能力 — 学习资料]]
  - [[#4.7 大模型基础理论 — 学习资料]]
- [[#5. 学习路径建议]]
- [[#附录：全部学习资料速查表]]

---

## 1. 岗位概述

### 1.1 岗位定位

AI模型评测岗位属于AI模型方向的核心研发岗位，专注于大语言模型（LLM）的能力评估、质量保障和能力提升。该岗位需要候选人具备从评测方法论设计到评测系统工程的全栈能力，是连接模型研发与产品落地的关键桥梁。

### 1.2 核心能力要求

| 能力维度 | 核心要求 |
|----------|----------|
| 评测方法论 | 掌握大模型评测的完整理论体系，能独立设计评测方案 |
| 数据工程 | 能构建高质量评测数据集，管理评测数据全生命周期 |
| 裁判模型 | 精通LLM-as-Judge技术，能训练和优化裁判模型 |
| Agent评测 | 理解Agent系统，能设计复杂的Agent评测方案 |
| 归因分析 | 能从评测结果进行深度归因分析，建立改进闭环 |
| 系统工程 | 能搭建完整的评测系统（数据管理、模型接入、自动评测、结果可视化） |
| Prompt工程 | 精通Prompt设计方法和Rubrics体系化设计 |

### 1.3 岗位掌握程度概览

该岗位岗位能力要求分为0-专家级，其中：
- **入门级**：缺乏基本认知（不合格）
- **基础级**：熟悉常用Benchmark和评测框架，了解Agent概念（入门级）
- **进阶级**：能说明Benchmark设计理念，理解评测数据构建方法（初级）
- **精通级**：能独立设计评测方案，构建完整评测系统（中级 — 目标）
- **专家级**：主导评测体系架构设计，推动方法论创新（高级/专家级）

**目标定位**：学完本计划后，应达到 **精通级水平**，具备独立完成评测体系设计、构建完整评测系统、进行深度归因分析的能力。

---

## 2. 知识体系全景图（知识树）

```
AI模型评测岗位知识体系
│
├── A. 大模型评测基础与Benchmark
│   ├── A1. 评测基本概念与理论
│   │   ├── 评测的定义、目的与意义
│   │   ├── 评测范式分类（客观评测 vs 主观评测）
│   │   ├── 评测的挑战（数据污染、泛化性、可复现性）
│   │   └── 评测与模型训练的关系
│   ├── A2. 业界主流Benchmark
│   │   ├── 多学科知识评测（MMLU、MMLU-Pro、CMMLU、C-Eval）
│   │   ├── 代码生成评测（HumanEval、HumanEval+、MBPP、LiveCodeBench）
│   │   ├── 数学推理评测（GSM8K、MATH、GPQA、AIME）
│   │   ├── 多轮对话评测（MT-Bench、AlpacaEval）
│   │   ├── 指令遵循评测（IFEval）
│   │   ├── 常识与推理评测（HellaSwag、BBH、ARC）
│   │   ├── 安全与对齐评测（TruthfulQA、ToxiGen）
│   │   └── 综合能力评测（BigBench、AGIEval）
│   ├── A3. 评测框架与工具
│   │   ├── lm-eval-harness（EleutherAI）
│   │   ├── OpenCompass（上海AI Lab）
│   │   ├── HELM（Stanford）
│   │   ├── OpenAI Evals
│   │   ├── FastChat评测工具
│   │   ├── Promptfoo
│   │   └── DeepEval / RAGAS / TruLens
│   ├── A4. 评测数据构建方法
│   │   ├── 人工标注流程设计
│   │   ├── 数据质量控制（标注一致性检查）
│   │   ├── 数据去重与防泄露策略
│   │   ├── 难度梯度设计
│   │   └── 对抗样本构造
│   ├── A5. 评测场景与能力维度体系
│   │   ├── 能力维度划分方法论
│   │   ├── 评测场景设计原则
│   │   ├── 能力图谱构建
│   │   └── 评测指标体系设计
│   └── A6. 前沿评测方法论
│       ├── 对抗性评测
│       ├── 分布外泛化评测
│       ├── 多模态评测框架
│       ├── 自适应评测
│       └── 评测数据动态更新机制
│
├── B. Prompt工程与Rubrics设计
│   ├── B1. Prompt设计基础
│   │   ├── Zero-shot Prompting
│   │   ├── Few-shot Prompting
│   │   ├── Chain-of-Thought (CoT) Prompting
│   │   ├── Self-Consistency
│   │   ├── Tree-of-Thought (ToT)
│   │   └── ReAct Prompting
│   ├── B2. Prompt设计方法论
│   │   ├── 角色设定
│   │   ├── 任务描述规范化
│   │   ├── 输出格式约束
│   │   ├── Few-shot示例选择策略
│   │   └── 多轮交互Prompt设计
│   ├── B3. Rubrics设计
│   │   ├── Rubrics基本概念与作用
│   │   ├── 多级评分Rubrics设计
│   │   ├── 不同评测维度的Rubrics设计
│   │   ├── Rubrics体系化设计（覆盖完整能力图谱）
│   │   └── 自适应Rubrics系统
│   ├── B4. LLM-as-Judge技术
│   │   ├── LLM-as-Judge基本原理
│   │   ├── Pairwise评判方法
│   │   ├── Pointwise评判方法
│   │   ├── Reference-based评判方法
│   │   ├── 裁判模型选型与校准
│   │   └── 系统性偏差识别与消除
│   ├── B5. 高级Prompt技术
│   │   ├── 对抗性Prompt设计
│   │   ├── 跨语言Prompt
│   │   ├── 多Agent协作Prompt评测
│   │   ├── 工具调用Prompt评测
│   │   └── 长上下文Prompt评测
│   └── B6. 评测结果自动化分析
│       ├── 使用LLM按错误类型自动归类
│       ├── 自动化评测分析Pipeline搭建
│       └── 评测结果可视化与报告生成
│
├── C. 评测归因分析与Agent评测
│   ├── C1. 评测归因分析基础
│   │   ├── 归因分析基本概念
│   │   ├── 归因维度（数据质量、训练阶段、模型架构、Prompt设计）
│   │   └── 评测数据分类方法
│   ├── C2. 系统化归因分析方法
│   │   ├── 数据切片分析
│   │   ├── 对比实验设计
│   │   ├── 数据分布归因
│   │   ├── 模型能力归因
│   │   └── Prompt设计归因
│   ├── C3. "评测-归因-改进-验证"闭环
│   │   ├── 闭环方法论设计
│   │   ├── 改进效果验证方法
│   │   ├── 模型能力短板自动化诊断
│   │   └── 因果推断在归因中的应用
│   ├── C4. Agent评测基础
│   │   ├── Agent基本概念
│   │   ├── 常见Agent框架（LangChain、AutoGen、CrewAI）
│   │   ├── Agent评测指标（工具调用准确性、多步推理成功率、任务完成率）
│   │   └── Agent评测Benchmark
│   ├── C5. Agent评测Benchmark详解
│   │   ├── SWE-bench（软件工程Agent评测）
│   │   ├── WebArena（Web自动化Agent评测）
│   │   ├── ToolBench（工具调用Agent评测）
│   │   ├── AgentBench
│   │   ├── GAIA
│   │   └── Tau-Bench / OSWorld
│   ├── C6. 复杂Agent评测方案设计
│   │   ├── 多步工具调用评测
│   │   ├── 跨工具协作评测
│   │   ├── 错误恢复评测
│   │   ├── 长程任务评测
│   │   ├── 多Agent协作评测
│   │   ├── 人机协作评测
│   │   └── 安全对齐评测
│   └── C7. Agent-as-Judge方法
│       ├── 执行轨迹评判
│       ├── 中间步骤验证
│       ├── 最终结果评估
│       └── 全链路评判方案
│
├── D. 评测数据构建与裁判模型应用
│   ├── D1. 评测数据构建
│   │   ├── 使用已有Benchmark数据集
│   │   ├── 按模板构建简单评测数据
│   │   ├── 高质量评测数据集构建
│   │   ├── 难度梯度设计
│   │   ├── 参考答案与评分Rubrics编写
│   │   └── 标注一致性检查
│   ├── D2. 大规模评测数据生产
│   │   ├── 标注规范制定
│   │   ├── 标注员培训与质检
│   │   ├── 多轮迭代优化
│   │   ├── 多轮数据构建
│   │   ├── 对抗样本构造
│   │   └── 领域专家标注流程管理
│   ├── D3. 评测数据全生命周期管理
│   │   ├── 数据生产
│   │   ├── 数据质检
│   │   ├── 数据更新
│   │   ├── 数据退役
│   │   └── 数据防泄露与动态更新机制
│   ├── D4. 裁判模型应用
│   │   ├── LLM-as-Judge进行主观评测
│   │   ├── 裁判模型评分Prompt设计
│   │   ├── 裁判模型选型与校准
│   │   ├── 裁判结果与人工标注一致性优化（>85%）
│   │   └── 系统性偏差识别与消除
│   ├── D5. 高级裁判模型技术
│   │   ├── 多模型集成裁判
│   │   ├── 对抗性裁判
│   │   ├── 领域自适应裁判
│   │   ├── 裁判数据标注
│   │   └── 偏好对齐
│   └── D6. 自动化评测Pipeline
│       ├── 数据准备模块
│       ├── 模型推理模块
│       ├── 裁判评判模块
│       ├── 归因分析模块
│       └── 报告生成模块
│
├── E. 评测体系设计与系统架构
│   ├── E1. 评测体系设计
│   │   ├── 评测方案制定方法论
│   │   ├── 评测场景与能力体系设计
│   │   ├── 评测数据分类标签体系
│   │   ├── 评测指标体系设计
│   │   └── 企业级评测标准与规范
│   ├── E2. 评测系统架构
│   │   ├── 评测数据管理模块
│   │   ├── 模型接入层
│   │   ├── 评测执行引擎
│   │   ├── 裁判模型服务
│   │   ├── 结果分析与可视化模块
│   │   └── 报告生成模块
│   ├── E3. 企业级评测平台
│   │   ├── 高可用架构设计
│   │   ├── 可扩展性设计
│   │   ├── 千万级评测任务调度
│   │   ├── 评测全生命周期工具链
│   │   └── 技术选型
│   ├── E4. Agent评测环境搭建
│   │   ├── 沙箱环境设计
│   │   ├── API Mock服务
│   │   ├── 评测指标采集
│   │   └── 评测环境管理
│   └── E5. 评测方法论创新
│       ├── 自适应评测方案
│       ├── 对抗性评测
│       ├── 分布外泛化评测
│       ├── 多模态评测框架设计
│       └── 行为可解释性评测
│
├── F. 编程与工程能力（隐含要求）
│   ├── F1. Python编程
│   ├── F2. 前后端开发（评测系统搭建）
│   ├── F3. 数据处理与分析
│   ├── F4. 分布式系统与任务调度
│   └── F5. 版本控制与DevOps
│
└── G. 大模型基础理论（前置知识）
    ├── G1. Transformer架构
    ├── G2. 预训练与微调
    ├── G3. RLHF与对齐技术
    ├── G4. 推理优化
    └── G5. 多模态模型基础
```

---

## 3. 分领域详细知识点梳理

### 3.1 大模型评测基础与Benchmark

#### 3.1.1 评测基本概念与理论

- **评测的定义与目的**
  - 什么是模型评测：对LLM能力进行系统化、可量化的衡量
  - 评测的核心目的：能力诊断、模型选型、迭代改进、产品准入
  - 评测在模型研发全流程中的位置（训练→评测→部署→监控）
- **评测范式分类**
  - 客观评测（Objective Evaluation）：有标准答案的评测，如选择题、代码执行测试
  - 主观评测（Subjective Evaluation）：无标准答案或需人工/模型判断的评测，如开放式问答、创意写作
  - 人类评测（Human Evaluation）：人工评分、偏好对比
  - 自动化评测（Automated Evaluation）：使用脚本或模型自动评分
- **评测的核心挑战**
  - 数据污染（Data Contamination）：训练数据包含评测数据
  - 泛化性问题：Benchmark表现好不代表实际应用好
  - 可复现性：不同评测环境、随机种子导致结果差异
  - 评测覆盖度：单一Benchmark无法覆盖所有能力
  - 评测偏差：评测数据分布与实际应用场景不匹配

#### 3.1.2 业界主流Benchmark详解

- **多学科知识评测**
  - **MMLU**（Massive Multitask Language Understanding）
    - 57个学科领域，多项选择题格式
    - 设计理念：广泛覆盖学术知识
    - 局限性：仅考察知识记忆，不考察推理；多项选择题格式有猜测概率
  - **MMLU-Pro**：MMLU的增强版，增加推理难度
  - **CMMLU**：中文多学科知识评测
  - **C-Eval**：中文大模型综合能力评测
- **代码生成评测**
  - **HumanEval**：164个Python编程问题，通过单元测试验证
    - 设计理念：函数级代码生成能力
    - 局限性：问题较简单，覆盖面有限
  - **HumanEval+**：增强版，增加更多测试用例
  - **MBPP**（Mostly Basic Python Problems）：974个基础Python编程任务
  - **LiveCodeBench**：从实时竞赛平台获取题目，防止数据污染
  - **SWE-bench**：真实GitHub Issue修复，考察工程级代码能力
- **数学推理评测**
  - **GSM8K**：8500道小学数学应用题
    - 设计理念：基础数学推理
    - 局限性：难度偏低，已接近饱和
  - **MATH**：12500道竞赛数学题
    - 设计理念：高难度数学推理
  - **GPQA**（Graduate-Level Google-Proof Q&A）
    - 生物学、物理学、化学研究生级别问题
    - 设计理念：专家级知识推理，防搜索
    - GPQA Diamond：最难的子集，约200题
  - **AIME**：美国数学竞赛题目
- **多轮对话评测**
  - **MT-Bench**：80个多轮对话问题，GPT-4评分
    - 设计理念：多轮对话连贯性和深度
  - **AlpacaEval**：基于AlpacaFarm的自动评测
- **指令遵循评测**
  - **IFEval**（Instruction-Following Evaluation）
    - 541个Prompt，包含可编程验证的指令
    - 设计理念：测试模型对显式指令的遵循能力
    - 特点：无需人工或模型裁判，完全自动化
- **常识与推理评测**
  - **HellaSwag**：常识推理完形填空
  - **BBH**（BIG-Bench Hard）：23个高难度推理任务
  - **ARC**（AI2 Reasoning Challenge）：科学推理问答
- **安全与对齐评测**
  - **TruthfulQA**：测试模型是否输出真实信息
  - **ToxiGen**：测试模型生成有害内容倾向
- **综合能力评测**
  - **BigBench**：200+任务的超大规模评测
  - **AGIEval**：多领域综合能力评测

#### 3.1.3 评测框架与工具

- **lm-eval-harness**（EleutherAI）
  - 最广泛使用的开源LLM评测框架
  - 支持200+Benchmark，标准化评测接口
  - GitHub: https://github.com/EleutherAI/lm-evaluation-harness
- **OpenCompass**（上海AI Lab）
  - 面向大模型的全能评测体系
  - 支持中英文多模态评测
  - GitHub: https://github.com/open-compass/opencompass
- **HELM**（Holistic Evaluation of Language Models，Stanford）
  - Stanford CRFM推出的全息评测框架
  - 强调多维度的全面评测
  - GitHub: https://github.com/stanford-crfm/helm
- **OpenAI Evals**
  - OpenAI官方评测框架
  - 提供标准化的评测注册机制
  - GitHub: https://github.com/openai/evals
- **FastChat评测工具**
  - 集成MT-Bench和Chatbot Arena评测
  - 支持多模型对比
  - GitHub: https://github.com/lm-sys/FastChat
- **Promptfoo**
  - LLM评测和红队测试框架
  - 支持CI/CD集成
  - GitHub: https://github.com/promptfoo/promptfoo
- **DeepEval**
  - 类似unittest的LLM输出评测框架
  - GitHub: https://github.com/confident-ai/deepeval
- **RAGAS**
  - RAG（检索增强生成）专用评测框架
  - GitHub: https://github.com/explodinggradients/ragas
- **TruLens**
  - LLM应用追踪和评测
  - GitHub: https://github.com/truera/trulens

#### 3.1.4 评测数据构建方法

- **人工标注流程设计**
  - 标注规范文档编写
  - 标注员培训流程
  - 标注质量控制（双标注、仲裁机制）
  - 标注一致性度量（Cohen's Kappa、Fleiss' Kappa）
- **数据质量控制**
  - 数据质量审核流程
  - 标注一致性检查方法
  - 数据质量量化指标
- **数据去重与防泄露策略**
  - 训练-评测数据去重（n-gram去重、语义相似度去重）
  - 数据污染检测（llm-decontaminator工具）
  - 动态评测数据更新（LiveBench思路：定期刷新评测数据）
- **难度梯度设计**
  - 按能力水平分级（基础/进阶/挑战）
  - 难度标注方法
  - 难度分布优化
- **对抗样本构造**
  - 对抗性Prompt设计
  - 边界条件测试
  - 模型弱点针对性构造

#### 3.1.5 评测场景与能力维度体系

- **能力维度划分方法论**
  - 认知能力维度（知识/理解/推理/创造）
  - 任务能力维度（代码/数学/写作/对话/工具调用）
  - 安全能力维度（真实性/无害性/公平性）
- **评测场景设计原则**
  - 覆盖性原则：覆盖目标能力全谱
  - 代表性原则：评测场景需代表实际应用
  - 可区分性原则：能区分不同能力水平的模型
  - 可扩展性原则：支持新增评测维度
- **能力图谱构建**
  - 能力树设计
  - 能力间关联关系建模
  - 能力短板识别

#### 3.1.6 前沿评测方法论

- **对抗性评测**：设计对抗性输入测试模型鲁棒性
- **分布外泛化评测**：测试模型在训练分布外的表现
- **多模态评测框架**：图文、视频、音频等多模态能力评测
- **自适应评测**：根据模型能力动态调整评测难度与数据分布
- **评测数据动态更新机制**：定期更新评测数据防止污染

---

### 3.2 Prompt工程与Rubrics设计

#### 3.2.1 Prompt设计基础技术

- **Zero-shot Prompting**
  - 不提供示例，直接给出任务描述
  - 适用场景：简单任务、模型已具备相关能力
  - 局限性：复杂任务效果不稳定
- **Few-shot Prompting**
  - 提供少量示例引导模型行为
  - 示例选择策略：多样性、代表性、难度递进
  - 示例数量影响：1-shot vs 3-shot vs 5-shot
- **Chain-of-Thought (CoT) Prompting**
  - 引导模型展示推理过程
  - Zero-shot CoT："Let's think step by step"
  - Few-shot CoT：提供带推理过程的示例
  - 适用场景：数学推理、逻辑推理、多步决策
- **Self-Consistency**
  - 多次采样推理路径，取多数投票结果
  - 提升CoT的稳定性和准确率
  - 原理：正确推理路径更可能收敛到同一答案
- **Tree-of-Thought (ToT)**
  - 树形搜索推理路径
  - 支持回溯和多路径探索
  - 适用场景：需要规划的复杂问题
- **ReAct Prompting**
  - Reasoning + Acting 结合
  - 交替进行推理和工具调用
  - Agent系统的核心Prompt模式

#### 3.2.2 Prompt设计方法论

- **角色设定**
  - 为模型定义专业角色
  - 角色设定对输出质量和风格的影响
- **任务描述规范化**
  - 清晰、无歧义的任务描述
  - 任务约束的显式声明
- **输出格式约束**
  - JSON/Markdown/结构化输出
  - 格式约束的Prompt技巧
- **Few-shot示例选择策略**
  - 基于相似度的示例选择
  - 基于多样性的示例选择
  - 动态示例选择
- **多轮交互Prompt设计**
  - 上下文管理策略
  - 对话状态跟踪
  - 多轮一致性维护

#### 3.2.3 Rubrics设计

- **Rubrics基本概念**
  - Rubrics定义：将主观判断转化为结构化、可量化的评分标准
  - Rubrics在评测中的作用：提升评测一致性和可复现性
  - 主观评测→客观评分的转化机制
- **多级评分Rubrics设计**
  - 评分等级设计（如1-5分制、1-10分制）
  - 每级的评分标准描述
  - 评分标准的行为锚定
- **不同评测维度的Rubrics设计**
  - 准确性维度Rubrics
  - 完整性维度Rubrics
  - 流畅性维度Rubrics
  - 创造性维度Rubrics
  - 安全性维度Rubrics
- **Rubrics体系化设计**
  - 覆盖完整能力图谱
  - 处理维度间关联与冲突
  - Rubrics权重设计
- **自适应Rubrics系统**
  - 根据模型能力水平动态调整评分标准难度
  - 自适应评分校准

#### 3.2.4 LLM-as-Judge技术

- **LLM-as-Judge基本原理**
  - 使用强模型（如GPT-4/Claude）作为裁判模型
  - 将主观评测自动化
  - 裁判Prompt设计原则
- **Pairwise评判方法**
  - 给裁判模型展示两个回答，选择更好的一个
  - 优势：相对判断比绝对评分更稳定
  - 类似Chatbot Arena的Elo评分机制
- **Pointwise评判方法**
  - 对单个回答进行独立评分
  - 优势：可并行评测，效率高
  - 局限性：绝对评分的一致性较差
- **Reference-based评判方法**
  - 提供参考答案，裁判模型对比评分
  - 优势：有标准参照，评分更客观
  - 局限性：需要高质量参考答案
- **裁判模型选型与校准**
  - 裁判模型选型标准（能力、成本、速度）
  - 裁判模型与人工标注一致性校准
  - 一致性目标：>85%
- **系统性偏差识别与消除**
  - **Position Bias（位置偏差）**：裁判倾向于选择特定位置的回答
    - 消除方法：交换位置取平均
  - **Verbosity Bias（冗长偏差）**：裁判倾向于选择更长的回答
    - 消除方法：长度归一化
  - **Self-Enhancement Bias（自我增强偏差）**：裁判倾向于偏好同族模型的回答
    - 消除方法：使用不同厂商模型交叉裁判

#### 3.2.5 高级Prompt技术

- **对抗性Prompt设计**
  - 越狱Prompt（Jailbreak）
  - Prompt注入攻击
  - 防御性Prompt设计
- **跨语言Prompt**
  - 多语言Prompt设计原则
  - 跨语言能力评测
- **多Agent协作Prompt评测**
  - 多Agent交互的Prompt设计
  - Agent间通信协议
- **工具调用Prompt评测**
  - Function Calling评测
  - 工具选择准确性评测
  - 参数生成准确性评测
- **长上下文Prompt评测**
  - 长文本理解能力评测
  - "Lost in the Middle"问题
  - Needle in a Haystack评测

#### 3.2.6 评测结果自动化分析

- **使用LLM按错误类型自动归类**
  - 错误分类体系设计
  - 自动分类Prompt设计
  - 分类结果质量控制
- **自动化评测分析Pipeline**
  - 数据准备→模型推理→裁判评判→归因分析→报告生成
  - Pipeline编排与调度
- **评测结果可视化与报告生成**
  - 多维度评测结果展示
  - 雷达图/热力图/趋势图
  - 自动化报告模板

---

### 3.3 评测归因分析与Agent评测

#### 3.3.1 评测归因分析基础

- **归因分析基本概念**
  - 什么是评测归因：从评测结果反推模型能力短板的原因
  - 归因分析的价值：指导模型改进方向
- **归因维度**
  - **数据质量归因**：评测数据本身是否存在问题
  - **训练阶段归因**：预训练/SFT/RLHF阶段的问题
  - **模型架构归因**：模型规模、架构设计导致的能力限制
  - **Prompt设计归因**：Prompt不当导致的表现下降
- **评测数据分类方法**
  - 按能力维度分类
  - 按难度级别分类
  - 按任务类型分类
  - 按错误类型分类

#### 3.3.2 系统化归因分析方法

- **数据切片分析**
  - 按能力维度切片
  - 按难度级别切片
  - 按数据分布特征切片
  - 切片间对比分析
- **对比实验设计**
  - 控制变量实验
  - A/B测试方法论
  - 消融实验
- **数据分布归因**
  - 训练数据与评测数据分布差异分析
  - 数据分布偏移检测
- **模型能力归因**
  - 能力维度交叉分析
  - 能力短板定位
  - 能力间依赖关系分析
- **Prompt设计归因**
  - Prompt变体对比
  - Prompt敏感性分析

#### 3.3.3 "评测-归因-改进-验证"闭环

- **闭环方法论**
  - 评测发现问题→归因定位原因→制定改进方案→验证改进效果
  - 闭环的迭代节奏与质量控制
- **改进效果验证方法**
  - 前后对比评测
  - 回归测试（确保改进不引入新问题）
  - 统计显著性检验
- **模型能力短板自动化诊断**
  - 自动化归因分析Pipeline
  - 能力短板报告自动生成
- **因果推断在归因中的应用**
  - 因果图构建
  - 反事实推理
  - DoWhy框架应用

#### 3.3.4 Agent评测基础

- **Agent基本概念**
  - 什么是AI Agent：能感知、推理、行动的自主系统
  - Agent核心组件：规划、记忆、工具使用、反思
  - Agent vs Chatbot：自主决策能力
- **常见Agent框架**
  - **LangChain**：最流行的LLM应用框架，支持链式调用、Agent、记忆
    - GitHub: https://github.com/langchain-ai/langchain
  - **AutoGen**（Microsoft）：多Agent对话框架
    - GitHub: https://github.com/microsoft/autogen
  - **CrewAI**：基于角色的多Agent协作框架
    - GitHub: https://github.com/crewAIInc/crewAI
- **Agent评测指标**
  - 工具调用准确性（Tool Call Accuracy）
  - 多步推理成功率（Multi-step Reasoning Success Rate）
  - 任务完成率（Task Completion Rate）
  - 路径效率（Path Efficiency）
  - 错误恢复能力（Error Recovery Rate）

#### 3.3.5 Agent评测Benchmark详解

- **SWE-bench**
  - 来源：Princeton NLP
  - 内容：2294个真实GitHub Issue修复任务
  - 评测方式：通过隐藏单元测试验证补丁
  - 变体：SWE-bench Lite（简化版）、SWE-bench Verified（人工验证版）
  - 网站: https://www.swebench.com/
- **WebArena**
  - 来源：CMU
  - 内容：812个Web自动化任务，覆盖5个领域
  - 评测方式：在真实Web环境中执行任务
  - 特点：需要多步Web交互
  - 网站: https://webarena.dev/
- **ToolBench**
  - 内容：大规模工具调用评测
  - 评测方式：多步工具使用和API驱动推理
- **AgentBench**
  - 多环境Agent能力综合评测
- **GAIA**
  - 通用AI Assistant评测
  - 需要多步推理、工具使用、Web搜索
- **Tau-Bench / OSWorld**
  - 操作系统级别的Agent评测

#### 3.3.6 复杂Agent评测方案设计

- **多步工具调用评测**
  - 工具链式调用
  - 条件分支工具调用
  - 工具组合策略评测
- **跨工具协作评测**
  - 多工具协同完成任务
  - 工具间数据传递
  - 工具冲突处理
- **错误恢复评测**
  - 工具调用失败后的恢复策略
  - 错误诊断能力
  - 重试与替代方案
- **长程任务评测**
  - 多步骤（10+步）任务执行
  - 长程记忆与上下文管理
  - 任务分解与规划
- **多Agent协作评测**
  - Agent间通信效率
  - 任务分配合理性
  - 协作结果质量
- **人机协作评测**
  - 人机交互效率
  - 人类干预频率
  - 协作满意度
- **安全对齐评测**
  - Agent行为安全性
  - 权限控制
  - 有害操作防护

#### 3.3.7 Agent-as-Judge方法

- **执行轨迹评判**
  - 评判Agent的每步决策是否合理
  - 轨迹质量评分
- **中间步骤验证**
  - 验证每步执行结果的正确性
  - 步骤间逻辑一致性检查
- **最终结果评估**
  - 任务完成质量评分
  - 结果与预期目标的匹配度
- **全链路评判方案**
  - 轨迹评判+中间验证+结果评估的完整方案
  - 多维度综合评分

---

### 3.4 评测数据构建与裁判模型应用

#### 3.4.1 评测数据构建

- **使用已有Benchmark数据集**
  - 数据集获取与格式转换
  - 数据集适配与定制化
- **按模板构建简单评测数据**
  - 评测数据模板设计
  - 批量生成方法
- **高质量评测数据集构建**
  - 数据需求分析
  - 数据来源规划
  - 数据质量标准制定
- **难度梯度设计**
  - 难度定义与量化
  - 难度梯度划分方法
  - 难度验证与校准
- **参考答案与评分Rubrics编写**
  - 高质量参考答案编写原则
  - 评分Rubrics与参考答案的匹配
  - 部分正确情况的处理
- **标注一致性检查**
  - Cohen's Kappa（双标注）
  - Fleiss' Kappa（多标注）
  - 一致性不达标时的处理流程

#### 3.4.2 大规模评测数据生产

- **标注规范制定**
  - 标注指南文档结构
  - 标注示例与反例
  - 边界情况处理规范
- **标注员培训与质检**
  - 标注员准入测试
  - 定期质检与反馈
  - 标注员能力分级管理
- **多轮迭代优化**
  - 初标→质检→修订→终标的流程
  - 标注质量趋势追踪
- **多轮数据构建**
  - 多轮对话数据构建方法
  - 上下文连贯性保证
- **对抗样本构造**
  - 基于模型弱点的对抗样本设计
  - 自动化对抗样本生成
- **领域专家标注流程管理**
  - 领域专家招募与资质验证
  - 专家标注质量管理
  - 专家标注与普通标注的融合

#### 3.4.3 评测数据全生命周期管理

- **数据生产**
  - 数据生产流程标准化
  - 生产力度量与优化
- **数据质检**
  - 自动化质检规则
  - 人工抽检流程
  - 质检报告生成
- **数据更新**
  - 定期更新机制
  - 增量更新策略
  - 版本管理
- **数据退役**
  - 退役标准（污染检测、时效性）
  - 退役流程
  - 退役数据归档
- **数据防泄露与动态更新机制**
  - 数据加密与访问控制
  - 污染检测自动化
  - LiveBench思路的动态更新

#### 3.4.4 裁判模型应用

- **LLM-as-Judge进行主观评测**
  - 裁判模型Prompt设计
  - 评测流程标准化
- **裁判模型评分Prompt设计**
  - 评分标准嵌入Prompt
  - 输出格式约束
  - 评分理由要求
- **裁判模型选型与校准**
  - 选型标准：能力、成本、速度、API可用性
  - 校准方法：与人工标注对齐
  - 校准数据集构建
- **裁判结果与人工标注一致性优化**
  - 一致性度量指标
  - 一致性<85%时的诊断与优化
  - 优化策略：Prompt调优、模型切换、集成裁判
- **系统性偏差识别与消除**
  - Position Bias：交换位置取平均
  - Verbosity Bias：长度归一化
  - Self-Enhancement Bias：跨厂商裁判
  - 偏差量化方法（Calm框架）

#### 3.4.5 高级裁判模型技术

- **多模型集成裁判**
  - 多裁判投票机制
  - 加权集成策略
  - 裁判多样性保证
- **对抗性裁判**
  - 对抗性裁判Prompt设计
  - 鲁棒性增强
- **领域自适应裁判**
  - 领域特定裁判微调
  - 领域知识注入
- **裁判数据标注**
  - 裁判训练数据构建
  - 偏好数据标注方法
- **偏好对齐**
  - DPO/RLHF在裁判模型上的应用
  - 偏好对齐训练流程

#### 3.4.6 自动化评测Pipeline

- **数据准备模块**
  - 数据加载与预处理
  - 数据格式标准化
  - 数据分发与并行化
- **模型推理模块**
  - 多模型接入（OpenAI API、本地模型、开源模型）
  - 推理参数管理（temperature、top_p等）
  - 推理结果缓存
- **裁判评判模块**
  - 裁判模型调用
  - 评判结果解析
  - 评判结果质量控制
- **归因分析模块**
  - 自动化归因分析
  - 能力短板报告
- **报告生成模块**
  - 评测报告模板
  - 可视化图表生成
  - 多格式报告输出

---

### 3.5 评测体系设计与系统架构

#### 3.5.1 评测体系设计

- **评测方案制定方法论**
  - 业务需求分析→能力维度拆解→评测场景设计→数据构建→指标设计→执行方案
- **评测场景与能力体系设计**
  - 自顶向下的能力体系设计
  - 场景-能力映射矩阵
- **评测数据分类标签体系**
  - 多维度标签设计（能力/难度/类型/领域）
  - 标签体系管理
- **评测指标体系设计**
  - 准确率类指标
  - 质量类指标（BLEU、ROUGE等）
  - 偏好类指标（Elo评分、胜率）
  - 效率类指标（延迟、吞吐）
- **企业级评测标准与规范**
  - 评测流程标准化
  - 评测报告规范
  - 评测准入标准

#### 3.5.2 评测系统架构

- **评测数据管理模块**
  - 数据存储与索引
  - 数据版本管理
  - 数据访问权限控制
- **模型接入层**
  - 统一模型接口设计
  - 多模型适配器
  - 推理负载均衡
- **评测执行引擎**
  - 任务调度与编排
  - 并行评测执行
  - 失败重试机制
- **裁判模型服务**
  - 裁判模型部署
  - 裁判结果缓存
  - 裁判质量监控
- **结果分析与可视化模块**
  - 多维度分析
  - 交互式可视化
  - 历史趋势对比
- **报告生成模块**
  - 自动化报告生成
  - 多格式输出（PDF/HTML/Markdown）

#### 3.5.3 企业级评测平台

- **高可用架构设计**
  - 服务高可用（多副本、故障转移）
  - 数据高可用（备份、容灾）
- **可扩展性设计**
  - 水平扩展能力
  - 插件化架构
  - 评测任务动态扩缩容
- **千万级评测任务调度**
  - 分布式任务调度框架
  - 任务优先级管理
  - 资源池管理
- **评测全生命周期工具链**
  - 数据生产→评测执行→结果分析→报告生成→改进追踪
- **技术选型**
  - 前端：React/Vue
  - 后端：Python(FastAPI/Django)/Go
  - 数据库：PostgreSQL/MongoDB/Redis
  - 消息队列：Kafka/RabbitMQ
  - 任务调度：Celery/Airflow
  - 容器化：Docker/Kubernetes

#### 3.5.4 Agent评测环境搭建

- **沙箱环境设计**
  - 安全隔离的执行环境
  - 资源限制与监控
  - Docker容器化沙箱
- **API Mock服务**
  - 工具API模拟
  - 响应可配置
  - 异常场景模拟
- **评测指标采集**
  - 执行轨迹记录
  - 性能指标采集
  - 行为日志收集
- **评测环境管理**
  - 环境配置管理
  - 环境复现性保证
  - 多环境并行管理

#### 3.5.5 评测方法论创新

- **自适应评测方案**
  - 根据模型能力动态调整评测难度
  - 自适应评测算法设计
- **对抗性评测**
  - 自动化对抗样本生成
  - 对抗鲁棒性评估
- **分布外泛化评测**
  - OOD检测与评估
  - 泛化能力量化
- **多模态评测框架设计**
  - 图文/视频/音频多模态评测
  - 跨模态能力评估
- **行为可解释性评测**
  - 模型行为解释
  - 决策可解释性量化

---

## 4. 分领域学习资料规划

### 4.1 大模型评测基础与Benchmark — 学习资料

#### 📚 书籍

| # | 名称 | 作者/来源 | 类型 | 难度 | 推荐理由 |
|---|------|-----------|------|------|----------|
| 1 | **Speech and Language Processing (3rd ed. draft)** | Dan Jurafsky & James H. Martin (Stanford) | 书籍（在线免费） | 中级 | NLP领域圣经级教材，第3版新增了大语言模型章节，涵盖LLM评测基础理论。在线地址: https://web.stanford.edu/~jurafsky/slp3/ |
| 2 | **Build a Large Language Model (From Scratch)** | Sebastian Raschka | 书籍 | 中级 | 从零构建LLM，深入理解模型内部机制，有助于理解评测中发现的模型行为背后的原因。GitHub: https://github.com/rasbt/LLMs-from-scratch |
| 3 | **Natural Language Processing with Transformers** | Lewis Tunstall, Leandro von Werra, Thomas Wolf (Hugging Face) | 书籍 | 中级 | Hugging Face团队出品，涵盖Transformer模型和评测实践，有专门章节讨论模型评估。 |
| 4 | **Evaluating Machine Learning Models** | Alice Zheng (O'Reilly) | 书籍 | 入门 | 虽然不是专门针对LLM，但系统介绍了ML评测方法论，包括离线评测、在线评测、A/B测试等基础概念。 |

#### 🎓 大学课程

| # | 名称 | 来源 | 类型 | 难度 | 推荐理由 |
|---|------|------|------|------|----------|
| 1 | **CS224N: Natural Language Processing with Deep Learning** | Stanford University | 大学课程 | 中级 | Stanford经典NLP课程，涵盖NLP基础到前沿，是理解LLM的必修课。2024版含Transformer和LLM内容。课程主页: https://web.stanford.edu/class/cs224n/ |
| 2 | **CS324: Large Language Models** | Stanford University | 大学课程 | 高级 | Stanford专门针对LLM的课程，涵盖模型训练、评测、安全等，有专门的评测章节。课程主页: https://stanford-cs324.github.io/ |
| 3 | **CS25: Transformers United** | Stanford University | 大学课程 | 中级 | 聚焦Transformer架构和应用的研讨会式课程，涵盖LLM评测前沿研究。 |
| 4 | **Practical Deep Learning for Coders** | Fast.ai (Jeremy Howard) | 在线课程 | 入门 | 实践导向的深度学习课程，快速上手NLP和LLM应用开发。网站: https://course.fast.ai/ |

#### 📝 博客与文章

| # | 名称 | 作者/来源 | 类型 | 难度 | 推荐理由 |
|---|------|-----------|------|------|----------|
| 1 | **A Survey on LLM-as-a-Judge** | arxiv (2411.15594) | 综述论文 | 高级 | LLM-as-Judge领域的全面综述，涵盖可靠性增强策略、偏差消除方法。https://arxiv.org/abs/2411.15594 |
| 2 | **Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena** | Lianmin Zheng et al. (NeurIPS 2023) | 论文 | 高级 | LLM-as-Judge的奠基性论文，首次系统分析了位置偏差、冗长偏差、自我增强偏差。https://arxiv.org/abs/2306.05685 |
| 3 | **HELM: Holistic Evaluation of Language Models** | Stanford CRFM | 论文/博客 | 高级 | Stanford的全息评测方法论，系统设计了评测维度和指标体系。https://crfm.stanford.edu/helm/ |
| 4 | **LiveBench: A Challenging, Contamination-Free LLM Benchmark** | LiveBench团队 | 项目/论文 | 中级 | 创新的无污染评测方案，定期刷新评测数据。https://livebench.ai/ |
| 5 | **Quantifying Biases in LLM-as-a-Judge** | arxiv (2410.02736) | 论文 | 高级 | Calm框架，系统量化12种LLM裁判偏差类型。 |
| 6 | **LLM Benchmarks Explained** | CanItRun | 博客 | 入门 | 通俗易懂地解释各主流Benchmark的设计理念、评测维度和局限性。https://canitrun.dev/guides/benchmark-scores-explained/ |
| 7 | **Instruction-Following Evaluation for Large Language Models (IFEval)** | Google Research | 论文 | 中级 | IFEval的原始论文，介绍了无需裁判的自动化指令遵循评测方法。https://arxiv.org/abs/2311.07911 |

#### 🔧 开源项目/代码库

| # | 名称 | 来源 | 类型 | 难度 | 推荐理由 |
|---|------|------|------|------|----------|
| 1 | **lm-eval-harness** | EleutherAI | 开源框架 | 中级 | 最广泛使用的LLM评测框架，支持200+Benchmark。**必学项目**，深入理解评测框架的设计模式和扩展方法。GitHub: https://github.com/EleutherAI/lm-evaluation-harness |
| 2 | **OpenCompass** | 上海AI Lab | 开源框架 | 中级 | 全能型大模型评测体系，支持中英文多模态。**必学项目**，学习企业级评测框架的设计。GitHub: https://github.com/open-compass/opencompass |
| 3 | **HELM** | Stanford CRFM | 开源框架 | 高级 | Stanford全息评测框架，强调多维度全面评测。学习其评测维度设计理念。GitHub: https://github.com/stanford-crfm/helm |
| 4 | **OpenAI Evals** | OpenAI | 开源框架 | 中级 | OpenAI官方评测框架，学习标准化的评测注册机制和评测流程。GitHub: https://github.com/openai/evals |
| 5 | **FastChat** | LMSYS (UC Berkeley) | 开源框架 | 中级 | 集成MT-Bench和Chatbot Arena评测。学习人类偏好评测和Elo评分机制。GitHub: https://github.com/lm-sys/FastChat |
| 6 | **llm-decontaminator** | LMSYS | 开源工具 | 中级 | 评测数据去污染工具，学习数据污染检测方法。GitHub: https://github.com/lm-sys/llm-decontaminator |
| 7 | **Awesome-LLMs-as-Judges** | CSHaitao | 资源汇总 | 中级 | LLM-as-Judge论文和资源汇总，持续更新。GitHub: https://github.com/CSHaitao/Awesome-LLMs-as-Judges |

---

### 4.2 Prompt工程与Rubrics设计 — 学习资料

#### 📚 书籍

| # | 名称 | 作者/来源 | 类型 | 难度 | 推荐理由 |
|---|------|------|------|------|----------|
| 1 | **Prompt Engineering Guide** | DAIR.AI | 在线书籍 | 入门→中级 | 最全面的Prompt工程指南，涵盖Zero-shot到高级技术。**必读**。https://www.promptingguide.ai/ |
| 2 | **The Art of Prompt Engineering** | Nathan Hunter | 书籍 | 入门 | 实践导向的Prompt工程入门书，包含大量案例。 |
| 3 | **Prompt Engineering for LLMs** | O'Reilly | 书籍 | 中级 | O'Reilly出品的Prompt工程实践指南，涵盖评测中的Prompt设计。 |

#### 🎓 课程

| # | 名称 | 来源 | 类型 | 难度 | 推荐理由 |
|---|------|------|------|------|----------|
| 1 | **ChatGPT Prompt Engineering for Developers** | DeepLearning.AI (Andrew Ng & OpenAI) | 在线课程（免费） | 入门 | Andrew Ng与OpenAI联合推出的Prompt工程课，**必学入门课**。https://www.deeplearning.ai/courses/chatgpt-prompt-engineering-for-developers/ |
| 2 | **Advanced Retrieval for AI with Chroma** | DeepLearning.AI | 在线课程 | 中级 | 深入检索增强和Prompt优化的高级课程。 |
| 3 | **Prompt Engineering Course** | elvatorhino/learn_prompting | 开源课程 | 入门→高级 | 社区驱动的Prompt工程教程，从基础到红队测试全覆盖。GitHub: https://github.com/elvatorhino/learn_prompting |
| 4 | **LangChain Academy** | LangChain | 在线课程 | 中级 | LangChain官方学院课程，涵盖Agent和Prompt工程实践。https://academy.langchain.com/ |

#### 📝 博客与论文

| # | 名称 | 作者/来源 | 类型 | 难度 | 推荐理由 |
|---|------|------|------|------|----------|
| 1 | **Chain-of-Thought Prompting Elicits Reasoning in Large Language Models** | Jason Wei et al. (Google) | 论文 | 中级 | CoT技术的奠基性论文，理解CoT原理的必读文献。https://arxiv.org/abs/2201.11903 |
| 2 | **Self-Consistency Improves Chain of Thought Reasoning** | Xuezhi Wang et al. (Google) | 论文 | 中级 | Self-Consistency技术的原始论文。https://arxiv.org/abs/2203.11171 |
| 3 | **Tree of Thoughts: Deliberate Problem Solving with Large Language Models** | Shunyu Yao et al. (Princeton) | 论文 | 高级 | ToT技术的原始论文，理解高级推理Prompt设计。https://arxiv.org/abs/2305.10601 |
| 4 | **ReAct: Synergizing Reasoning and Acting in Language Models** | Shunyu Yao et al. (Princeton) | 论文 | 中级 | ReAct技术的原始论文，Agent Prompt设计的核心。https://arxiv.org/abs/2210.03629 |
| 5 | **Designing Evaluation Prompts** | HuggingFace Evaluation Guidebook | 博客 | 中级 | HuggingFace的评测Prompt设计指南，实践性强。https://deepwiki.com/huggingface/evaluation-guidebook/2.3.2-designing-evaluation-prompts |
| 6 | **Lost in the Middle: How Language Models Use Long Contexts** | Liu et al. (Stanford) | 论文 | 中级 | 长上下文评测的关键论文，理解"Lost in the Middle"问题。https://arxiv.org/abs/2307.03172 |
| 7 | **LLM-Judge Prompt Engineering Guide** | FutureAGI | 博客 | 中级 | 裁判模型Prompt工程的实践指南。https://futureagi.com/blog/llm-judge-prompt-engineering-guide-2026/ |

#### 🔧 开源项目/代码库

| # | 名称 | 来源 | 类型 | 难度 | 推荐理由 |
|---|------|------|------|------|----------|
| 1 | **Promptfoo** | promptfoo | 开源框架 | 中级 | LLM评测和红队测试框架，支持CI/CD集成。学习评测Prompt的工程化实践。GitHub: https://github.com/promptfoo/promptfoo |
| 2 | **f/awesome-chatgpt-prompts** | GitHub社区 | 资源汇总 | 入门 | 大量Prompt示例集合，学习不同场景的Prompt设计。GitHub: https://github.com/f/awesome-chatgpt-prompts |
| 3 | **DeepEval** | Confident AI | 开源框架 | 中级 | 类似unittest的LLM评测框架，内置多种评测指标。学习评测指标实现。GitHub: https://github.com/confident-ai/deepeval |
| 4 | **OpenAI Cookbook - Evaluation** | OpenAI | 代码示例 | 入门→中级 | OpenAI官方的评测实践示例，包含Prompt设计和评测流程。GitHub: https://github.com/openai/openai-cookbook |

---

### 4.3 评测归因分析与Agent评测 — 学习资料

#### 📚 书籍

| # | 名称 | 作者/来源 | 类型 | 难度 | 推荐理由 |
|---|------|------|------|------|----------|
| 1 | **Causal Inference in Statistics: A Primer** | Judea Pearl et al. | 书籍 | 高级 | 因果推断入门教材，Pearl是因果推断之父，理解归因分析的理论基础。 |
| 2 | **The Book of Why: The New Science of Cause and Effect** | Judea Pearl & Dana Mackenzie | 书籍（科普） | 中级 | 因果推断的科普读物，建立因果思维框架。 |
| 3 | **Causality: Models, Reasoning, and Inference** | Judea Pearl | 书籍 | 专家级 | 因果推断的学术圣经，深入理解因果图和结构方程模型。 |
| 4 | **Experimentation for Engineers** | Gojko Adzic | 书籍 | 中级 | 实验设计与分析方法，适用于评测中的对比实验设计。 |

#### 🎓 课程

| # | 名称 | 来源 | 类型 | 难度 | 推荐理由 |
|---|------|------|------|------|----------|
| 1 | **AI Agents in LangChain** | DeepLearning.AI | 在线课程 | 中级 | 学习如何使用LangChain构建Agent，理解Agent工作原理。https://www.deeplearning.ai/courses/ai-agents-in-langchain/ |
| 2 | **Functions, Tools and Agents with LangChain** | DeepLearning.AI | 在线课程 | 中级 | 深入LangChain的函数调用和工具使用，理解Agent评测的核心对象。 |
| 3 | **Causal Inference** | Brady Neal (YouTube) | 在线课程 | 中级 | 因果推断的优质视频课程，从基础到应用。https://www.youtube.com/@BradyNeal |
| 4 | **CS288: Natural Language Processing** | UC Berkeley | 大学课程 | 高级 | Berkeley NLP课程，涵盖LLM和Agent相关内容。 |

#### 📝 博客与论文

| # | 名称 | 作者/来源 | 类型 | 难度 | 推荐理由 |
|---|------|------|------|------|----------|
| 1 | **SWE-bench: Can Language Models Resolve Real-World GitHub Issues?** | Carlos Jimenez et al. (Princeton) | 论文 | 高级 | SWE-bench的原始论文，理解Agent评测Benchmark的设计。https://arxiv.org/abs/2310.06770 |
| 2 | **WebArena: A Realistic Web Environment for Building Autonomous Agents** | Shuyan Zhou et al. (CMU) | 论文 | 高级 | WebArena的原始论文，学习Web Agent评测环境设计。https://arxiv.org/abs/2307.13854 |
| 3 | **ToolBench / ToolLLM** | Yujia Qin et al. (Tsinghua) | 论文 | 高级 | ToolBench的原始论文，学习工具调用Agent评测。 |
| 4 | **AgentBench: Evaluating LLMs as Agents** | Xiao Liu et al. (Tsinghua) | 论文 | 高级 | AgentBench的原始论文，多环境Agent能力综合评测。https://arxiv.org/abs/2308.03688 |
| 5 | **GAIA: A Benchmark for General AI Assistants** | Grégoire Mialon et al. (Meta) | 论文 | 高级 | GAIA的原始论文，通用AI助手评测。 |
| 6 | **AdaRubric: Task-Adaptive Rubrics for LLM Agent Evaluation** | arxiv | 论文 | 高级 | 自适应Rubrics在Agent评测中的应用。https://arxiv.org/abs/2603.21362 |
| 7 | **Efficient LLM Safety Evaluation through Multi-Agent Debate** | arxiv | 论文 | 高级 | 多Agent辩论用于安全评测的前沿研究。https://arxiv.org/abs/2511.06396 |
| 8 | **Agent Evaluation: LLM-as-Judge, Pass-at-K, and Benchmarks** | OneNeural AI | 博客 | 中级 | Agent评测的综合指南，涵盖主流Benchmark和方法。https://www.oneneural.ai/blog/agent-evaluation-llm-as-judge |

#### 🔧 开源项目/代码库

| # | 名称 | 来源 | 类型 | 难度 | 推荐理由 |
|---|------|------|------|------|----------|
| 1 | **LangChain** | LangChain AI | 开源框架 | 中级 | 最流行的LLM应用框架，**必学**，理解Agent构建和评测对象。GitHub: https://github.com/langchain-ai/langchain |
| 2 | **AutoGen** | Microsoft | 开源框架 | 中级 | 多Agent对话框架，学习多Agent协作模式和评测方法。GitHub: https://github.com/microsoft/autogen |
| 3 | **CrewAI** | CrewAI Inc | 开源框架 | 中级 | 基于角色的多Agent协作框架，学习Agent角色设计和协作评测。GitHub: https://github.com/crewAIInc/crewAI |
| 4 | **SWE-agent** | Princeton NLP | 开源框架 | 高级 | SWE-bench的Agent系统实现，学习Agent评测环境的搭建。GitHub: https://github.com/princeton-nlp/SWE-agent |
| 5 | **SWE-bench** | Princeton NLP | 开源项目 | 高级 | SWE-bench评测框架本身，学习Agent评测Benchmark的实现。GitHub: https://github.com/princeton-nlp/SWE-bench |
| 6 | **WebArena** | CMU | 开源项目 | 高级 | WebArena评测环境，学习Web Agent评测环境搭建。GitHub: https://github.com/web-arena-x/webarena |
| 7 | **DoWhy** | Microsoft/Amazon | 开源框架 | 高级 | 因果推断框架，应用于评测归因分析中的因果推断。GitHub: https://github.com/py-why/dowhy |
| 8 | **Awesome-Agent-Papers** | luo-junyu | 资源汇总 | 中级 | Agent论文汇总，持续追踪Agent评测前沿。GitHub: https://github.com/luo-junyu/Awesome-Agent-Papers |

---

### 4.4 评测数据构建与裁判模型应用 — 学习资料

#### 📚 书籍

| # | 名称 | 作者/来源 | 类型 | 难度 | 推荐理由 |
|---|------|------|------|------|----------|
| 1 | **Dataset Shift in Machine Learning** | Joaquin Quinonero-Candela et al. (MIT Press) | 书籍 | 高级 | 数据分布偏移的经典教材，理解评测数据分布与实际应用差异。 |
| 2 | **Data Labeling: A Practical Guide to Creating High-Quality Datasets** | various | 书籍 | 中级 | 数据标注实践指南，学习标注流程设计和质量控制。 |
| 3 | **Human-in-the-Loop Machine Learning** | Robert Munro (Manning) | 书籍 | 中级 | 人机协作标注的方法论，适用于评测数据构建中的主动学习场景。 |

#### 🎓 课程

| # | 名称 | 来源 | 类型 | 难度 | 推荐理由 |
|---|------|------|------|------|----------|
| 1 | **Data-Centric AI (DCAI)** | MIT / Andrew Ng | 在线课程 | 中级 | MIT和Andrew Ng推动的以数据为中心的AI课程，学习数据质量管理和标注流程。 |
| 2 | **Human-in-the-Loop Machine Learning** | Robert Munro (O'Reilly) | 书籍+课程 | 中级 | 人工标注与机器学习结合的最佳实践。 |

#### 📝 博客与论文

| # | 名称 | 作者/来源 | 类型 | 难度 | 推荐理由 |
|---|------|------|------|------|----------|
| 1 | **Rethinking Benchmark Contamination** | lm-sys | 论文 | 高级 | 评测数据污染的深度分析，学习污染检测和去污染方法。GitHub: https://github.com/lm-sys/llm-decontaminator |
| 2 | **Training an LLM-as-a-Judge Model: Pipeline, Insights, and Practical Considerations** | ECNU | 论文 | 高级 | Themis裁判模型的训练流程，学习裁判模型训练实践。 |
| 3 | **AI Benchmarking: Methodologies, Datasets, and the Contamination Problem** | TokenDyno | 博客 | 中级 | 评测方法论的综合指南，涵盖数据构建、污染控制和可复现性。https://tokendyno.com/blog/ai-benchmarking/ |
| 4 | **Benchmarking LLM accuracy for engineering technical QA** | Patsnap | 博客 | 中级 | 领域特定评测的实践案例，学习去污染、领域相关性评分、置信度校准。 |
| 5 | **LLM Evaluation Rubric: A Scoring Template** | Prodinit | 博客 | 入门 | 实用的LLM评测Rubric模板，可直接参考使用。https://prodinit.com/blog/llm-evaluation-rubric |
| 6 | **Why no one wants to decontaminate their LLM training data** | Martin Gubri | 博客 | 中级 | 深入讨论数据污染的实际影响和去污染的挑战。 |

#### 🔧 开源项目/代码库

| # | 名称 | 来源 | 类型 | 难度 | 推荐理由 |
|---|------|------|------|------|----------|
| 1 | **RAGAS** | explodinggradients | 开源框架 | 中级 | RAG评测专用框架，学习评测指标设计和自动化评测流程。GitHub: https://github.com/explodinggradients/ragas |
| 2 | **TruLens** | TruEra | 开源框架 | 中级 | LLM应用追踪和评测，学习评测指标采集和可视化。GitHub: https://github.com/truera/trulens |
| 3 | **Label Studio** | HumanSignal | 开源工具 | 中级 | 开源数据标注平台，学习标注流程管理和质量控制。GitHub: https://github.com/HumanSignal/label-studio |
| 4 | **Argilla** | Argilla | 开源工具 | 中级 | 开源数据标注和反馈平台，支持LLM评测数据标注。GitHub: https://github.com/argilla-io/argilla |
| 5 | **Cleanlab** | Cleanlab Inc | 开源框架 | 中级 | 数据质量管理和噪声标签检测，学习数据质量控制方法。GitHub: https://github.com/cleanlab/cleanlab |
| 6 | **Promptfoo LLM Rubric** | promptfoo | 开源工具 | 中级 | LLM-as-Judge的Rubric评测工具实现。https://www.promptfoo.dev/docs/configuration/expected-outputs/model-graded/llm-rubric/ |

---

### 4.5 评测体系设计与系统架构 — 学习资料

#### 📚 书籍

| # | 名称 | 作者/来源 | 类型 | 难度 | 推荐理由 |
|---|------|------|------|------|----------|
| 1 | **Designing Data-Intensive Applications** | Martin Kleppmann (O'Reilly) | 书籍 | 高级 | 分布式系统设计圣经，评测平台架构设计的理论基础。**必读**。 |
| 2 | **System Design Interview** | Alex Xu | 书籍 | 中级 | 系统设计实践指南，帮助理解大规模评测平台的架构设计。 |
| 3 | **Building Machine Learning Powered Applications** | Emmanuel Ameisen (O'Reilly) | 书籍 | 中级 | ML应用工程化实践，包含模型评测和监控的工程实践。 |

#### 🎓 课程

| # | 名称 | 来源 | 类型 | 难度 | 推荐理由 |
|---|------|------|------|------|----------|
| 1 | **MIT 6.5840: Distributed Systems** | MIT | 大学课程 | 高级 | 分布式系统课程，学习大规模评测任务调度的理论基础。 |
| 2 | **CS162: Operating Systems and System Programming** | UC Berkeley | 大学课程 | 高级 | 系统设计基础，理解评测平台的底层系统设计。 |
| 3 | **Full Stack Deep Learning** | FSDL | 在线课程 | 中级 | 全栈深度学习课程，涵盖ML系统的前后端开发。https://fullstackdeeplearning.com/ |

#### 📝 博客与论文

| # | 名称 | 作者/来源 | 类型 | 难度 | 推荐理由 |
|---|------|------|------|------|----------|
| 1 | **How to Run LLM Evals That Predict Production** | BestAIWeb | 博客 | 中级 | 实战指南，对比lm-eval-harness、HELM和OpenCompass的适用场景。https://www.bestaiweb.ai/how-to-benchmark-llms-with-lm-evaluation-harness-helm-and-opencompass-in-2026/ |
| 2 | **AI Safety Eval Frameworks Compared** | AIPromptsHub | 博客 | 高级 | 对比HELM、Inspect、OpenAI Evals等8个评测框架的优劣。https://aipromptshub.co/safety/ai-safety-evals-comparison-2026 |
| 3 | **LLM Evaluation: The Quality Engineering Layer** | LinkedIn | 博客 | 中级 | 评测作为产品质量工程层的实践方法论。 |
| 4 | **Eval frameworks for LLMs: DeepEval, OpenAI Evals, Promptfoo, Inspect** | Fairlane Systems | 博客 | 中级 | 主流评测框架的对比选型指南。https://fairlane.systems/en/wissen/eval-frameworks-fuer-llms |

#### 🔧 开源项目/代码库

| # | 名称 | 来源 | 类型 | 难度 | 推荐理由 |
|---|------|------|------|------|----------|
| 1 | **OpenCompass (全栈部分)** | 上海AI Lab | 开源框架 | 高级 | 学习企业级评测平台的完整架构，包括前端可视化、后端调度、数据管理。GitHub: https://github.com/open-compass/opencompass |
| 2 | **MLflow** | LF AI & Data Foundation | 开源框架 | 中级 | ML生命周期管理平台，学习实验追踪和模型管理。GitHub: https://github.com/mlflow/mlflow |
| 3 | **LangFuse** | LangFuse | 开源框架 | 中级 | LLM工程平台，学习评测结果追踪、Prompt版本管理和生产可观测性。GitHub: https://github.com/langfuse/langfuse |
| 4 | **Phoenix (Arize AI)** | Arize AI | 开源框架 | 中级 | LLM可观测性平台，学习评测结果可视化和分析。GitHub: https://github.com/Arize-ai/phoenix |
| 5 | **Docker** | Docker Inc | 开源工具 | 入门 | 容器化技术，Agent评测沙箱环境的核心。学习Docker构建安全隔离的评测环境。 |
| 6 | **Airflow** | Apache | 开源框架 | 中级 | 分布式任务调度框架，学习大规模评测任务编排。GitHub: https://github.com/apache/airflow |

---

### 4.6 编程与工程能力 — 学习资料

#### 📚 书籍

| # | 名称 | 作者/来源 | 类型 | 难度 | 推荐理由 |
|---|------|------|------|------|----------|
| 1 | **Fluent Python** | Luciano Ramalho (O'Reilly) | 书籍 | 中级 | Python进阶圣经，写出高质量Python代码。评测系统开发必备。 |
| 2 | **Python for Data Analysis** | Wes McKinney (O'Reilly) | 书籍 | 入门 | Pandas作者编写，数据处理和分析的基础。评测数据分析必备。 |
| 3 | **Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow** | Aurélien Géron (O'Reilly) | 书籍 | 入门→中级 | ML实践圣经，包含模型评估章节。 |

#### 🎓 课程

| # | 名称 | 来源 | 类型 | 难度 | 推荐理由 |
|---|------|------|------|------|----------|
| 1 | **FastAPI Documentation** | Sebastián Ramírez | 官方文档/教程 | 入门 | 评测系统后端API开发的首选框架，文档质量极高。https://fastapi.tiangolo.com/ |
| 2 | **React Official Tutorial** | Meta/React Team | 官方教程 | 入门 | 评测系统前端开发，构建可视化界面。https://react.dev/learn |
| 3 | **Docker & Kubernetes: The Practical Guide** | Academind (YouTube) | 在线课程 | 入门→中级 | 容器化部署，评测环境管理必备。 |

#### 🔧 开源项目/代码库

| # | 名称 | 来源 | 类型 | 难度 | 推荐理由 |
|---|------|------|------|------|----------|
| 1 | **FastAPI** | Tiangolo | 开源框架 | 入门 | 现代Python Web框架，评测系统后端开发首选。GitHub: https://github.com/tiangolo/fastapi |
| 2 | **Plotly / Dash** | Plotly | 开源框架 | 中级 | 交互式数据可视化，评测结果可视化展示。GitHub: https://github.com/plotly/dash |
| 3 | **Celery** | Celery Project | 开源框架 | 中级 | Python分布式任务队列，评测任务调度。GitHub: https://github.com/celery/celery |
| 4 | **Redis** | Redis Ltd | 开源工具 | 入门 | 缓存和消息队列，评测系统的基础设施。 |

---

### 4.7 大模型基础理论 — 学习资料

#### 📚 书籍

| # | 名称 | 作者/来源 | 类型 | 难度 | 推荐理由 |
|---|------|------|------|------|----------|
| 1 | **Deep Learning** | Ian Goodfellow, Yoshua Bengio, Aaron Courville (MIT Press) | 书籍 | 中级 | 深度学习圣经，理解神经网络基础。在线免费: https://www.deeplearningbook.org/ |
| 2 | **Attention Is All You Need** | Vaswani et al. (Google) | 论文 | 中级 | Transformer架构的原始论文，理解LLM架构基础。https://arxiv.org/abs/1706.03762 |
| 3 | **Natural Language Processing with Transformers** | Hugging Face团队 | 书籍 | 中级 | Hugging Face团队出品，实践导向的Transformer NLP指南。 |

#### 🎓 课程

| # | 名称 | 来源 | 类型 | 难度 | 推荐理由 |
|---|------|------|------|------|----------|
| 1 | **CS224N: NLP with Deep Learning** | Stanford | 大学课程 | 中级 | **必学基础课**，NLP和深度学习的经典课程。https://web.stanford.edu/class/cs224n/ |
| 2 | **CS324: Large Language Models** | Stanford | 大学课程 | 高级 | **必学进阶课**，专门针对LLM，涵盖训练、评测、安全。https://stanford-cs324.github.io/ |
| 3 | **Hugging Face NLP Course** | Hugging Face | 在线课程（免费） | 入门 | Hugging Face官方NLP课程，实践导向。https://huggingface.co/learn/nlp-course |
| 4 | **Hugging Face LLM Course** | Hugging Face | 在线课程（免费） | 中级 | Hugging Face LLM专项课程。https://huggingface.co/learn/llm-course |
| 5 | **Let's build GPT: from scratch, in code** | Andrej Karpathy (YouTube) | 视频教程 | 中级 | Karpathy从零实现GPT，深入理解Transformer。https://www.youtube.com/watch?v=kCc8FmEb1nY |

#### 📝 关键论文

| # | 名称 | 作者/来源 | 类型 | 难度 | 推荐理由 |
|---|------|------|------|------|----------|
| 1 | **GPT-3: Language Models are Few-Shot Learners** | Tom Brown et al. (OpenAI) | 论文 | 中级 | GPT-3论文，理解大规模语言模型的能力和评测。https://arxiv.org/abs/2005.14165 |
| 2 | **InstructGPT: Training language models to follow instructions with human feedback** | Long Ouyang et al. (OpenAI) | 论文 | 高级 | RLHF的奠基性论文，理解对齐训练如何影响模型评测表现。https://arxiv.org/abs/2203.02155 |
| 3 | **LLaMA: Open and Efficient Foundation Language Models** | Hugo Touvron et al. (Meta) | 论文 | 中级 | 开源大模型的里程碑，理解模型架构和训练数据。https://arxiv.org/abs/2302.13971 |
| 4 | **Constitutional AI: Harmlessness from AI Feedback** | Anthropic | 论文 | 高级 | Anthropic的宪法AI方法，理解安全对齐评测。https://arxiv.org/abs/2212.08073 |

#### 🔧 开源项目/代码库

| # | 名称 | 来源 | 类型 | 难度 | 推荐理由 |
|---|------|------|------|------|----------|
| 1 | **Hugging Face Transformers** | Hugging Face | 开源框架 | 入门→中级 | **必学**，最广泛使用的Transformer模型库。理解模型加载、推理和评测的完整流程。GitHub: https://github.com/huggingface/transformers |
| 2 | **vLLM** | UC Berkeley | 开源框架 | 高级 | 高性能LLM推理引擎，学习大规模推理优化。GitHub: https://github.com/vllm-project/vllm |
| 3 | **LLMs-from-scratch** | Sebastian Raschka | 开源项目 | 中级 | 从零实现LLM，深入理解模型内部机制。GitHub: https://github.com/rasbt/LLMs-from-scratch |

---

## 5. 学习路径建议

### 5.1 总体学习路线图

```
阶段0 (前置)        阶段1 (基础)         阶段2 (进阶)         阶段3 (高级)         阶段4 (精通)
    │                   │                   │                   │                   │
    ▼                   ▼                   ▼                   ▼                   ▼
┌────────┐      ┌────────────┐      ┌────────────┐      ┌────────────┐      ┌────────────┐
│ 大模型  │      │ 评测基础与  │      │ Prompt工程 │      │ 评测体系   │      │ 前沿创新与 │
│ 基础理论│  →   │ Benchmark  │  →   │ 与Rubrics  │  →   │ 设计与系统 │  →   │ 企业级实践 │
│        │      │            │      │ 设计       │      │ 架构       │      │            │
└────────┘      └────────────┘      └────────────┘      └────────────┘      └────────────┘
  约4-8周           约3-4周             约3-4周             约4-6周             约4-8周
```

### 5.2 分阶段详细学习计划

#### 阶段0：大模型基础理论（前置知识，约4-8周）

**目标**：建立对LLM的基本理解，为学习评测方法论打下基础。

| 周次 | 学习内容 | 学习资料 | 实践任务 |
|------|----------|----------|----------|
| 1-2 | 深度学习基础 | 《Deep Learning》前5章 + Karpathy "Let's build GPT"视频 | 用PyTorch实现简单的神经网络 |
| 3-4 | NLP与Transformer | CS224N课程前8讲 + "Attention Is All You Need"论文 | 用Hugging Face Transformers加载并运行GPT模型 |
| 5-6 | 大语言模型基础 | CS324课程 + GPT-3论文 + LLaMA论文 | 阅读并总结3篇关键论文 |
| 7-8 | RLHF与对齐 | InstructGPT论文 + Constitutional AI论文 | 理解SFT和RLHF的训练流程 |

**阶段产出**：能解释Transformer架构、预训练流程、RLHF对齐方法，理解LLM的能力来源和局限性。

---

#### 阶段1：大模型评测基础与Benchmark（约3-4周）

**目标**：熟悉业界主流Benchmark和评测框架，能使用框架进行基本评测。

| 周次 | 学习内容 | 学习资料 | 实践任务 |
|------|----------|----------|----------|
| 1 | 评测基本概念 | HELM论文 + "LLM Benchmarks Explained"博客 + Jurafsky SLP3相关章节 | 撰写一篇大模型评测综述笔记 |
| 2 | 主流Benchmark详解 | 阅读MMLU/HumanEval/GSM8K/GPQA/IFEval/MT-Bench原始论文 | 制作Benchmark对比表格（设计理念/数据特点/局限性） |
| 3 | 评测框架实践 | lm-eval-harness文档 + OpenCompass文档 + HELM文档 | 使用lm-eval-harness对开源模型（如Llama）进行MMLU评测 |
| 4 | 评测框架深入 | OpenCompass源码阅读 + OpenAI Evals实践 | 使用OpenCompass搭建一个自定义评测任务 |

**阶段产出**：能使用至少2个评测框架对LLM进行标准Benchmark评测，理解各Benchmark的设计理念和局限性。

---

#### 阶段2：Prompt工程与Rubrics设计（约3-4周）

**目标**：精通Prompt设计方法，能设计多级评分Rubrics，掌握LLM-as-Judge技术。

| 周次 | 学习内容 | 学习资料 | 实践任务 |
|------|----------|----------|----------|
| 1 | Prompt设计基础 | DeepLearning.AI Prompt Engineering课程 + Prompt Engineering Guide | 设计并测试Zero-shot/Few-shot/CoT Prompt |
| 2 | 高级Prompt技术 | CoT/Self-Consistency/ToT/ReAct原始论文 | 实现Self-Consistency评测流程 |
| 3 | Rubrics设计 | "LLM Evaluation Rubric"博客 + HuggingFace Evaluation Guidebook | 为"代码生成"和"创意写作"两个维度设计多级Rubrics |
| 4 | LLM-as-Judge | "Judging LLM-as-a-Judge"论文 + "A Survey on LLM-as-a-Judge"综述 | 使用GPT-4作为裁判模型进行Pairwise评测，分析位置偏差 |

**阶段产出**：能独立设计评测Prompt和评分Rubrics，能使用LLM-as-Judge进行主观评测并识别系统性偏差。

---

#### 阶段3：评测归因分析与Agent评测（约4-6周）

**目标**：掌握评测归因分析方法论，能设计Agent评测方案，建立"评测-归因-改进"闭环。

| 周次 | 学习内容 | 学习资料 | 实践任务 |
|------|----------|----------|----------|
| 1 | 归因分析基础 | "Causality: Models, Reasoning, and Inference"前3章 + DoWhy文档 | 对一次评测结果进行多维度归因分析 |
| 2 | 系统化归因方法 | 数据切片分析实践 + 对比实验设计 | 设计并执行一个对比实验，分析模型能力短板 |
| 3 | Agent基础 | LangChain文档 + DeepLearning.AI "AI Agents in LangChain"课程 | 使用LangChain搭建一个简单Agent |
| 4 | Agent评测Benchmark | SWE-bench/WebArena/ToolBench/AgentBench论文 | 分析各Agent Benchmark的评测维度和方法 |
| 5 | Agent评测方案设计 | AgentBench论文 + AdaRubric论文 | 设计一个工具调用Agent的评测方案 |
| 6 | Agent-as-Judge | "Agent-as-Judge"相关论文 + Multi-Agent Judge论文 | 实现一个简单的Agent-as-Judge评测流程 |

**阶段产出**：能进行系统化评测归因分析，能设计完整的Agent评测方案，理解Agent-as-Judge方法。

---

#### 阶段4：评测体系设计与系统架构（约4-6周）

**目标**：能独立设计评测体系，搭建完整的评测系统，掌握企业级评测平台架构。

| 周次 | 学习内容 | 学习资料 | 实践任务 |
|------|----------|----------|----------|
| 1 | 评测体系设计 | OpenCompass架构分析 + HELM方法论 | 设计一个面向特定业务场景的完整评测方案 |
| 2 | 评测系统后端 | FastAPI文档 + Celery文档 + "Designing Data-Intensive Applications" | 搭建评测系统后端API（数据管理+模型接入+评测执行） |
| 3 | 评测系统前端 | React教程 + Plotly/Dash文档 | 搭建评测结果可视化界面 |
| 4 | Agent评测环境 | Docker文档 + SWE-bench环境搭建 | 搭建一个Docker化的Agent评测沙箱 |
| 5 | 自动化Pipeline | Airflow文档 + MLflow文档 | 搭建端到端自动化评测Pipeline |
| 6 | 评测方法论创新 | LiveBench + 自适应评测论文 + 前沿评测方法论 | 撰写一份评测方法论创新提案 |

**阶段产出**：能独立设计评测体系方案，搭建包含数据管理、模型接入、评测执行、裁判服务、结果分析和报告生成的完整评测系统。

---

#### 阶段5：前沿创新与企业级实践（约4-8周，持续学习）

**目标**：跟踪评测前沿，主导企业级评测体系建设，推动方法论创新。

| 持续学习内容 | 学习资料 | 实践方向 |
|-------------|----------|----------|
| 自适应评测方案 | 前沿论文 + LiveBench | 设计自适应评测算法原型 |
| 多模态评测 | 多模态Benchmark论文 | 搭建图文多模态评测原型 |
| 因果推断在归因中的应用 | DoWhy/CausalPy框架 | 实现因果归因分析原型 |
| 企业级评测平台架构 | "Designing Data-Intensive Applications" + OpenCompass架构 | 设计千万级评测任务调度方案 |
| 评测数据资产全生命周期管理 | 数据管理最佳实践 | 设计评测数据管理规范 |
| Agent评测前沿 | 行为可解释性评测 + 安全性评测论文 | 设计复杂Agent评测框架 |

### 5.3 学习优先级排序

根据岗位岗位能力要求的掌握程度递进关系，建议按以下优先级学习：

| 优先级 | 知识领域 | 理由 |
|--------|----------|------|
| P0（最高） | 大模型评测基础与Benchmark | 一切评测工作的基础，1-进阶级能力要求 |
| P0 | Prompt工程与Rubrics设计 | 评测核心技能，贯穿1-专家级要求 |
| P1 | 评测数据构建与裁判模型应用 | 关键技能项，2-精通级核心要求 |
| P1 | LLM-as-Judge技术 | 主观评测的核心方法，2-精通级要求 |
| P2 | 评测归因分析 | 2-精通级要求，体现深度分析能力 |
| P2 | Agent评测 | 1-精通级要求，是岗位特色方向 |
| P3 | 评测系统架构 | 3-专家级要求，体现系统工程能力 |
| P3 | 评测方法论创新 | 专家级要求，体现前沿研究能力 |

### 5.4 关键里程碑检查清单

完成每个阶段后，用以下检查清单验证学习成果：

#### 阶段0完成检查 ☐
- [ ] 能解释Transformer的Self-Attention机制
- [ ] 能描述预训练→SFT→RLHF的完整训练流程
- [ ] 能使用Hugging Face Transformers加载和运行模型
- [ ] 理解LLM的能力来源和主要局限性

#### 阶段1完成检查 ☐
- [ ] 能列举至少10个主流Benchmark及其评测维度
- [ ] 能说明MMLU/HumanEval/GSM8K的设计理念和局限性
- [ ] 能使用lm-eval-harness执行标准Benchmark评测
- [ ] 能使用OpenCompass搭建自定义评测任务
- [ ] 了解评测数据污染问题和防泄露策略

#### 阶段2完成检查 ☐
- [ ] 能设计Zero-shot/Few-shot/CoT/Self-Consistency Prompt
- [ ] 能为不同评测维度设计多级评分Rubrics
- [ ] 能使用GPT-4/Claude作为裁判模型进行Pairwise评测
- [ ] 能识别并消除Position Bias/Verbosity Bias/Self-Enhancement Bias
- [ ] 能搭建简单的自动化评测分析Pipeline

#### 阶段3完成检查 ☐
- [ ] 能进行多维度评测归因分析
- [ ] 能设计数据切片分析方案
- [ ] 能使用LangChain搭建简单Agent
- [ ] 能设计Agent评测方案（环境、指标、数据）
- [ ] 能建立"评测-归因-改进-验证"闭环
- [ ] 了解Agent-as-Judge方法

#### 阶段4完成检查 ☐
- [ ] 能独立设计完整的评测体系方案
- [ ] 能搭建包含前后端的评测系统
- [ ] 能搭建Agent评测沙箱环境
- [ ] 能设计自动化评测Pipeline
- [ ] 能进行评测方法论创新思考

---

## 附录：全部学习资料速查表

### 按类型分类

#### 📚 书籍（共17本）
| # | 书名 | 作者 | 领域 |
|---|------|------|------|
| 1 | Speech and Language Processing (3rd ed.) | Jurafsky & Martin | 评测基础 |
| 2 | Build a Large Language Model (From Scratch) | Sebastian Raschka | 大模型基础 |
| 3 | Natural Language Processing with Transformers | Hugging Face团队 | 大模型基础 |
| 4 | Evaluating Machine Learning Models | Alice Zheng | 评测基础 |
| 5 | Prompt Engineering Guide | DAIR.AI | Prompt工程 |
| 6 | The Art of Prompt Engineering | Nathan Hunter | Prompt工程 |
| 7 | Prompt Engineering for LLMs | O'Reilly | Prompt工程 |
| 8 | Causal Inference in Statistics: A Primer | Judea Pearl | 归因分析 |
| 9 | The Book of Why | Judea Pearl | 归因分析 |
| 10 | Causality: Models, Reasoning, and Inference | Judea Pearl | 归因分析 |
| 11 | Dataset Shift in Machine Learning | MIT Press | 数据构建 |
| 12 | Human-in-the-Loop Machine Learning | Robert Munro | 数据构建 |
| 13 | Designing Data-Intensive Applications | Martin Kleppmann | 系统架构 |
| 14 | System Design Interview | Alex Xu | 系统架构 |
| 15 | Deep Learning | Goodfellow et al. | 大模型基础 |
| 16 | Fluent Python | Luciano Ramalho | 编程能力 |
| 17 | Hands-On ML with Scikit-Learn, Keras, TF | Aurélien Géron | 编程能力 |

#### 🎓 课程（共15门）
| # | 课程名 | 来源 | 领域 |
|---|--------|------|------|
| 1 | CS224N: NLP with Deep Learning | Stanford | 大模型基础 |
| 2 | CS324: Large Language Models | Stanford | 大模型基础 |
| 3 | CS25: Transformers United | Stanford | 大模型基础 |
| 4 | Practical Deep Learning for Coders | Fast.ai | 大模型基础 |
| 5 | ChatGPT Prompt Engineering for Developers | DeepLearning.AI | Prompt工程 |
| 6 | AI Agents in LangChain | DeepLearning.AI | Agent评测 |
| 7 | Functions, Tools and Agents with LangChain | DeepLearning.AI | Agent评测 |
| 8 | Causal Inference | Brady Neal | 归因分析 |
| 9 | Data-Centric AI | MIT/Andrew Ng | 数据构建 |
| 10 | Hugging Face NLP Course | Hugging Face | 大模型基础 |
| 11 | Hugging Face LLM Course | Hugging Face | 大模型基础 |
| 12 | MIT 6.5840: Distributed Systems | MIT | 系统架构 |
| 13 | Full Stack Deep Learning | FSDL | 系统架构 |
| 14 | LangChain Academy | LangChain | Prompt工程 |
| 15 | Let's build GPT (Karpathy) | YouTube | 大模型基础 |

#### 🔧 开源项目（共25个）
| # | 项目名 | 来源 | 领域 |
|---|--------|------|------|
| 1 | lm-eval-harness | EleutherAI | 评测框架 |
| 2 | OpenCompass | 上海AI Lab | 评测框架 |
| 3 | HELM | Stanford CRFM | 评测框架 |
| 4 | OpenAI Evals | OpenAI | 评测框架 |
| 5 | FastChat | LMSYS | 评测框架 |
| 6 | Promptfoo | promptfoo | 评测工具 |
| 7 | DeepEval | Confident AI | 评测框架 |
| 8 | RAGAS | explodinggradients | RAG评测 |
| 9 | TruLens | TruEra | 评测追踪 |
| 10 | LangChain | LangChain AI | Agent框架 |
| 11 | AutoGen | Microsoft | Agent框架 |
| 12 | CrewAI | CrewAI Inc | Agent框架 |
| 13 | SWE-bench | Princeton NLP | Agent评测 |
| 14 | WebArena | CMU | Agent评测 |
| 15 | DoWhy | py-why | 因果推断 |
| 16 | Hugging Face Transformers | Hugging Face | 模型库 |
| 17 | vLLM | UC Berkeley | 推理引擎 |
| 18 | LLMs-from-scratch | Sebastian Raschka | 教学 |
| 19 | llm-decontaminator | LMSYS | 数据去污染 |
| 20 | Label Studio | HumanSignal | 数据标注 |
| 21 | Cleanlab | Cleanlab | 数据质量 |
| 22 | MLflow | LF AI | ML管理 |
| 23 | LangFuse | LangFuse | LLM工程 |
| 24 | Phoenix | Arize AI | 可观测性 |
| 25 | Airflow | Apache | 任务调度 |

#### 📝 关键论文（共20篇）
| # | 论文名 | 来源 | 领域 |
|---|--------|------|------|
| 1 | Attention Is All You Need | Google | Transformer |
| 2 | GPT-3: Language Models are Few-Shot Learners | OpenAI | LLM基础 |
| 3 | InstructGPT (RLHF) | OpenAI | 对齐训练 |
| 4 | LLaMA | Meta | 开源LLM |
| 5 | Constitutional AI | Anthropic | 安全对齐 |
| 6 | Chain-of-Thought Prompting | Google | Prompt工程 |
| 7 | Self-Consistency | Google | Prompt工程 |
| 8 | Tree of Thoughts | Princeton | Prompt工程 |
| 9 | ReAct | Princeton | Agent Prompt |
| 10 | Judging LLM-as-a-Judge (MT-Bench) | NeurIPS 2023 | LLM裁判 |
| 11 | A Survey on LLM-as-a-Judge | arxiv 2024 | LLM裁判 |
| 12 | Quantifying Biases in LLM-as-a-Judge | arxiv 2024 | 裁判偏差 |
| 13 | IFEval | Google | 指令遵循评测 |
| 14 | SWE-bench | Princeton | Agent评测 |
| 15 | WebArena | CMU | Agent评测 |
| 16 | AgentBench | Tsinghua | Agent评测 |
| 17 | GAIA | Meta | Agent评测 |
| 18 | Lost in the Middle | Stanford | 长上下文 |
| 19 | Rethinking Benchmark Contamination | LMSYS | 数据污染 |
| 20 | AdaRubric | arxiv | 自适应Rubrics |

---

### 按学习阶段分类的资料包

#### 📦 阶段0资料包：大模型基础理论
- CS224N (Stanford) + CS324 (Stanford) + Karpathy视频
- 《Deep Learning》+ 《Speech and Language Processing》
- Hugging Face Transformers + LLMs-from-scratch
- 关键论文: Attention Is All You Need / GPT-3 / InstructGPT / LLaMA

#### 📦 阶段1资料包：评测基础与Benchmark
- lm-eval-harness + OpenCompass + HELM（三个核心框架）
- "LLM Benchmarks Explained"博客 + 各Benchmark原始论文
- llm-decontaminator（数据去污染工具）
- LiveBench（无污染评测方案）

#### 📦 阶段2资料包：Prompt工程与Rubrics设计
- DeepLearning.AI Prompt Engineering课程 + Prompt Engineering Guide
- CoT/Self-Consistency/ToT/ReAct原始论文
- "Judging LLM-as-a-Judge"论文 + "A Survey on LLM-as-a-Judge"综述
- Promptfoo + DeepEval（实践工具）

#### 📦 阶段3资料包：归因分析与Agent评测
- LangChain + AutoGen + CrewAI（Agent框架）
- SWE-bench + WebArena + AgentBench（Agent Benchmark）
- DoWhy（因果推断）+ Judea Pearl书籍
- "Agent-as-Judge"相关论文

#### 📦 阶段4资料包：评测体系与系统架构
- OpenCompass源码（学习企业级架构）
- FastAPI + React + Docker + Celery + Airflow（系统开发）
- 《Designing Data-Intensive Applications》
- MLflow + LangFuse + Phoenix（评测平台工具链）

---

> **最后提醒**：本学习计划覆盖了AI模型评测岗位从入门级到精通级的全部知识点。建议按照阶段顺序系统学习，每个阶段完成后用检查清单验证学习成果。在学习过程中，**实践是最重要的**——每学一个知识点，都要动手实现或运行相关代码。建议重点关注以下三个"必学"开源项目：**lm-eval-harness**（评测框架基础）、**OpenCompass**（企业级评测平台）、**LangChain**（Agent框架），它们分别代表了评测框架、评测系统和Agent评测三个核心方向的最佳实践。
