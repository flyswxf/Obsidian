## 目录

1. [[#1. 岗位概述|岗位概述]]
2. [[#2. 知识体系全景图（知识树）|知识体系全景图（知识树）]]
3. [[#3. 分领域详细知识点梳理与学习资料规划|分领域详细知识点梳理与学习资料规划]]
   - 3.1 [[#3.1 SFT数据合成、配比与多能力协同|SFT 数据合成、配比与多能力协同]]
   - 3.2 [[#3.2 奖励模型、奖励系统|奖励模型、奖励系统]]
   - 3.3 [[#3.3 强化学习训练算法|强化学习训练算法]]
   - 3.4 [[#3.4 训练数据处理、数据清洗与数据合成技术实践|训练数据处理、数据清洗与数据合成技术实践]]
   - 3.5 [[#3.5 MoE模型RL训练一致性和训练稳定性分析|MoE模型RL训练一致性和训练稳定性分析]]
   - 3.6 [[#3.6 多专项模型融合、蒸馏策略|多专项模型融合、蒸馏策略]]
4. [[#4. 学习路径建议|学习路径建议]]
5. [[#5. 推荐学习时间线|推荐学习时间线]]

---

## 1. 岗位概述

### 1.1 岗位定位

本岗位聚焦于大语言模型（LLM）的**后训练（Post-training）**阶段，核心使命是通过**监督微调（SFT）**和**强化学习（RL）**技术，将预训练基座模型转化为具备推理、编码、Agent、多模态等能力的实用模型。岗位要求候选人深入理解从数据构建到算法设计再到训练工程的全链路技术。

### 1.2 核心能力要求

| 维度 | 能力领域 | 核心关键词 |
|------|----------|------------|
| 专业知识1 | SFT数据合成与配比 | 数据合成、Data Mixture、Co-training、Alignment |
| 专业知识2 | 奖励模型与奖励系统 | RLHF/RLAIF、PRM、稀疏奖励、Reward Hacking |
| 专业知识3 | 强化学习训练算法 | PPO、DPO、GRPO、DAPO、KL惩罚、分布式RL |
| 关键技能1 | 数据处理实践 | Self-QA、Self-Instruct、数据清洗流水线 |
| 关键技能2 | MoE RL训练稳定性 | IcePop、Routing Replay、训推一致性 |
| 关键技能3 | 模型融合与蒸馏 | SLERP、TIES、On-policy KD、Multi-task RL |

### 1.3 掌握程度体系

每个知识点分为基础级-专家级，其中：
- **基础级**：基础知识（了解概念、流程）
- **进阶级**：实践能力（能动手操作、理解原理）
- **精通级**：深度掌握（能设计方案、解决复杂问题）
- **专家级**：专家级别（创新性贡献、第一性原理分析）

---

## 2. 知识体系全景图（知识树）

```
AI模型-后训练与强化学习
├── A. SFT数据合成、配比与多能力协同
│   ├── A1. SFT基础理论
│   │   ├── A1.1 Pre-train vs Post-train 数据差异
│   │   ├── A1.2 指令微调（Instruction Tuning）基本流程
│   │   ├── A1.3 多模态SFT（图文对数据）
│   │   └── A1.4 高质量指令数据标准
│   ├── A2. 数据配比与能力平衡
│   │   ├── A2.1 Data Mixture 策略
│   │   ├── A2.2 能力坍缩（Capability Collapse）现象与机制
│   │   ├── A2.3 General vs Domain 能力平衡
│   │   └── A2.4 动态配比调整策略
│   ├── A3. SFT与RL协同训练
│   │   ├── A3.1 Co-training 逻辑
│   │   ├── A3.2 SFT为RL预留探索空间
│   │   ├── A3.3 分阶段训练策略
│   │   └── A3.4 Alignment与知识获取的界限
│   └── A4. 针对性数据合成方案
│       ├── A4.1 Reasoning任务数据合成
│       ├── A4.2 Coding任务数据合成
│       ├── A4.3 Agent任务数据合成
│       └── A4.4 自动化数据评测与迭代流水线
│
├── B. 奖励模型、奖励系统
│   ├── B1. RLHF/RLAIF基础
│   │   ├── B1.1 RLHF标准三阶段流程
│   │   ├── B1.2 RLAIF（AI反馈替代人类反馈）
│   │   ├── B1.3 偏好数据收集与标注
│   │   └── B1.4 Bradley-Terry模型与排序偏好转标量
│   ├── B2. 奖励模型训练
│   │   ├── B2.1 RM训练方法（Pointwise/Pairwise/Listwise）
│   │   ├── B2.2 混合奖励函数设计（数学/代码）
│   │   ├── B2.3 奖励模型架构选择
│   │   └── B2.4 奖励模型的工程实现
│   ├── B3. 过程监督与稀疏奖励
│   │   ├── B3.1 过程奖励模型（PRM）
│   │   ├── B3.2 Step-level标注与验证
│   │   ├── B3.3 稀疏奖励问题及解决策略
│   │   └── B3.4 过程监督的成本收益分析
│   └── B4. 高级奖励系统
│       ├── B4.1 Distribution Shift与RM漂移
│       ├── B4.2 Generative RL（生成式奖励模型）
│       ├── B4.3 Ensemble RM（集成奖励模型）
│       ├── B4.4 Iterative RLHF与动态RM更新
│       └── B4.5 多目标奖励平衡（安全 vs 有用）
│
├── C. 强化学习训练算法
│   ├── C1. RL基础理论
│   │   ├── C1.1 策略梯度 vs 价值学习
│   │   ├── C1.2 策略梯度公式推导（REINFORCE）
│   │   ├── C1.3 Actor-Critic架构
│   │   └── C1.4 优势函数（Advantage Function）估计
│   ├── C2. 主流RL算法
│   │   ├── C2.1 PPO（Proximal Policy Optimization）
│   │   ├── C2.2 DPO（Direct Preference Optimization）
│   │   ├── C2.3 GRPO（Group Relative Policy Optimization）
│   │   ├── C2.4 DAPO（Decoupled clip & Dynamic sampling）
│   │   └── C2.5 其他变体（KTO, IPO, ORPO等）
│   ├── C3. KL惩罚与训练稳定性
│   │   ├── C3.1 KL散度的数学表达与物理意义
│   │   ├── C3.2 PPO中的KL惩罚机制
│   │   ├── C3.3 DPO中的隐式KL约束
│   │   └── C3.4 训练不稳定因素及处理
│   ├── C4. 场景化RL设计
│   │   ├── C4.1 Reasoning场景RL
│   │   ├── C4.2 Coding场景RL
│   │   ├── C4.3 Agent场景RL
│   │   └── C4.4 多模态场景RL
│   └── C5. 大规模分布式RL训练
│       ├── C5.1 分布式RL架构设计
│       ├── C5.2 Gradient Norm / Value Loss / Explained Variance诊断
│       ├── C5.3 训练崩塌根因分析
│       └── C5.4 从第一性原理改进算法
│
├── D. 训练数据处理、数据清洗与数据合成技术实践
│   ├── D1. 数据清洗基础
│   │   ├── D1.1 数据清洗流程设计
│   │   ├── D1.2 去重、过滤、质量评估
│   │   └── D1.3 数据污染检测
│   ├── D2. 数据合成方法
│   │   ├── D2.1 Self-Instruct方法
│   │   ├── D2.2 Self-QA方法
│   │   ├── D2.3 Evol-Instruct方法
│   │   └── D2.4 基于强模型的蒸馏合成
│   ├── D3. 高级数据工程
│   │   ├── D3.1 多种合成清洗算法综合运用
│   │   ├── D3.2 数据质量自动评估
│   │   └── D3.3 数据配比自动化搜索
│   └── D4. 创新性数据方法
│       ├── D4.1 创新数据合成算法设计
│       ├── D4.2 创新清洗与配比策略
│       └── D4.3 效果验证与迭代
│
├── E. MoE模型RL训练一致性和训练稳定性分析
│   ├── E1. MoE基础
│   │   ├── E1.1 MoE稀疏激活机制
│   │   ├── E1.2 路由（Routing）机制
│   │   ├── E1.3 Top-k门控与负载均衡
│   │   └── E1.4 MoE在LLM中的应用
│   ├── E2. MoE RL训练特殊问题
│   │   ├── E2.1 训推不一致性（Training-Inference Mismatch）
│   │   ├── E2.2 RL训练中的路由漂移
│   │   └── E2.3 MoE RL常见不稳定问题
│   ├── E3. 稳定性提升算法
│   │   ├── E3.1 IcePop（双面遮蔽）
│   │   ├── E3.2 Routing Replay（路由重放）
│   │   ├── E3.3 Off-policy Masking
│   │   └── E3.4 Importance Sampling
│   └── E4. 高级MoE RL
│       ├── E4.1 Gradient Norm爆炸根因
│       ├── E4.2 Reward Hacking检测与防御
│       ├── E4.3 分布式MoE RL架构
│       └── E4.4 Inference-time Compute优化
│
└── F. 多专项模型融合、蒸馏策略
    ├── F1. 知识蒸馏基础
    │   ├── F1.1 知识蒸馏基本原理
    │   ├── F1.2 软标签与硬标签
    │   ├── F1.3 Logit蒸馏 vs 特征蒸馏
    │   └── F1.4 多教师蒸馏
    ├── F2. 模型融合（Model Merge）
    │   ├── F2.1 SLERP（球面线性插值）
    │   ├── F2.2 TIES-Merging
    │   ├── F2.3 DARE（Drop And REscale）
    │   ├── F2.4 Task Arithmetic
    │   └── F2.5 Model Soup
    ├── F3. 多任务学习与梯度冲突
    │   ├── F3.1 多任务学习中的梯度冲突
    │   ├── F3.2 梯度冲突检测与解决
    │   └── F3.3 多任务RL（Multi-task RL）
    └── F4. On-policy KD与高级策略
        ├── F4.1 On-policy KD原理
        ├── F4.2 On-policy KD缓解Distribution Shift
        ├── F4.3 On-policy KD与RL的关系
        ├── F4.4 在线蒸馏算法变体
        └── F4.5 多专项在线蒸馏适用场景
```

---

## 3. 分领域详细知识点梳理与学习资料规划

---

### 3.1 SFT数据合成、配比与多能力协同

#### 3.1.1 知识点详细梳理

**A1. SFT基础理论**

| 子知识点 | 详细内容 | 对应等级 |
|----------|----------|----------|
| A1.1 Pre-train vs Post-train 数据差异 | 预训练使用海量无标注文本（Web爬取、书籍、代码），后训练使用高质量指令-响应对；预训练目标是next-token prediction，后训练目标是行为对齐 | 0-基础级 |
| A1.2 指令微调基本流程 | 数据格式化（Alpaca格式/ChatML格式）→ 模板设计 → Loss Masking（仅对response计算loss）→ 训练超参选择 | 基础级 |
| A1.3 多模态SFT | 图文对数据构建、视觉编码器与LLM对齐、多模态指令数据格式、跨模态对齐训练 | 基础级 |
| A1.4 高质量指令数据标准 | 指令多样性、响应准确性、格式规范性、难度分布、领域覆盖度 | 基础级 |

**A2. 数据配比与能力平衡**

| 子知识点 | 详细内容 | 对应等级 |
|----------|----------|----------|
| A2.1 Data Mixture策略 | 不同领域/任务数据的混合比例设计；基于领域权重、任务类型、难度分层的配比方案 | 进阶级 |
| A2.2 能力坍缩现象 | 过多某类数据导致其他能力退化（如过多代码数据导致对话能力下降）；灾难性遗忘的数学机制 | 进阶级 |
| A2.3 General vs Domain平衡 | 通用能力与专业领域能力的权衡；如何通过数据配比和训练策略同时保持两者 | 进阶级 |
| A2.4 动态配比调整 | 训练过程中动态调整数据配比；基于模型弱项反馈的自适应配比 | 精通级 |

**A3. SFT与RL协同训练**

| 子知识点 | 详细内容 | 对应等级 |
|----------|----------|----------|
| A3.1 Co-training逻辑 | SFT和RL不是串行的独立阶段，而是需要协同设计；SFT数据需要为RL阶段奠定基础 | 精通级 |
| A3.2 SFT为RL预留探索空间 | SFT不应过度拟合，需保留策略多样性以供RL探索；温度参数、数据多样性对RL效果的影响 | 精通级 |
| A3.3 分阶段训练策略 | SFT → RLHF → 迭代RL的多阶段设计；各阶段目标与数据需求 | 精通级 |
| A3.4 Alignment与知识获取界限 | 区分"教会模型新知识"与"激发模型已有能力"；何时该用SFT注入知识，何时该用RL激发能力 | 专家级 |

**A4. 针对性数据合成方案**

| 子知识点 | 详细内容 | 对应等级 |
|----------|----------|----------|
| A4.1 Reasoning数据合成 | 链式思维（CoT）数据合成；数学推理、逻辑推理数据构建；思维链质量验证 | 精通级 |
| A4.2 Coding数据合成 | 代码生成、代码补全、代码审查数据；多语言代码覆盖；测试用例驱动的代码数据 | 精通级 |
| A4.3 Agent数据合成 | 工具调用、多步规划、环境交互数据；ReAct格式数据构建 | 精通级 |
| A4.4 自动化数据评测与迭代 | 基于模型弱项自动合成迭代数据；自动化评测流水线设计；数据质量闭环 | 专家级 |

#### 3.1.2 学习资料规划

##### 📚 书籍

| 序号 | 书名 | 作者/来源 | 类型 | 难度 | 推荐理由 |
|------|------|-----------|------|------|----------|
| 1 | **The RLHF Book: Reinforcement Learning from Human Feedback, Alignment, and Post-Training LLMs** | Nathan Lambert (Manning Publications, 2026) | 书籍 | 中-高 | 目前最系统的RLHF与后训练书籍，覆盖SFT→RM→RL全流程，作者为Allen AI研究员，有丰富实战经验。在线版：https://rlhfbook.com |
| 2 | **LLM Engineer's Handbook** | Paul Iusztin, Maxime Labonne (Packt, 2024) | 书籍 | 中 | 第5章详细讲解SFT实践，包含数据准备、模板设计、训练代码，适合工程师上手 |
| 3 | **Hands-On Large Language Models** | Jay Alammar, Maarten Grootendorst (O'Reilly, 2024) | 书籍 | 中 | 涵盖LLM微调、指令调优、数据处理的实践指南，图文并茂 |

##### 🎓 大学课程

| 序号 | 课程名 | 来源 | 类型 | 难度 | 推荐理由 |
|------|--------|------|------|------|----------|
| 1 | **CS224N: Natural Language Processing with Deep Learning** | Stanford University | 课程 | 中-高 | NLP领域标杆课程，涵盖语言模型基础、微调技术。网站：https://web.stanford.edu/class/cs224n/ |
| 2 | **CS336: Language Modeling from Scratch** | Stanford University | 课程 | 高 | Stanford最新课程，从零构建LLM，包含后训练章节（Lecture 15: Alignment — SFT/RLHF）。课程主页：https://stanford-cs336.github.io/ |
| 3 | **Post-training of LLMs** | DeepLearning.AI | 课程 | 中 | Andrew Ng团队出品，系统讲解SFT、DPO、Online RL三种后训练方法。地址：https://www.deeplearning.ai/courses/post-training-of-llms |
| 4 | **LLM Fine-Tuning and Customization Training** | Coursera (DeepLearning.AI) | 课程 | 中 | 覆盖数据准备、超参调优、评估策略，以及LLM框架实践 |

##### 📝 博客与技术文章

| 序号 | 文章名 | 来源 | 类型 | 难度 | 推荐理由 |
|------|--------|------|------|------|----------|
| 1 | **Instruction Tuning: SFT Data, Formatting & Best Practices** | neelmishra.github.io | 博客 | 中 | 深入讲解指令微调的数据格式（ChatML、Llama模板）、数据质量策略和TRL SFTTrainer代码 |
| 2 | **The Ultimate Guide to Fine-Tuning LLMs** | arXiv:2408.13296 | 论文/综述 | 高 | 从基础到突破的全面综述，覆盖SFT技术全貌、最佳实践和挑战 |
| 3 | **Unveiling the Secret Recipe: A Guide For Supervised Fine-Tuning** | arXiv:2412.13337 | 论文 | 中-高 | 系统研究SFT的最佳实践，数据配比、学习率、epoch等超参影响 |
| 4 | **Hugging Face LLM Course - Chapter 11: Supervised Fine-Tuning** | Hugging Face | 教程 | 中 | HF官方课程，SFT章节含完整代码示例。地址：https://huggingface.co/learn/llm-course/en/chapter11/1 |

##### 💻 开源项目/代码库

| 序号 | 项目名 | 来源 | 类型 | 难度 | 推荐理由 |
|------|--------|------|------|------|----------|
| 1 | **TRL (Transformer Reinforcement Learning)** | Hugging Face | 代码库 | 中 | HF官方RLHF/SFT训练库，支持SFTTrainer、DPOTrainer、PPOTrainer、GRPOTrainer。GitHub: https://github.com/huggingface/trl |
| 2 | **Axolotl** | Open Access AI Collective | 代码库 | 中 | 高度可配置的LLM微调工具，支持多种SFT数据格式和训练策略。GitHub: https://github.com/OpenAccess-AI-Collective/axolotl |
| 3 | **LLaMA-Factory** | hiyouga | 代码库 | 中-低 | 中文社区最流行的LLM微调框架，支持SFT/RLHF/DPO，文档完善。GitHub: https://github.com/hiyouga/LLaMA-Factory |
| 4 | **post_training_llms** | YanCotta | 代码库 | 中 | DeepLearning.AI课程配套代码，展示不同后训练方法实现。GitHub: https://github.com/YanCotta/post_training_llms |

---

### 3.2 奖励模型、奖励系统

#### 3.2.1 知识点详细梳理

**B1. RLHF/RLAIF基础**

| 子知识点 | 详细内容 | 对应等级 |
|----------|----------|----------|
| B1.1 RLHF三阶段流程 | ① SFT模型训练 ② 奖励模型训练 ③ RL优化（PPO等）；各阶段的目标函数与数据需求 | 基础级 |
| B1.2 RLAIF | 用AI模型替代人类标注偏好数据；Constitutional AI方法；AI标注的质量与成本权衡 | 基础级 |
| B1.3 偏好数据收集 | 人类标注者比较模型输出对（A vs B）；标注一致性、标注者偏差处理 | 基础级 |
| B1.4 Bradley-Terry模型 | 将排序偏好转化为标量分数的数学基础；偏好概率 = σ(r(x,y₁) - r(x,y₂))；Pairwise Loss推导 | 基础级 |

**B2. 奖励模型训练**

| 子知识点 | 详细内容 | 对应等级 |
|----------|----------|----------|
| B2.1 RM训练方法 | Pointwise（直接打分）、Pairwise（对比学习）、Listwise（列表排序）；不同方法的优缺点 | 1-进阶级 |
| B2.2 混合奖励函数设计 | 数学任务（正确性验证+步骤评分）、代码任务（编译通过+测试通过+代码质量）；多信号融合 | 进阶级 |
| B2.3 RM架构选择 | 基于SFT模型初始化、分类头 vs 生成式RM；不同模型规模对RM质量的影响 | 进阶级 |
| B2.4 RM工程实现 | RM训练的数据准备、训练超参、评估指标（准确率、一致性）；大规模RM训练的工程挑战 | 进阶级 |

**B3. 过程监督与稀疏奖励**

| 子知识点 | 详细内容 | 对应等级 |
|----------|----------|----------|
| B3.1 过程奖励模型（PRM） | 对推理每一步打分而非仅对最终结果打分；PRM vs ORM（Outcome Reward Model）对比 | 精通级 |
| B3.2 Step-level标注 | 人工标注每步正确性 vs 自动化标注（如用更强模型标注）；Monte Carlo Tree Search辅助标注 | 精通级 |
| B3.3 稀疏奖励解决 | 仅有最终结果对错的场景下如何提供中间步骤奖励；Reward Shaping；Hindsight Experience Replay | 精通级 |
| B3.4 成本收益分析 | Step-level标注的人力成本 vs 训练效果提升；何时值得使用过程监督 | 精通级 |

**B4. 高级奖励系统**

| 子知识点 | 详细内容 | 对应等级 |
|----------|----------|----------|
| B4.1 Distribution Shift | RM在训练分布外的表现下降；策略模型更新后RM与策略分布不匹配 | 专家级 |
| B4.2 Generative RL | 使用LLM本身作为奖励模型（LLM-as-a-Judge）；生成式评估 vs 判别式RM | 专家级 |
| B4.3 Ensemble RM | 训练多个RM取平均/最小值降低方差；不同RM捕捉不同偏好维度 | 专家级 |
| B4.4 Iterative RLHF | 策略更新后重新收集偏好数据并重训RM；动态RM更新策略 | 专家级 |
| B4.5 多目标奖励平衡 | 安全性 vs 有用性的冲突目标；约束优化、动态权重调整、Pareto最优 | 专家级 |

#### 3.2.2 学习资料规划

##### 📚 书籍

| 序号 | 书名 | 作者/来源 | 类型 | 难度 | 推荐理由 |
|------|------|-----------|------|------|----------|
| 1 | **The RLHF Book**（同上） | Nathan Lambert | 书籍 | 高 | 第3-5章深入讲解奖励模型训练、偏好数据收集、RM评估，以及Iterative RLHF |
| 2 | **Reinforcement Learning: An Introduction (2nd Edition)** | Richard S. Sutton, Andrew G. Barto (MIT Press, 2018) | 书籍 | 中-高 | RL领域圣经，第13章讲解策略梯度，第16章讲解RL与心理学/奖励的关系。免费在线版：http://www.incompleteideas.net/book/the-book-2nd.html |

##### 🎓 大学课程

| 序号 | 课程名 | 来源 | 类型 | 难度 | 推荐理由 |
|------|--------|------|------|------|----------|
| 1 | **CS285: Deep Reinforcement Learning** | UC Berkeley (Sergey Levine) | 课程 | 高 | 深度RL最权威课程，覆盖策略梯度、Actor-Critic、奖励设计。YouTube有完整录播 |
| 2 | **CS336: Language Modeling from Scratch - Lecture 15: Alignment** | Stanford | 课程 | 高 | 专门讲解RLHF中的奖励模型设计。课程主页：https://stanford-cs336.github.io/ |
| 3 | **Post-training of LLMs** | DeepLearning.AI | 课程 | 中 | 系统讲解RM训练、DPO等，含实操代码 |

##### 📝 博客与技术文章

| 序号 | 文章名 | 来源 | 类型 | 难度 | 推荐理由 |
|------|--------|------|------|------|----------|
| 1 | **Complete guide to RLHF for LLMs** | Toloka Blog | 博客 | 中 | 完整讲解RLHF三阶段流程、DPO vs RLHF对比、偏好数据收集方法。地址：https://toloka.ai/blog/what-is-rlhf/ |
| 2 | **Let's Verify Step by Step** | OpenAI (arXiv:2305.20050) | 论文 | 高 | PRM的开创性论文，提出过程监督方法，ICLR 2024 |
| 3 | **Let's reward step by step: Step-Level reward model as the Navigators for Reasoning** | OpenReview | 论文 | 高 | 系统研究Step-level奖励模型，PRM训练与应用 |
| 4 | **Reward Model Overoptimisation in Iterated RLHF** | arXiv:2505.18126 | 论文 | 高 | 深入分析Iterative RLHF中RM过优化问题，对理解Distribution Shift至关重要 |
| 5 | **Reward Hacking in RLHF** | TheoremPath | 博客 | 中-高 | 系统讲解Reward Hacking现象、成因和防御方法（Ensemble RM等）。地址：https://theorempath.com/topics/reward-hacking |
| 6 | **The Unverifiable Reward Problem: The Real Frontier of RL for LLMs** | ajing.github.io | 博客 | 高 | 讨论不可验证奖励的前沿问题，涵盖iStar、MA-RLHF等新方法 |

##### 💻 开源项目/代码库

| 序号 | 项目名 | 来源 | 类型 | 难度 | 推荐理由 |
|------|--------|------|------|------|----------|
| 1 | **RLHF-Reward-Modeling** | RLHFlow | 代码库 | 高 | 提供RM训练完整方案，含ArmoRM（多目标奖励+MoE）、半监督RM、迭代自训练RM。GitHub: https://github.com/RLHFlow/RLHF-Reward-Modeling |
| 2 | **Awesome-Process-Reward-Models** | RyanLiu112 | 资源列表 | 中-高 | PRM领域最全资源汇总，含论文、数据集、代码。GitHub: https://github.com/RyanLiu112/Awesome-Process-Reward-Models |
| 3 | **OpenRLHF** | OpenRLHF Team | 代码库 | 高 | 生产级RLHF框架，支持RM训练、PPO、DPO等。GitHub: https://github.com/OpenRLHF/OpenRLHF |
| 4 | **verl (Volcano Engine RL)** | ByteDance | 代码库 | 高 | 字节跳动开源的高性能RL训练框架，支持RM训练和多种RL算法。GitHub: https://github.com/verl-project/verl |
| 5 | **awesome-RLHF** | opendilab | 资源列表 | 中 | RLHF论文和资源汇总。GitHub: https://github.com/opendilab/awesome-RLHF |

---

### 3.3 强化学习训练算法

#### 3.3.1 知识点详细梳理

**C1. RL基础理论**

| 子知识点 | 详细内容 | 对应等级 |
|----------|----------|----------|
| C1.1 策略梯度 vs 价值学习 | Policy-based方法直接优化策略π(a\|s)；Value-based方法学习Q值函数；Actor-Critic结合两者 | 0-基础级 |
| C1.2 策略梯度推导 | REINFORCE算法；J(θ) = E[∑γ^t r_t]；∇J(θ) = E[∇log π(a\|s) · A(s,a)]；基线函数减少方差 | 基础级 |
| C1.3 Actor-Critic | Actor更新策略，Critic估计价值函数；GAE（Generalized Advantage Estimation）；偏差-方差权衡 | 基础级 |
| C1.4 优势函数估计 | Advantage A(s,a) = Q(s,a) - V(s)；TD误差、GAE、n-step return；优势函数归一化 | 基础级 |

**C2. 主流RL算法**

| 子知识点 | 详细内容 | 对应等级 |
|----------|----------|----------|
| C2.1 PPO | Clipped Surrogate Objective；ratio clipping防止策略更新过大；PPO在LLM中的具体实现（多模型协同：Actor/Critic/Reference/RM） | 1-进阶级 |
| C2.2 DPO | 将RLHF转化为监督学习；直接从偏好数据学习策略；β参数控制与参考模型的KL距离；无需RM和在线采样 | 进阶级 |
| C2.3 GRPO | DeepSeek提出，去除Critic模型；组内相对优势估计；适用于可验证奖励场景（数学、代码） | 进阶级 |
| C2.4 DAPO | ByteDance提出，解耦Clip + 动态采样；解决GRPO中的熵坍缩和训练不稳定 | 进阶级 |
| C2.5 其他变体 | KTO（Kahneman-Tversky Optimization）、IPO（Identity PO）、ORPO、SimPO等 | 进阶级 |

**C3. KL惩罚与训练稳定性**

| 子知识点 | 详细内容 | 对应等级 |
|----------|----------|----------|
| C3.1 KL散度 | D_KL(π_new \|\| π_ref)；衡量新策略与参考策略的偏离程度；在RLHF中防止模型偏离SFT太远 | 进阶级 |
| C3.2 PPO中的KL惩罚 | 显式KL惩罚项加入奖励；KL自适应控制器（目标KL → 动态调整惩罚系数） | 进阶级 |
| C3.3 DPO中的隐式KL | DPO通过β参数隐式约束KL；β越大越接近参考模型；DPO无需显式KL计算 | 进阶级 |
| C3.4 训练不稳定因素 | 奖励作弊、策略坍缩、梯度爆炸、Critic不稳定；诊断与处理方法 | 进阶级 |

**C4. 场景化RL设计**

| 子知识点 | 详细内容 | 对应等级 |
|----------|----------|----------|
| C4.1 Reasoning RL | 数学推理RL（可验证奖励）、逻辑推理RL；CoT + RL协同；搜索增强推理（MCTS + RL） | 精通级 |
| C4.2 Coding RL | 代码生成RL（编译/测试奖励）、代码修复RL；Test-time Compute与训练策略协同 | 精通级 |
| C4.3 Agent RL | 多步决策RL、工具调用RL；长horizon奖励分配；Agent环境的RL设计 | 精通级 |
| C4.4 多模态 RL | 视觉-语言任务的RL；多模态偏好数据；跨模态奖励设计 | 精通级 |

**C5. 大规模分布式RL训练**

| 子知识点 | 详细内容 | 对应等级 |
|----------|----------|----------|
| C5.1 分布式RL架构 | Actor/Critic/RM/Reference模型的分布式部署；Rollout与Training的并行架构；vLLM加速生成 | 专家级 |
| C5.2 训练诊断指标 | Gradient Norm（梯度范数监控）、Value Loss（Critic收敛情况）、Explained Variance（优势估计质量）、Entropy（策略多样性）、KL散度 | 专家级 |
| C5.3 训练崩塌根因 | 策略坍缩、奖励Hacking、Critic发散、梯度爆炸；定性定量诊断方法 | 专家级 |
| C5.4 第一性原理改进 | 从数学推导出发改进算法；如GRPO从PPO中去除Critic的理论依据；DAPO对GRPO的改进原理 | 专家级 |

#### 3.3.2 学习资料规划

##### 📚 书籍

| 序号 | 书名 | 作者/来源 | 类型 | 难度 | 推荐理由 |
|------|------|-----------|------|------|----------|
| 1 | **Reinforcement Learning: An Introduction (2nd Ed.)** | Sutton & Barto (MIT Press, 2018) | 书籍 | 中-高 | RL领域圣经。第13章策略梯度、第16章RL应用必读。免费版：http://www.incompleteideas.net/book/the-book-2nd.html |
| 2 | **The RLHF Book** | Nathan Lambert (Manning, 2026) | 书籍 | 高 | 专门聚焦RLHF，PPO/DPO/GRPO等算法在LLM中的实践。在线版：https://rlhfbook.com |
| 3 | **Deep Reinforcement Learning Hands-On (2nd Ed.)** | Maxim Lapan (Packt, 2020) | 书籍 | 中 | 实战导向，含PyTorch代码实现PPO、A2C等算法 |

##### 🎓 大学课程

| 序号 | 课程名 | 来源 | 类型 | 难度 | 推荐理由 |
|------|--------|------|------|------|----------|
| 1 | **CS285: Deep Reinforcement Learning** | UC Berkeley (Sergey Levine) | 课程 | 高 | 深度RL最权威课程，覆盖策略梯度、PPO、Actor-Critic全链路。YouTube完整录播 |
| 2 | **CS234: Reinforcement Learning** | Stanford (Emma Brunskill) | 课程 | 中-高 | Stanford RL课程，理论基础扎实。课程主页：https://web.stanford.edu/class/cs234/ |
| 3 | **CS336: Language Modeling from Scratch** | Stanford | 课程 | 高 | Lecture 15专讲SFT/RLHF Alignment。课程主页：https://stanford-cs336.github.io/ |
| 4 | **Spinning Up in Deep RL** | OpenAI | 教程 | 中 | OpenAI出品的深度RL入门教程，含PPO完整实现。地址：https://spinningup.openai.com/ |
| 5 | **Deep Reinforcement Learning** | David Silver (DeepMind) | 课程 | 中-高 | AlphaGo作者经典课程，覆盖MCTS、策略梯度。YouTube有完整录播 |

##### 📝 博客与技术文章

| 序号 | 文章名 | 来源 | 类型 | 难度 | 推荐理由 |
|------|--------|------|------|------|----------|
| 1 | **Reinforcement Learning for LLM Post-Training: A Survey** | arXiv:2407.16216 | 综述论文 | 高 | LLM后训练RL最全面综述，覆盖PPO/DPO/GRPO/DAPO等所有主流算法 |
| 2 | **From PPO to GDPO — Understanding the Evolution of RL Algorithms** | blog.munhou.com | 博客 | 中-高 | 清晰梳理从PPO→DPO→GRPO→DAPO→GDPO的算法演进脉络 |
| 3 | **GRPO and DAPO: Within-Group Reward Dispersion Gates** | blog.diffio.ai | 博客 | 高 | 深入分析GRPO/DAPO的数学原理和组内奖励分散机制 |
| 4 | **Grounding Language Models in Human Values: The RLHF-PPO Pipeline** | arraysoflight.com | 博客 | 中 | 详细讲解RLHF中PPO的工作机制、KL惩罚、Reward Hacking |
| 5 | **DAPO: An Open-Source LLM Reinforcement Learning System at Scale** | ByteDance (arXiv:2503.14476) | 论文 | 高 | DAPO原始论文，提出解耦Clip和动态采样策略 |
| 6 | **DeepSeekMath: Pushing the Limits of Mathematical Reasoning** | DeepSeek (arXiv:2402.03300) | 论文 | 高 | GRPO算法的原始论文，去除Critic模型的创新设计 |

##### 💻 开源项目/代码库

| 序号 | 项目名 | 来源 | 类型 | 难度 | 推荐理由 |
|------|--------|------|------|------|----------|
| 1 | **OpenRLHF** | OpenRLHF Team | 代码库 | 高 | 生产级RLHF框架，Ray + vLLM + DeepSpeed架构，支持PPO/DPO/GRPO。GitHub: https://github.com/OpenRLHF/OpenRLHF |
| 2 | **verl** | ByteDance | 代码库 | 高 | 字节开源的高性能RL框架，支持大规模分布式RL训练。GitHub: https://github.com/verl-project/verl |
| 3 | **TRL** | Hugging Face | 代码库 | 中 | HF官方训练库，支持PPOTrainer、DPOTrainer、GRPOTrainer等。GitHub: https://github.com/huggingface/trl |
| 4 | **DeepSpeed-Chat** | Microsoft | 代码库 | 中-高 | 微软开源的RLHF训练框架，PPO三阶段实现。GitHub: https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat |
| 5 | **DAPO** | ByteDance | 代码库 | 高 | DAPO算法官方实现。项目页：https://dapo-sia.github.io/ |

---

### 3.4 训练数据处理、数据清洗与数据合成技术实践

#### 3.4.1 知识点详细梳理

**D1. 数据清洗基础**

| 子知识点 | 详细内容 | 对应等级 |
|----------|----------|----------|
| D1.1 数据清洗流程 | 原始数据→去重（MinHash/SimHash）→质量过滤→毒性过滤→PII脱敏→格式标准化 | 基础级 |
| D1.2 去重与过滤 | 精确去重、模糊去重；基于规则的过滤（长度、语言、编码）；基于模型的过滤（质量分类器） | 基础级 |
| D1.3 数据污染检测 | 检测训练数据中的测试集泄露；Benchmark contamination检测方法 | 进阶级 |

**D2. 数据合成方法**

| 子知识点 | 详细内容 | 对应等级 |
|----------|----------|----------|
| D2.1 Self-Instruct | 用LLM自己生成指令数据；种子指令→生成→过滤→迭代；Stanford Alpaca的核心方法 | 基础级 |
| D2.2 Self-QA | 模型自问自答生成QA对；基于知识源自动构造问答数据 | 基础级 |
| D2.3 Evol-Instruct | WizardLM提出，逐步增加指令复杂度；深度进化和广度进化策略 | 进阶级 |
| D2.4 基于强模型的蒸馏合成 | 用GPT-4/Claude等强模型生成高质量训练数据；蒸馏数据的质量控制 | 进阶级 |

**D3. 高级数据工程**

| 子知识点 | 详细内容 | 对应等级 |
|----------|----------|----------|
| D3.1 多算法综合运用 | 结合Self-Instruct + Evol-Instruct + 蒸馏 + 过滤的完整流水线 | 精通级 |
| D3.2 数据质量自动评估 | 用RM/LLM-as-Judge自动评估合成数据质量；数据质量评分模型 | 精通级 |
| D3.3 数据配比自动化搜索 | 基于模型反馈自动搜索最优数据配比；Data Mixing Law；DoReMi等方法 | 精通级 |

**D4. 创新性数据方法**

| 子知识点 | 详细内容 | 对应等级 |
|----------|----------|----------|
| D4.1 创新合成算法 | 设计新的数据合成方法（如CoT-Self-Instruct、Source2Synth等） | 专家级 |
| D4.2 创新清洗与配比 | 提出新的数据清洗策略或配比算法，获得显著效果提升 | 专家级 |
| D4.3 效果验证 | 设计A/B实验验证数据方法效果；消融实验设计 | 专家级 |

#### 3.4.2 学习资料规划

##### 📚 书籍与论文

| 序号 | 名称 | 作者/来源 | 类型 | 难度 | 推荐理由 |
|------|------|-----------|------|------|----------|
| 1 | **Self-Instruct: Aligning Language Models with Self-Generated Instructions** | arXiv:2212.10560 | 论文 | 中 | Self-Instruct方法的开创性论文，Stanford Alpaca的基础 |
| 2 | **WizardLM: Empowering Large Language Models to Follow Complex Instructions** | arXiv:2304.12244 | 论文 | 中 | Evol-Instruct方法的原始论文 |
| 3 | **CoT-Self-Instruct: Building high-quality synthetic data** | Papers with Code | 论文 | 中-高 | 结合CoT推理和Self-Instruct的最新数据合成方法 |
| 4 | **Data Cleaning for NLP** | Kaggle Learn | 教程 | 低-中 | 数据清洗基础实践教程。地址：https://www.kaggle.com/learn/data-cleaning |

##### 🎓 课程与教程

| 序号 | 课程名 | 来源 | 类型 | 难度 | 推荐理由 |
|------|--------|------|------|------|----------|
| 1 | **Post-training of LLMs** | DeepLearning.AI | 课程 | 中 | 包含数据准备和清洗的实践章节 |
| 2 | **Hugging Face LLM Course** | Hugging Face | 教程 | 中 | 包含数据预处理、Tokenization、数据集构建的完整教程 |
| 3 | **LLM Fine-Tuning and Customization** | Coursera | 课程 | 中 | 覆盖数据准备、超参调优、评估策略 |

##### 💻 开源项目/代码库

| 序号 | 项目名 | 来源 | 类型 | 难度 | 推荐理由 |
|------|--------|------|------|------|----------|
| 1 | **stanford_alpaca** | Stanford | 代码库 | 中 | Self-Instruct方法的原始实现，含数据生成脚本。GitHub: https://github.com/tatsu-lab/stanford_alpaca |
| 2 | **WizardLM** | Xu et al. | 代码库 | 中 | Evol-Instruct方法的实现。GitHub: https://github.com/nlpxucan/WizardLM |
| 3 | **datasets** | Hugging Face | 代码库 | 低-中 | HF数据集库，含大量SFT数据集和数据处理工具。GitHub: https://github.com/huggingface/datasets |
| 4 | **LLaMA-Factory** | hiyouga | 代码库 | 中 | 含完整的数据处理流水线，支持多种数据格式。GitHub: https://github.com/hiyouga/LLaMA-Factory |

---

### 3.5 MoE模型RL训练一致性和训练稳定性分析

#### 3.5.1 知识点详细梳理

**E1. MoE基础**

| 子知识点 | 详细内容 | 对应等级 |
|----------|----------|----------|
| E1.1 稀疏激活机制 | MoE只激活部分专家（Top-k），减少计算量；稀疏激活的数学表达 | 基础级 |
| E1.2 路由机制 | Gate/Router网络计算专家分配概率；Top-k选择；负载均衡损失（Load Balancing Loss） | 基础级 |
| E1.3 Top-k门控 | k=1（硬路由）vs k=2（软路由）；门控函数设计（线性门控、噪声门控） | 基础级 |
| E1.4 MoE在LLM中的应用 | Mixtral、DeepSeek-MoE、GPT-4 MoE架构；专家数量与粒度设计 | 基础级 |

**E2. MoE RL训练特殊问题**

| 子知识点 | 详细内容 | 对应等级 |
|----------|----------|----------|
| E2.1 训推不一致性 | 训练时和推理时的路由行为不同；训练用Teacher Forcing，推理用自回归；精度差异导致路由不同 | 1-进阶级 |
| E2.2 路由漂移 | RL训练过程中策略变化导致路由分布变化；路由不稳定影响训练收敛 | 进阶级 |
| E2.3 常见不稳定问题 | 梯度通过路由的可微性问题；专家利用率不均；部分专家"死亡" | 1-进阶级 |

**E3. 稳定性提升算法**

| 子知识点 | 详细内容 | 对应等级 |
|----------|----------|----------|
| E3.1 IcePop | 双面遮蔽（Double-sided Masking）方法；对训练和推理的路由进行对齐 | 2-精通级 |
| E3.2 Routing Replay | 在RL训练时重放推理时的路由路径；确保梯度沿相同专家路径传播 | 2-精通级 |
| E3.3 Off-policy Masking | 处理off-policy数据中的路由不一致；掩蔽不匹配的路由 | 2-精通级 |
| E3.4 Importance Sampling | 修正off-policy分布偏差；重要性权重计算与截断 | 2-精通级 |

**E4. 高级MoE RL**

| 子知识点 | 详细内容 | 对应等级 |
|----------|----------|----------|
| E4.1 Gradient Norm爆炸 | MoE RL中梯度范数异常增长的根因分析；路由不稳定与梯度爆炸的关联 | 专家级 |
| E4.2 Reward Hacking | MoE模型可能利用路由机制进行奖励作弊；检测与防御策略 | 专家级 |
| E4.3 分布式MoE RL架构 | 专家并行 + 数据并行 + 流水线并行的混合架构；通信优化 | 专家级 |
| E4.4 Inference-time Compute优化 | 推理时搜索（如MCTS）与MoE路由的协同优化；Test-time Compute Scaling | 专家级 |

#### 3.5.2 学习资料规划

##### 📚 论文与技术报告

| 序号 | 名称 | 作者/来源 | 类型 | 难度 | 推荐理由 |
|------|------|-----------|------|------|----------|
| 1 | **Stabilizing MoE Reinforcement Learning by Aligning Training and Inference Routers** | ai-search.io | 论文 | 高 | 直接研究MoE RL训练推理一致性问题的核心论文 |
| 2 | **IcePop: Double-sided Masking for MoE RL Training Stability** | 相关研究组 | 论文 | 高 | IcePop方法的原始论文，双面遮蔽解决训推不一致 |
| 3 | **Taming Training-Inference Mismatch in RL with Adaptive Masking** | ringtech.notion.site | 技术文档 | 高 | IcePop的改进方法，自适应遮蔽替代均匀约束 |
| 4 | **Mixtral of Experts** | Mistral AI (arXiv:2401.04088) | 论文 | 中 | MoE在LLM中的经典应用，理解MoE架构基础 |
| 5 | **DeepSeek-MoE: Towards Ultimate Expert Specialization** | DeepSeek (arXiv:2401.06066) | 论文 | 中-高 | 细粒度专家分割和共享专家设计 |
| 6 | **Structuring MoE Expert Selection for Agentic RL** | ResearchGate | 论文 | 高 | 研究多轮Agent任务中MoE RL的路由控制问题 |

##### 📝 博客与技术文章

| 序号 | 文章名 | 来源 | 类型 | 难度 | 推荐理由 |
|------|--------|------|------|------|----------|
| 1 | **MiMo-V2.6: Scaling RL Towards Self-Improvement** | chapterpal.com | 技术报告 | 高 | 分析MoE路由对齐和Top-p采样在RL中的应用 |
| 2 | **Macaron-V1: 749B MoL Agent Model post-trained from GLM5.1** | macaron.im | 技术报告 | 高 | 描述Router Replay（R3）在MoE RL中的实践 |
| 3 | **ERNIE 5.0 Technical Report** | Baidu | 技术报告 | 高 | 弹性训练中路由和专家池的设计 |

##### 🎓 基础课程

| 序号 | 课程名 | 来源 | 类型 | 难度 | 推荐理由 |
|------|--------|------|------|------|----------|
| 1 | **CS285: Deep RL** | UC Berkeley | 课程 | 高 | 理解RL训练不稳定性的理论基础 |
| 2 | **Mixtral/MoE Architecture Analysis** | Various blogs | 博客 | 中 | 理解MoE架构设计，为RL稳定性分析打基础 |

##### 💻 开源项目/代码库

| 序号 | 项目名 | 来源 | 类型 | 难度 | 推荐理由 |
|------|--------|------|------|------|----------|
| 1 | **Megatron-LM** | NVIDIA | 代码库 | 高 | 支持MoE训练的大规模分布式框架。GitHub: https://github.com/NVIDIA/Megatron-LM |
| 2 | **DeepSpeed-MoE** | Microsoft | 代码库 | 高 | 微软MoE训练框架，含MoE RL相关支持。GitHub: https://github.com/microsoft/DeepSpeed |
| 3 | **verl** | ByteDance | 代码库 | 高 | 支持MoE模型的RL训练。GitHub: https://github.com/verl-project/verl |
| 4 | **OpenRLHF** | OpenRLHF Team | 代码库 | 高 | 支持MoE模型的RLHF训练。GitHub: https://github.com/OpenRLHF/OpenRLHF |

---

### 3.6 多专项模型融合、蒸馏策略

#### 3.6.1 知识点详细梳理

**F1. 知识蒸馏基础**

| 子知识点 | 详细内容 | 对应等级 |
|----------|----------|----------|
| F1.1 蒸馏基本原理 | Teacher-Student框架；KL散度最小化；温度参数T的作用 | 基础级 |
| F1.2 软标签 vs 硬标签 | Soft labels（logit分布）携带更多信息；Hard labels（one-hot）信息量少；软标签的"暗知识" | 基础级 |
| F1.3 Logit蒸馏 vs 特征蒸馏 | Logit层面匹配（输出分布）；Feature层面匹配（中间层表示）；不同蒸馏层面的优缺点 | 基础级 |
| F1.4 多教师蒸馏 | 多个Teacher模型同时指导Student；加权融合Teacher输出；多教师蒸馏的挑战 | 基础级 |

**F2. 模型融合（Model Merge）**

| 子知识点 | 详细内容 | 对应等级 |
|----------|----------|----------|
| F2.1 SLERP | 球面线性插值；在权重空间中沿球面路径插值；仅支持两模型融合 | 进阶级 |
| F2.2 TIES-Merging | Trim（修剪冗余参数）+ Elect Sign（选举符号）+ Merge（合并）；支持多模型同时融合 | 进阶级 |
| F2.3 DARE | Drop And REscale；随机丢弃参数差异再缩放；保留预训练知识 | 进阶级 |
| F2.4 Task Arithmetic | 任务向量 = 微调模型 - 预训练模型；任务向量加减实现能力增删 | 进阶级 |
| F2.5 Model Soup | 同一预训练模型不同微调配置的等权平均；简单但有效 | 进阶级 |

**F3. 多任务学习与梯度冲突**

| 子知识点 | 详细内容 | 对应等级 |
|----------|----------|----------|
| F3.1 梯度冲突 | 不同任务的梯度方向相反；负梯度余弦相似度；导致训练震荡 | 进阶级 |
| F3.2 梯度冲突解决 | PCGrad（投影冲突梯度）、CAGrad（冲突避免梯度）、GradNorm；梯度操作策略 | 进阶级 |
| F3.3 多任务RL | Multi-task RL中的任务调度；共享与任务特定参数的设计；任务间知识迁移 | 精通级 |

**F4. On-policy KD与高级策略**

| 子知识点 | 详细内容 | 对应等级 |
|----------|----------|----------|
| F4.1 On-policy KD原理 | Student在自己生成的轨迹上接收Teacher的token级反馈；区别于Off-policy蒸馏（在Teacher数据上学习） | 精通级 |
| F4.2 缓解Distribution Shift | On-policy KD通过在Student自身分布上训练，减少Teacher-Student分布不匹配 | 精通级 |
| F4.3 On-policy KD与RL的关系 | On-policy KD可视为RL的一种特殊形式（Teacher提供密集奖励信号）；与RLHF的协同 | 精通级 |
| F4.4 算法变体 | Generalized On-policy KD with Reward Extrapolation；双向蒸馏；交互式蒸馏 | 精通级 |
| F4.5 适用场景 | 多专项模型融合场景；大模型蒸馏到小模型；RL训练后的能力保持 | 3-专家级 |

#### 3.6.2 学习资料规划

##### 📚 论文与书籍

| 序号 | 名称 | 作者/来源 | 类型 | 难度 | 推荐理由 |
|------|------|-----------|------|------|----------|
| 1 | **Model Merging: SLERP, TIES, DARE** | neelmishra.github.io | 博客 | 中 | 清晰讲解SLERP、TIES、DARE三种融合方法的原理和区别 |
| 2 | **An Introduction to Model Merging for LLMs** | NVIDIA Technical Blog | 博客 | 中 | NVIDIA出品，系统介绍Model Soup、SLERP、Task Arithmetic、TIES、DARE。地址：https://developer.nvidia.com/blog/an-introduction-to-model-merging-for-llms/ |
| 3 | **Awesome-Model-Merging-Methods-Theories-Applications** | EnnengYang | 资源列表 | 中-高 | 模型融合领域最全资源汇总，含25+种融合策略。GitHub: https://github.com/EnnengYang/Awesome-Model-Merging-Methods-Theories-Applications |
| 4 | **Learning beyond Teacher: Generalized On-Policy Distillation with Reward Extrapolation** | greeksharifa.github.io | 博客 | 高 | 深入讲解On-policy KD的原理和改进方法，含与Off-policy蒸馏的对比 |
| 5 | **Why On-Policy Distillation Sometimes Fails: Vanishing Learning Signals** | arXiv:2610.11247 | 论文 | 高 | 分析On-policy KD失败的原因，对理解其适用场景至关重要 |
| 6 | **VOLD: Reasoning Transfer via On-Policy Distillation** | themoonlight.io | 综述 | 高 | 系统综述On-policy KD在推理能力迁移中的应用 |
| 7 | **A Comprehensive Guide on Merging Language Models** | ionio.ai | 博客 | 中 | 实践导向的模型融合指南 |

##### 🎓 课程

| 序号 | 课程名 | 来源 | 类型 | 难度 | 推荐理由 |
|------|--------|------|------|------|----------|
| 1 | **CS224N: NLP with Deep Learning** | Stanford | 课程 | 中-高 | 涵盖知识蒸馏在NLP中的应用 |
| 2 | **CS336: Language Modeling from Scratch** | Stanford | 课程 | 高 | 含模型蒸馏和融合的现代LLM技术 |

##### 💻 开源项目/代码库

| 序号 | 项目名 | 来源 | 类型 | 难度 | 推荐理由 |
|------|--------|------|------|------|----------|
| 1 | **mergekit** | arcee-ai | 代码库 | 中 | 最流行的模型融合工具，支持SLERP、TIES、DARE、Task Arithmetic等。GitHub: https://github.com/arcee-ai/mergekit |
| 2 | **Model Merging with MergeKit** | LLM Dojo | 教程 | 中 | MergeKit实践教程，含2024-2025最新技术。地址：https://llm-dojo.grotto.run/notebook/68-model-merging |
| 3 | **Merge Large Language Models with mergekit** | Hugging Face Blog | 教程 | 中 | HF官方博客，手把手教学模型融合。地址：https://huggingface.co/blog/mlabonne/merge-models |
| 4 | **Knowledge Distillation for LLMs** | Krishnaik | 教程 | 中 | 知识蒸馏实践教程，含面试要点。地址：https://www.krishnaik.in/ai-tutorials/llm-fundamentals-interview-questions/knowledge-distillation-llms |

---

## 4. 学习路径建议

### 4.1 推荐学习顺序

基于知识依赖关系，推荐以下学习顺序：

```
Phase 1: 基础理论筑基（6-8周）
├── Step 1: RL基础理论
│   ├── Sutton & Barto 第1-9章（核心概念、MDP、动态规划、TD学习）
│   ├── Sutton & Barto 第13章（策略梯度）
│   └── CS285 前5讲（深度RL基础）
│
├── Step 2: LLM基础
│   ├── CS224N 核心章节（Transformer、预训练）
│   └── Hugging Face LLM Course 第1-4章
│
└── Step 3: 后训练入门
    ├── DeepLearning.AI "Post-training of LLMs" 课程
    └── The RLHF Book 第1-2章（历史与概述）

Phase 2: 核心技术深入（8-10周）
├── Step 4: SFT技术
│   ├── The RLHF Book 第3章（SFT）
│   ├── TRL SFTTrainer 实践
│   └── 数据合成方法学习（Self-Instruct、Evol-Instruct）
│
├── Step 5: 奖励模型
│   ├── The RLHF Book 第4章（RM）
│   ├── RLHF-Reward-Modeling 项目实践
│   ├── PRM论文精读（Let's Verify Step by Step）
│   └── Awesome-Process-Reward-Models 资源学习
│
└── Step 6: RL算法
    ├── PPO原理与实现（Spinning Up + OpenRLHF）
    ├── DPO原理与实践（TRL DPOTrainer）
    ├── GRPO/DAPO论文精读
    └── KL惩罚机制深入理解

Phase 3: 高级专题（8-10周）
├── Step 7: MoE RL训练
│   ├── MoE架构基础（Mixtral、DeepSeek-MoE论文）
│   ├── IcePop、Routing Replay 论文精读
│   └── verl/OpenRLHF MoE RL 实践
│
├── Step 8: 模型融合与蒸馏
│   ├── Model Merge 实践（mergekit）
│   ├── On-policy KD 论文精读
│   └── 多任务RL与梯度冲突
│
└── Step 9: 数据工程进阶
    ├── 数据合成高级方法
    ├── 自动化数据评测流水线搭建
    └── 数据配比自动化搜索

Phase 4: 实战与综合复习（4-6周）
├── Step 10: 综合实战
│   ├── 完整RLHF流水线搭建（SFT→RM→PPO）
│   ├── DPO/GRPO训练实践
│   └── MoE RL训练实验
│
├── Step 11: 论文复现
│   ├── 选择1-2篇关键论文复现
│   └── 在公开数据集上验证
│
└── Step 12: 综合复习
    ├── 梳理各知识点0-专家级能力描述
    ├── 准备项目经历和技术深度展示
    └── 模拟自测练习
```

### 4.2 各阶段重点资源

| 阶段 | 核心资源 | 输出目标 |
|------|----------|----------|
| Phase 1 | Sutton & Barto + CS285 + CS224N + DL.AI课程 | 掌握RL和LLM基础理论 |
| Phase 2 | RLHF Book + TRL/OpenRLHF + 关键论文 | 能独立完成SFT→RM→RL训练 |
| Phase 3 | MoE论文 + mergekit + On-policy KD论文 | 掌握MoE RL和模型融合技术 |
| Phase 4 | 完整项目实践 + 论文复现 | 具备岗位所需的全部能力 |

---

## 5. 推荐学习时间线

### 5.1 全量学习计划（约6-7个月）

| 月份 | 周次 | 学习内容 | 核心产出 |
|------|------|----------|----------|
| **第1月** | W1-W2 | RL基础：Sutton & Barto 第1-5章 + CS285 L1-L3 | MDP建模、TD学习笔记 |
| | W3-W4 | RL基础：Sutton & Barto 第13章 + CS285 L4-L5 | 策略梯度推导、Actor-Critic理解 |
| **第2月** | W1-W2 | LLM基础：CS224N核心章节 + HF LLM Course | Transformer原理、预训练流程 |
| | W3-W4 | 后训练入门：DL.AI课程 + RLHF Book 第1-2章 | 后训练全流程理解 |
| **第3月** | W1-W2 | SFT深入：RLHF Book 第3章 + TRL SFT实践 | SFT训练代码可运行 |
| | W3-W4 | 数据合成：Self-Instruct/Evol-Instruct论文 + 实践 | 合成数据集产出 |
| **第4月** | W1-W2 | 奖励模型：RLHF Book 第4章 + RM训练实践 | RM训练代码可运行 |
| | W3-W4 | PRM与过程监督：论文精读 + PRM实践 | PRM理解与实现 |
| **第5月** | W1-W2 | RL算法：PPO/DPO/GRPO/DAPO论文精读 | 算法对比分析文档 |
| | W3-W4 | RL实践：OpenRLHF/verl 完整RLHF训练 | RLHF训练完成 |
| **第6月** | W1-W2 | MoE RL：MoE架构 + IcePop/Routing Replay | MoE RL训练实验 |
| | W3-W4 | 模型融合与蒸馏：mergekit + On-policy KD | 融合与蒸馏实验 |
| **第7月** | W1-W2 | 综合实战：完整后训练流水线搭建 | 端到端项目 |
| | W3-W4 | 综合复习：知识点梳理 + 模拟自测 | 面试就绪 |

### 5.2 加速学习计划（约4个月，适合有ML基础者）

| 月份 | 学习内容 | 跳过/简化 |
|------|----------|-----------|
| **第1月** | DL.AI课程 + RLHF Book + CS285精选讲座 | 跳过Sutton & Barto前9章（直接学第13章） |
| **第2月** | SFT + RM + PPO/DPO 实践（TRL/OpenRLHF） | 简化理论推导，聚焦实践 |
| **第3月** | GRPO/DAPO + MoE RL + 模型融合 | 聚焦核心论文，跳过边缘方法 |
| **第4月** | 综合实战 + 综合复习 | — |

---

## 附录：核心资源快速索引

### 📖 必读书籍
1. **The RLHF Book** — Nathan Lambert — https://rlhfbook.com
2. **Reinforcement Learning: An Introduction (2nd Ed.)** — Sutton & Barto — http://www.incompleteideas.net/book/the-book-2nd.html

### 🎓 必学课程
1. **CS285: Deep RL** — UC Berkeley — YouTube
2. **CS336: Language Modeling from Scratch** — Stanford — https://stanford-cs336.github.io/
3. **Post-training of LLMs** — DeepLearning.AI — https://www.deeplearning.ai/courses/post-training-of-llms
4. **CS224N: NLP with Deep Learning** — Stanford — https://web.stanford.edu/class/cs224n/

### 💻 必练项目
1. **TRL** — Hugging Face — https://github.com/huggingface/trl
2. **OpenRLHF** — https://github.com/OpenRLHF/OpenRLHF
3. **verl** — ByteDance — https://github.com/verl-project/verl
4. **mergekit** — https://github.com/arcee-ai/mergekit
5. **RLHF-Reward-Modeling** — https://github.com/RLHFlow/RLHF-Reward-Modeling

### 📄 必读论文
1. **PPO** — Schulman et al. (2017)
2. **DPO** — Rafailov et al. (2023) — arXiv:2305.18290
3. **GRPO** — DeepSeekMath — arXiv:2402.03300
4. **DAPO** — ByteDance — arXiv:2503.14476
5. **Let's Verify Step by Step** — OpenAI — arXiv:2305.20050
6. **Self-Instruct** — arXiv:2212.10560
7. **Reward Model Overoptimisation in Iterated RLHF** — arXiv:2505.18126
8. **Reinforcement Learning for LLM Post-Training: A Survey** — arXiv:2407.16216

### 📚 必看资源列表
1. **awesome-RLHF** — https://github.com/opendilab/awesome-RLHF
2. **Awesome-Process-Reward-Models** — https://github.com/RyanLiu112/Awesome-Process-Reward-Models
3. **Awesome-Model-Merging-Methods-Theories-Applications** — https://github.com/EnnengYang/Awesome-Model-Merging-Methods-Theories-Applications

---

> **文档说明**：本学习计划针对"AI模型-后训练与强化学习"岗位编制，覆盖全部3个专业知识点和3个关键技能点，每个知识点均展开到子知识点级别，并配有从基础到高级的完整学习资料。建议按Phase 1→4的顺序系统学习，确保在每个阶段都完成实践项目后再进入下一阶段。