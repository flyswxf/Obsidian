## 目录

- [[#1. 岗位概述]]
- [[#2. 知识体系全景图（知识树）]]
- [[#3. 分领域详细知识点梳理]]
  - [[#3.1 智能体记忆能力]]
  - [[#3.2 智能体轨迹数据合成与微调训练]]
  - [[#3.3 智能体强化学习]]
  - [[#3.4 智能体评估]]
  - [[#3.5 通用型Agent工具使用（ClaudeCode/OpenClaw等）]]
  - [[#3.6 面向基模的智能体Harness搭建]]
  - [[#3.7 沙箱环境工程|3.7 沙箱环境工程能力]]
- [[#4. 分领域学习资料规划]]
- [[#5. 学习路径建议]]

---

## 1. 岗位概述

### 1.1 岗位定位

AI模型-Agent技术岗位聚焦于**大语言模型智能体（LLM Agent）**的全栈技术能力，涵盖从底层记忆系统设计、训练数据合成、强化学习对齐、评测体系构建，到上层 harness 框架开发与沙箱工程基础设施的完整技术链路。该岗位要求候选人既具备扎实的机器学习/深度学习理论基础，又拥有工程实践能力，能够独立搭建、训练、评估和部署智能体系统。

### 1.2 核心能力矩阵

| 能力维度 | 考察重点                     | 目标等级     |
| ---- | ------------------------ | -------- |
| 专业知识 | 记忆能力、轨迹数据合成、强化学习、评估      | 精通级（实操经验） |
| 关键技能 | Agent工具使用、Harness搭建、沙箱工程 | 精通级（独立开发） |

### 1.3 能力发展路径

```
入门（基础级）→ 理解原理（进阶级）→ 实操经验（精通级）→ 业界影响力（专家级）
    ↓              ↓                ↓                  ↓
 了解概念      跑通开源项目     独立开发/优化      开源/论文/产品
```

---

## 2. 知识体系全景图（知识树）

```
AI模型-Agent技术
├── 1. 智能体记忆能力
│   ├── 1.1 基础概念
│   │   ├── 序列长度与上下文窗口
│   │   ├── 记忆对智能体的意义
│   │   └── 记忆类型分类（短期/长期/工作记忆/情景记忆）
│   ├── 1.2 长短期记忆构建方法
│   │   ├── 短期记忆（上下文窗口管理）
│   │   │   ├── 滑动窗口策略
│   │   │   ├── 注意力机制与上下文压缩
│   │   │   └── Token预算分配
│   │   ├── 长期记忆（外部存储）
│   │   │   ├── 向量数据库检索（RAG）
│   │   │   ├── 知识图谱存储
│   │   │   ├── 结构化数据库存储
│   │   │   └── 混合存储架构
│   │   └── 工作记忆（中间状态管理）
│   ├── 1.3 记忆Refine方法
│   │   ├── 记忆摘要与压缩
│   │   ├── 记忆合并与去重
│   │   ├── 记忆重要性评分与遗忘机制
│   │   └── 记忆检索排序优化
│   ├── 1.4 记忆加载方法
│   │   ├── 全量加载 vs 选择性加载
│   │   ├── 检索增强加载（RAG-based）
│   │   ├── 上下文窗口注入策略
│   │   └── 动态记忆加载与卸载
│   └── 1.5 个性化记忆与经验记忆
│       ├── 用户画像建模
│       ├── 偏好学习与记忆
│       ├── 经验复用与迁移
│       └── 长时任务中的记忆管理
│
├── 2. 智能体轨迹数据合成与微调训练
│   ├── 2.1 大模型训练流程基础
│   │   ├── 预训练（Pre-training）
│   │   ├── 退火（Annealing / Cooldown）
│   │   ├── 持续预训练（CPT）
│   │   ├── 监督微调（SFT）
│   │   └── 对齐训练（RLHF/DPO等）
│   ├── 2.2 智能体轨迹数据合成
│   │   ├── 轨迹数据定义与结构
│   │   │   ├── Thought-Action-Observation三元组
│   │   │   ├── 多轮对话轨迹格式
│   │   │   └── 工具调用轨迹格式
│   │   ├── 轨迹合成方法
│   │   │   ├── 专家示范合成
│   │   │   ├── 模型自生成（Self-play / Self-instruct）
│   │   │   ├── 蒸馏合成（强模型→弱模型）
│   │   │   ├── 回溯合成（Backtracking）
│   │   │   └── 混合合成策略
│   │   └── 模仿学习的局限性
│   │       ├── 分布偏移问题
│   │       ├── 错误累积与级联失败
│   │       ├── 探索不足
│   │       └── 长程任务泛化困难
│   ├── 2.3 轨迹数据质量筛选
│   │   ├── Turn-level质量筛选
│   │   │   ├── 单步质量评分
│   │   │   ├── 动作合法性检查
│   │   │   └── 工具调用正确性验证
│   │   ├── 步骤级（Step-level）质量筛选
│   │   │   ├── 过程奖励模型（PRM）
│   │   │   ├── 蒙特卡洛树搜索评估
│   │   │   └── 轨迹一致性检查
│   │   └── 数据Curation策略
│   │       ├── 去重与多样性保证
│   │       ├── 难度分级
│   │       ├── 课程学习排序
│   │       └── 数据配比与混合
│   ├── 2.4 SFT训练实践
│   │   ├── SFT数据格式设计
│   │   ├── 多轮对话SFT
│   │   ├── 工具调用SFT
│   │   ├── 损失函数设计（Mask策略）
│   │   └── 训练超参调优
│   └── 2.5 大规模轨迹合成（Scaling）
│       ├── 10B+ token级合成策略
│       ├── 自动化合成管线设计
│       ├── 合成数据质量监控
│       └── 工业级规模化的挑战
│
├── 3. 智能体强化学习
│   ├── 3.1 强化学习基础
│   │   ├── MDP与POMDP
│   │   ├── 策略梯度方法
│   │   ├── 价值函数估计
│   │   └── 探索与利用
│   ├── 3.2 核心RL算法
│   │   ├── PPO（Proximal Policy Optimization）
│   │   │   ├── 裁剪目标函数
│   │   │   ├── GAE（Generalized Advantage Estimation）
│   │   │   └── PPO在LLM中的应用
│   │   ├── GRPO（Group Relative Policy Optimization）
│   │   │   ├── 组内相对优势估计
│   │   │   ├── 去除Critic的设计
│   │   │   └── DeepSeek-R1中的应用
│   │   ├── OPD（Optimal Policy Distillation）
│   │   │   ├── 策略蒸馏原理
│   │   │   └── 在线/离线蒸馏
│   │   └── 其他相关算法
│   │       ├── DPO（Direct Preference Optimization）
│   │       ├── RLAIF / RLHF
│   │       ├── ReST（Reinforced Self-Training）
│   │       └── RFT（Rejection Sampling Fine-Tuning）
│   ├── 3.3 智能体RL特有挑战
│   │   ├── Agent RL vs Reasoning RL的差异化
│   │   │   ├── 动作空间复杂性（工具调用vs文本生成）
│   │   │   ├── 环境交互的不可逆性
│   │   │   ├── 延迟奖励与稀疏奖励
│   │   │   └── 长程轨迹的信用分配
│   │   ├── 黑盒强化学习
│   │   │   ├── 环境不可微的挑战
│   │   │   ├── 黑盒优化方法
│   │   │   └── 代理模型方法
│   │   └── 长程轨迹训练稳定性
│   │       ├── 梯度消失/爆炸
│   │       ├── 训练崩溃与恢复
│   │       └── 课程学习策略
│   ├── 3.4 Task合成技术
│   │   ├── 任务自动生成
│   │   ├── 任务难度分级
│   │   ├── 任务多样性保证
│   │   └── 课程学习任务编排
│   ├── 3.5 环境模拟与沙箱构建
│   │   ├── 环境接口设计
│   │   ├── 状态空间建模
│   │   ├── 奖励函数设计
│   │   ├── 环境并行化
│   │   └── 模拟器与真实环境对齐
│   └── 3.6 训练超参与调优
│       ├── KL散度约束（KL coefficient）
│       ├── 熵正则化（Entropy coefficient）
│       ├── 学习率调度
│       ├── 批大小与梯度累积
│       ├── Rollout batch size
│       └── 超参搜索策略
│
├── 4. 智能体评估
│   ├── 4.1 评估基础理论
│   │   ├── 评估的作用与意义
│   │   ├── 评估维度分类
│   │   │   ├── 工具使用能力
│   │   │   ├── 推理能力
│   │   │   ├── 多轮对话能力
│   │   │   ├── 长程任务完成能力
│   │   │   └── 安全性与鲁棒性
│   │   └── 评估方法论
│   │       ├── 静态评估 vs 动态评估
│   │       ├── 离线评估 vs 在线评估
│   │       └── 人工评估 vs 自动评估
│   ├── 4.2 主流评测榜单
│   │   ├── BFCL（Berkeley Function Calling Leaderboard）
│   │   │   ├── 评估维度与任务类型
│   │   │   ├── 评测方法与指标
│   │   │   └── 局限性分析
│   │   ├── TauBench
│   │   │   ├── 用户模拟评估
│   │   │   ├── 多轮工具调用评估
│   │   │   └── 与真实体验的关联
│   │   ├── Toolathlon
│   │   │   ├── 多工具组合任务
│   │   │   └── 评估指标设计
│   │   ├── BrowseComp
│   │   │   ├── 浏览器操作评估
│   │   │   └── 信息检索与推理
│   │   ├── Claw-Eval
│   │   │   ├── 评估维度
│   │   │   └── 动态环境评测
│   │   └── 其他重要榜单
│   │       ├── AgentBench
│   │       ├── SWE-bench
│   │       ├── WebArena
│   │       ├── GAIA
│   │       └── ToolBench
│   ├── 4.3 评估局限性分析
│   │   ├── 静态榜单的过拟合风险
│   │   ├── 评估与真实体验的Gap
│   │   ├── 评估覆盖面不足
│   │   └── 数据污染问题
│   └── 4.4 高级评估方向
│       ├── 动态环境评测
│       ├── 超长程任务评测
│       ├── 多智能体协作评测
│       └── 安全性与对抗性评测
│
├── 5. 通用型Agent工具使用
│   ├── 5.1 ClaudeCode
│   │   ├── 基本使用与配置
│   │   ├── 上下文工程
│   │   ├── Skill系统
│   │   ├── MCP协议集成
│   │   └── 高效使用模式
│   ├── 5.2 Cursor
│   │   ├── AI辅助编程工作流
│   │   ├── 代码补全与重构
│   │   └── 多文件编辑与理解
│   ├── 5.3 Codex / OpenAI Codex
│   │   ├── 代码生成与理解
│   │   └── 集成开发流程
│   ├── 5.4 OpenClaw
│   │   ├── 开源Agent框架使用
│   │   ├── 工具集成与扩展
│   │   └── 自定义Agent开发
│   └── 5.5 生产级Agent工作流
│       ├── 全周期开发实践
│       ├── Agent协作模式
│       └── 效率最大化策略
│
├── 6. 智能体Harness搭建
│   ├── 6.1 Harness基础概念
│   │   ├── Harness定义与作用
│   │   ├── 上下文工程（Context Engineering）
│   │   │   ├── 上下文窗口管理
│   │   │   ├── 上下文压缩与摘要
│   │   │   ├── 上下文优先级排序
│   │   │   └── 动态上下文组装
│   │   ├── Skill系统
│   │   │   ├── Skill定义与注册
│   │   │   ├── Skill加载与卸载
│   │   │   └── Skill组合与编排
│   │   ├── ReAct范式
│   │   │   ├── Reasoning + Acting循环
│   │   │   ├── Thought-Action-Observation
│   │   │   └── ReACT的变体与改进
│   │   └── MCP（Model Context Protocol）
│   │       ├── 协议规范
│   │       ├── Tool/Resource/Prompt定义
│   │       ├── Server与Client架构
│   │       └── 传输层（stdio/SSE/Streamable HTTP）
│   ├── 6.2 主流Harness框架
│   │   ├── OpenClaw
│   │   │   ├── 架构设计
│   │   │   ├── 工具基建
│   │   │   ├── Runtime Loop实现
│   │   │   ├── 上下文管理
│   │   │   └── 记忆管理
│   │   ├── Hermes
│   │   │   ├── 架构设计
│   │   │   ├── 多智能体支持
│   │   │   └── 异步通信机制
│   │   ├── LangGraph
│   │   │   ├── 图结构Agent编排
│   │   │   ├── 状态管理
│   │   │   └── 条件路由
│   │   └── 其他框架
│   │       ├── AutoGPT / BabyAGI
│   │       ├── CrewAI
│   │       └── MetaGPT
│   ├── 6.3 Harness核心组件实现
│   │   ├── Runtime Loop（运行时循环）
│   │   │   ├── 主循环设计
│   │   │   ├── 中断与恢复机制
│   │   │   └── 错误处理与重试
│   │   ├── 工具基建（Tool Infrastructure）
│   │   │   ├── 工具注册与发现
│   │   │   ├── 工具执行引擎
│   │   │   ├── 工具权限控制
│   │   │   └── 工具结果处理
│   │   ├── 上下文管理器
│   │   │   ├── 上下文窗口监控
│   │   │   ├── 上下文压缩策略
│   │   │   └── 上下文持久化
│   │   └── 记忆管理器
│   │       ├── 记忆读写接口
│   │       ├── 记忆索引与检索
│   │       └── 记忆生命周期管理
│   ├── 6.4 高阶特性
│   │   ├── 多智能体同步机制
│   │   │   ├── 共享状态管理
│   │   │   ├── 同步屏障与协调
│   │   │   └── 冲突解决
│   │   ├── 多智能体异步机制
│   │   │   ├── 消息队列与事件驱动
│   │   │   ├── 异步任务调度
│   │   │   └── 结果聚合
│   │   └── 自演化（Self-evolution）
│   │       ├── 自我改进循环
│   │       ├── 经验积累与复用
│   │       └── 在线学习与适应
│   └── 6.5 Harness二次开发与独立搭建
│       ├── 基于已有框架的二次开发
│       ├── 从零搭建Harness的架构设计
│       ├── 模块化与可扩展性设计
│       └── 性能优化与生产化
│
└── 7. 沙箱环境工程
    ├── 7.1 沙箱基础
    │   ├── 沙箱概念与作用
    │   ├── 隔离级别与安全模型
    │   └── 常见沙箱技术对比
    ├── 7.2 单实例沙箱搭建
    │   ├── Docker容器沙箱
    │   │   ├── Dockerfile编写
    │   │   ├── 容器启停管理
    │   │   ├── 依赖管理
    │   │   └── 基础配置
    │   ├── 进程级沙箱
    │   │   ├── seccomp / AppArmor
    │   │   ├── namespace隔离
    │   │   └── cgroup资源限制
    │   └── 故障排查
    │       ├── 依赖缺失诊断
    │       ├── 权限不足排查
    │       └── 网络问题排查
    ├── 7.3 多实例隔离沙箱
    │   ├── 容器编排
    │   │   ├── Docker Compose多实例
    │   │   ├── Kubernetes Pod隔离
    │   │   └── 容器网络隔离
    │   ├── 精细化权限管控
    │   │   ├── RBAC权限模型
    │   │   ├── 文件系统权限
    │   │   └── 网络策略
    │   ├── 资源管控
    │   │   ├── CPU/内存限制
    │   │   ├── 磁盘IO限制
    │   │   ├── 网络带宽限制
    │   │   └── 资源监控与告警
    │   └── 跨实例通信
    │       ├── 共享存储
    │       ├── 消息队列
    │       └── gRPC/REST API
    ├── 7.4 高级沙箱工程
    │   ├── 高并发沙箱运行
    │   │   ├── 并发调度策略
    │   │   ├── 资源池管理
    │   │   ├── 负载均衡
    │   │   └── 弹性伸缩
    │   ├── 动态隔离技术
    │   │   ├── 微VM（Firecracker / Kata）
    │   │   ├── gVisor沙箱
    │   │   └── WASM沙箱
    │   └── 安全防护
    │       ├── 恶意代码注入防护
    │       ├── 代码执行审计
    │       ├── 沙箱逃逸检测
    │       └── 安全基线与合规
    └── 7.5 可复用框架与规范
        ├── 沙箱即服务（Sandbox-as-a-Service）
        ├── 自动化部署管线
        ├── 监控与可观测性
        └── 最佳实践与规范输出
```

---

## 3. 分领域详细知识点梳理

### 3.1 智能体记忆能力

#### 3.1.1 基础概念

**序列长度与上下文窗口**
- 大模型支持的序列长度演进：4K → 8K → 32K → 128K → 1M+（如 Gemini 1.5 Pro 的 2M token）
- 上下文窗口的物理限制与计算开销（Attention的O(n²)复杂度）
- 长上下文模型的技术方案：稀疏注意力、滑动窗口注意力、Ring Attention等
- 主流模型序列长度对比：GPT-4 Turbo (128K)、Claude 3 (200K)、Gemini 1.5 Pro (2M)、Llama 3.1 (128K)

**记忆对智能体的意义**
- 突破上下文窗口限制，实现跨会话信息保持
- 支持长程任务执行（需要记住之前步骤的结果和决策）
- 个性化交互（记住用户偏好和历史交互）
- 经验积累与学习（从过往交互中提取可复用知识）
- 记忆是智能体从"无状态文本生成器"进化为"有状态自主代理"的关键

**记忆类型分类（借鉴认知科学）**
- **短期记忆（Short-term Memory）**：当前上下文窗口内的信息，包括当前对话历史、当前任务状态
- **长期记忆（Long-term Memory）**：持久化存储的信息，跨会话保持
  - **语义记忆（Semantic Memory）**：事实知识、概念关系（如"用户是Python开发者"）
  - **情景记忆（Episodic Memory）**：具体事件和经历（如"上次帮用户调试了Docker问题"）
  - **程序记忆（Procedural Memory）**：技能和操作流程（如"如何调用某个API"）
- **工作记忆（Working Memory）**：当前任务执行中的中间状态和临时信息

#### 3.1.2 长短期记忆构建方法

**短期记忆（上下文窗口管理）**
- **滑动窗口策略**：保留最近N轮对话，丢弃旧消息
- **注意力机制与上下文压缩**：
  - LLM-based摘要压缩（将旧对话压缩为摘要）
  - 结构化信息提取（提取关键实体、决策点）
  - 递归摘要（层次化压缩长对话）
- **Token预算分配**：在系统提示、对话历史、工具结果之间分配有限的token预算
- **上下文窗口分段管理**：将上下文分为系统区、记忆区、对话区、工具区

**长期记忆（外部存储）**
- **向量数据库检索（RAG）**：
  - 向量化（Embedding）：将记忆条目编码为向量
  - 存储：Pinecone / Weaviate / Chroma / FAISS / Milvus
  - 检索：相似度搜索（cosine similarity）、混合检索（向量+关键词）
  - 重排序：Cross-encoder重排序提升检索精度
- **知识图谱存储**：
  - 实体-关系-实体三元组存储
  - 图数据库：Neo4j / NebulaGraph
  - 图遍历与推理
- **结构化数据库存储**：
  - 关系型数据库存储结构化记忆
  - 键值存储（Redis）用于快速记忆读写
- **混合存储架构**：
  - 向量数据库 + 知识图谱 + 结构化数据库的协同
  - 记忆路由：根据查询类型选择合适的存储后端

**工作记忆（中间状态管理）**
- 任务执行中间状态跟踪
- 多步骤推理的中间结果缓存
- 工具调用结果的临时存储与整合

#### 3.1.3 记忆Refine方法

**记忆摘要与压缩**
- LLM-based摘要：使用大模型对长记忆条目生成摘要
- 渐进式压缩：分层摘要，从细粒度到粗粒度
- 信息密度优化：保留高信息密度内容，丢弃冗余

**记忆合并与去重**
- 语义相似度去重：基于embedding相似度合并重复记忆
- 冲突检测与解决：当新记忆与旧记忆矛盾时的更新策略
- 时间衰减合并：旧记忆逐渐合并为新记忆

**记忆重要性评分与遗忘机制**
- 重要性评分模型：基于频率、时效性、相关性打分
- Ebbinghaus遗忘曲线启发的时间衰减
- 主动遗忘策略：低分记忆定期清理
- 记忆巩固（Memory Consolidation）：短期记忆转化为长期记忆的过程

**记忆检索排序优化**
- 多路召回 + 重排序
- 时间衰减因子 + 相关性因子 + 重要性因子加权
- 上下文感知检索：根据当前任务调整检索策略

#### 3.1.4 记忆加载方法

**全量加载 vs 选择性加载**
- 全量加载：将所有记忆注入上下文（仅适用于记忆量小的情况）
- 选择性加载：根据当前任务/查询检索相关记忆
- 混合策略：核心记忆全量加载 + 辅助记忆按需检索

**检索增强加载（RAG-based）**
- 查询构建：将当前上下文转化为记忆检索查询
- 多跳检索：链式检索关联记忆
- 时间感知检索：考虑记忆的时间属性

**上下文窗口注入策略**
- 记忆注入位置：系统提示后 / 对话历史前 / 每轮对话后
- 记忆格式化：结构化模板 vs 自然语言描述
- 记忆容量控制：注入记忆的token预算管理

**动态记忆加载与卸载**
- 根据任务进展动态调整加载的记忆集
- 记忆预加载：预测下一步可能需要的记忆
- 记忆卸载：任务完成后释放不再需要的记忆

#### 3.1.5 个性化记忆与经验记忆

**用户画像建模**
- 显式偏好收集：直接询问用户偏好
- 隐式偏好推断：从交互行为中推断偏好
- 画像更新与演化：随时间更新用户画像

**偏好学习与记忆**
- 偏好记忆的存储与检索
- 偏好冲突解决
- 偏好迁移（跨任务/跨领域）

**经验复用与迁移**
- 成功经验模板化：将成功的任务解决方案模板化
- 失败经验记录：记录失败原因和避免策略
- 经验迁移：将一个领域的经验应用到类似领域

**长时任务中的记忆管理**
- 长程任务的状态追踪与记忆管理
- 任务中断与恢复的记忆持久化
- 多任务并行时的记忆隔离与共享

---

### 3.2 智能体轨迹数据合成与微调训练

#### 3.2.1 大模型训练流程基础

**预训练（Pre-training）**
- 自回归语言建模（Next-token prediction）
- 训练数据规模与质量（万亿级token）
- 训练基础设施：数据并行、张量并行、流水线并行
- 训练超参：学习率、batch size、warmup步数
- Scaling Laws：模型大小、数据量、计算量的关系

**退火（Annealing / Cooldown）**
- 训练后期学习率衰减策略
- 高质量数据在退火阶段的重要性
- 退火对模型性能的提升机制
- 数据混合策略在退火阶段的变化

**持续预训练（CPT, Continual Pre-Training）**
- 在已有基模上继续预训练
- 领域适配的CPT
- 灾难性遗忘问题与缓解
- CPT数据配比策略

**监督微调（SFT, Supervised Fine-Tuning）**
- 指令跟随数据格式（Instruction-Response对）
- 多轮对话SFT
- 工具调用SFT（Function Calling格式）
- SFT vs Prefix Tuning vs LoRA
- SFT的损失函数设计：对响应部分计算loss，对提示部分mask

**对齐训练（RLHF/DPO等）**
- RLHF三阶段：SFT → Reward Model → PPO
- DPO：直接偏好优化，无需显式奖励模型
- RLAIF：AI反馈替代人类反馈
- 对齐的目标：有用性、无害性、诚实性

#### 3.2.2 智能体轨迹数据合成

**轨迹数据定义与结构**
- **Thought-Action-Observation三元组**：
  - Thought：模型的推理过程
  - Action：模型执行的动作（如工具调用）
  - Observation：环境返回的观察结果
- **多轮对话轨迹格式**：
  - System Prompt → User → Assistant (Thought + Action) → Tool Result → Assistant (Thought + Action) → ...
- **工具调用轨迹格式**：
  - 函数名、参数、返回值的结构化记录
  - 多工具组合调用的轨迹
  - 错误处理与重试的轨迹

**轨迹合成方法**
- **专家示范合成**：
  - 人工编写高质量Agent轨迹
  - 专家标注的质量控制
  - 成本高、规模有限
- **模型自生成（Self-play / Self-instruct）**：
  - 强模型生成轨迹（如GPT-4生成Agent轨迹）
  - Self-instruct：模型自己生成指令和响应
  - Self-play：两个模型对弈生成轨迹
  - Evol-Instruct：逐步复杂化指令
- **蒸馏合成（强模型→弱模型）**：
  - 使用强模型（如GPT-4）生成轨迹
  - 蒸馏到目标模型规模
  - 蒸馏数据的多样性保证
- **回溯合成（Backtracking）**：
  - 从任务目标出发，回溯生成合理轨迹
  - 反向推理生成轨迹
- **混合合成策略**：
  - 多种方法组合使用
  - 数据来源多样化
  - 质量交叉验证

**模仿学习的局限性**
- **分布偏移问题**：合成数据的分布与真实部署环境不一致
- **错误累积与级联失败**：长轨迹中一个错误步骤导致后续全错
- **探索不足**：模仿学习只学习专家行为，缺乏探索
- **长程任务泛化困难**：短轨迹合成方法难以泛化到长程任务
- **动作空间覆盖不全**：合成数据难以覆盖所有可能的工具组合

#### 3.2.3 轨迹数据质量筛选

**Turn-level质量筛选**
- **单步质量评分**：对轨迹中每一步的质量打分
  - 推理合理性评分
  - 动作正确性评分
  - 工具调用参数正确性
- **动作合法性检查**：验证每步动作是否在合法动作空间内
- **工具调用正确性验证**：
  - 函数名是否正确
  - 参数类型和值是否合法
  - 返回值处理是否正确

**步骤级（Step-level）质量筛选**
- **过程奖励模型（PRM, Process Reward Model）**：
  - 对每一步打分而非仅对最终结果打分
  - PRM训练数据收集与标注
  - PRM在轨迹筛选中的应用
- **蒙特卡洛树搜索评估**：
  - 从某一步开始展开多个分支
  - 评估该步的长期价值
- **轨迹一致性检查**：
  - 前后步骤的逻辑一致性
  - 工具调用与推理的一致性

**数据Curation策略**
- **去重与多样性保证**：
  - 语义去重（embedding相似度）
  - 精确去重（hash去重）
  - 多样性采样（聚类后按比例采样）
- **难度分级**：
  - 基于任务复杂度分级（工具数量、步骤数量、推理深度）
  - 基于模型成功率分级
- **课程学习排序**：
  - 由易到难的数据排列
  - 动态难度调整
- **数据配比与混合**：
  - 不同类型轨迹的比例控制
  - 与通用SFT数据的混合策略
  - 领域平衡

#### 3.2.4 SFT训练实践

**SFT数据格式设计**
- Agent轨迹的标准化格式设计
- System Prompt设计
- 工具定义格式（JSON Schema / 函数签名）
- 多轮对话的JSONL格式

**多轮对话SFT**
- 多轮对话的损失计算（仅对Assistant轮计算loss）
- 长对话的截断策略
- 工具结果轮的特殊处理

**工具调用SFT**
- 函数调用格式的SFT
- 多函数组合调用的SFT
- 错误处理轨迹的SFT

**损失函数设计（Mask策略）**
- Prompt部分mask（不计算loss）
- Response部分计算loss
- 工具返回结果部分的处理策略
- 多轮对话中各轮的mask策略

**训练超参调优**
- 学习率选择（通常比预训练小1-2个数量级）
- Epoch数选择（通常2-5个epoch）
- Batch size与梯度累积
- Warmup比例
- Weight decay
- 学习率调度器选择

#### 3.2.5 大规模轨迹合成（Scaling）

**10B+ token级合成策略**
- 自动化合成管线设计
- 多模型协同生成
- 合成质量控制管线
- 成本优化（使用小模型做初筛，大模型做精筛）

**自动化合成管线设计**
- 任务模板自动生成
- 轨迹自动执行与收集
- 质量自动评估与筛选
- 管线监控与异常处理

**合成数据质量监控**
- 统计指标监控（轨迹长度分布、工具使用分布等）
- 语义质量监控
- 下游任务性能监控

**工业级规模化的挑战**
- 计算资源需求
- 存储与IO瓶颈
- 数据版本管理
- 合成数据的分布漂移

---

### 3.3 智能体强化学习

#### 3.3.1 强化学习基础

**MDP与POMDP**
- 马尔可夫决策过程（MDP）：状态S、动作A、转移P、奖励R、折扣γ
- 部分可观测马尔可夫决策过程（POMDP）：智能体只能观察到部分状态
- Agent RL中的POMDP特性：环境状态部分可观测（如网页内容、API返回）

**策略梯度方法**
- REINFORCE算法
- 策略梯度定理
- 基线减少方差
- 自然策略梯度

**价值函数估计**
- 状态价值函数V(s)
- 动作价值函数Q(s,a)
- 优势函数A(s,a)
- TD学习与蒙特卡洛估计

**探索与利用**
- ε-greedy策略
- 熵正则化鼓励探索
- 好奇心驱动探索
- 在LLM RL中的探索策略

#### 3.3.2 核心RL算法

**PPO（Proximal Policy Optimization）**
- 裁剪目标函数（Clipped Surrogate Objective）：限制策略更新幅度
- GAE（Generalized Advantage Estimation）：偏差-方差权衡的优势估计
- PPO在LLM中的应用：
  - RLHF中的PPO：SFT模型 → Reward Model → PPO优化
  - Rollout生成与批量更新
  - KL散度约束防止偏离参考模型
  - PPO的超参：clip ratio (0.2)、GAE λ (0.95)、KL coefficient

**GRPO（Group Relative Policy Optimization）**
- 组内相对优势估计：
  - 对同一问题生成N个回答
  - 用组内奖励的均值和标准差归一化
  - 去除Critic网络，降低计算成本
- DeepSeek-R1中的应用：
  - GRPO驱动的推理能力涌现
  - 规则化奖励 vs 模型化奖励
  - GRPO在数学推理和代码生成中的效果
- GRPO vs PPO对比：
  - 计算效率：GRPO无需训练Critic
  - 稳定性：组内归一化提供更稳定的梯度
  - 适用场景

**OPD（Optimal Policy Distillation）**
- 策略蒸馏原理：将强策略的知识蒸馏到弱策略
- 在线蒸馏：强策略在线生成数据，弱策略学习
- 离线蒸馏：使用预收集的强策略数据
- 在Agent RL中的应用：用强模型生成的轨迹蒸馏到目标模型

**其他相关算法**
- **DPO（Direct Preference Optimization）**：
  - 直接从偏好数据学习，无需显式奖励模型
  - DPO的损失函数
  - DPO在Agent对齐中的应用
- **RLAIF / RLHF**：
  - 人类反馈 vs AI反馈
  - 奖励模型训练
  - 偏好数据收集
- **ReST（Reinforced Self-Training）**：
  - 自训练与强化学习结合
  - 迭代式自我改进
- **RFT（Rejection Sampling Fine-Tuning）**：
  - 拒绝采样：只保留高质量生成
  - 采样-筛选-微调循环

#### 3.3.3 智能体RL特有挑战

**Agent RL vs Reasoning RL的差异化**
- **动作空间复杂性**：
  - Reasoning RL：动作=生成token（连续文本空间）
  - Agent RL：动作=工具调用（结构化、离散、组合爆炸）
  - 工具调用的参数空间巨大
- **环境交互的不可逆性**：
  - Reasoning RL：推理过程可回溯
  - Agent RL：工具调用可能有副作用（如发送邮件、删除文件）
  - 不可逆动作的信用分配
- **延迟奖励与稀疏奖励**：
  - Agent任务通常只在最终给出奖励
  - 中间步骤的奖励设计困难
  - 长程任务的信用分配问题
- **长程轨迹的信用分配**：
  - 几十甚至上百步的轨迹
  - 哪些步骤对最终结果贡献最大
  - 分层强化学习方法

**黑盒强化学习**
- 环境不可微的挑战：
  - 工具调用环境是黑盒（如浏览器、数据库）
  - 无法通过环境反向传播梯度
- 黑盒优化方法：
  - 进化策略（Evolution Strategies）
  - 贝叶斯优化
  - CMA-ES
- 代理模型方法：
  - 训练环境代理模型
  - 在代理模型上做梯度优化
  - 代理模型与真实环境的对齐

**长程轨迹训练稳定性**
- 梯度消失/爆炸：
  - 长轨迹的梯度传播困难
  - 截断反向传播（Truncated BPTT）
- 训练崩溃与恢复：
  - 策略突然退化
  - 检测与恢复机制
- 课程学习策略：
  - 从短轨迹到长轨迹
  - 从简单任务到复杂任务
  - 动态课程调整

#### 3.3.4 Task合成技术

**任务自动生成**
- 模板化任务生成
- LLM驱动的任务生成
- 任务参数空间采样
- 组合任务生成（多工具组合）

**任务难度分级**
- 基于工具数量：1工具 → 2工具 → 多工具组合
- 基于步骤数量：1步 → 5步 → 20步+
- 基于推理深度：直接调用 → 需要规划 → 需要回溯
- 基于环境复杂度：静态环境 → 动态环境

**任务多样性保证**
- 任务类型覆盖（搜索、代码、数据分析、多模态等）
- 工具覆盖矩阵
- 语义多样性（embedding聚类）

**课程学习任务编排**
- 前置任务依赖分析
- 难度递进曲线设计
- 自适应课程（根据训练进展调整）

#### 3.3.5 环境模拟与沙箱构建

**环境接口设计**
- 标准化环境接口（类似OpenAI Gym接口）
- 观察空间定义
- 动作空间定义
- 奖励函数接口

**状态空间建模**
- 环境状态的表示
- 部分可观测状态的处理
- 状态序列的编码

**奖励函数设计**
- 结果奖励（Outcome Reward）：任务完成与否
- 过程奖励（Process Reward）：中间步骤的质量
- 奖励塑形（Reward Shaping）：引导学习
- 多目标奖励的加权

**环境并行化**
- 多环境实例并行运行
- 异步环境（Async environments）
- 向量化环境（Vectorized environments）
- 环境状态同步

**模拟器与真实环境对齐**
- 模拟环境与真实环境的差异（Sim-to-Real Gap）
- 域随机化（Domain Randomization）
- 环境多样性训练

#### 3.3.6 训练超参与调优

**KL散度约束（KL coefficient）**
- KL散度的作用：防止策略偏离参考模型太远
- KL系数的选择：太小→偏离，太大→学不动
- 自适应KL系数
- KL发散的监控

**熵正则化（Entropy coefficient）**
- 熵正则化的作用：鼓励探索，防止策略过早收敛
- 熵系数的选择
- 熵的监控与调整

**学习率调度**
- 学习率warmup
- 余弦退火
- 线性衰减
- 学习率与batch size的关系

**批大小与梯度累积**
- 大batch训练的优势与挑战
- 梯度累积模拟大batch
- micro-batch与global batch

**Rollout batch size**
- Rollout并行度
- GPU利用率优化
- Rollout与训练的流水线

**超参搜索策略**
- 网格搜索
- 贝叶斯优化
- 人口训练（Population Based Training）
- 超参重要性分析

---

### 3.4 智能体评估

#### 3.4.1 评估基础理论

**评估的作用与意义**
- 模型选择与比较：选择最适合Agent任务的基模
- 训练效果验证：验证SFT/RL训练后的提升
- 能力边界探测：发现模型的能力短板
- 真实体验预测：评估结果与真实用户体验的关联

**评估维度分类**
- **工具使用能力**：能否正确选择和调用工具
- **推理能力**：能否合理推理出下一步行动
- **多轮对话能力**：能否在多轮交互中保持一致性
- **长程任务完成能力**：能否完成需要多步骤的复杂任务
- **安全性与鲁棒性**：面对恶意输入和异常情况的表现

**评估方法论**
- **静态评估 vs 动态评估**：
  - 静态：固定测试集，固定答案
  - 动态：环境实时响应，答案不固定
- **离线评估 vs 在线评估**：
  - 离线：在测试集上评估
  - 在线：在真实使用中评估（A/B测试）
- **人工评估 vs 自动评估**：
  - 人工：人类专家打分
  - 自动：规则匹配 / LLM-as-Judge / 程序化验证

#### 3.4.2 主流评测榜单

**BFCL（Berkeley Function Calling Leaderboard）**
- 来源：UC Berkeley
- 评估维度：
  - 简单函数调用
  - 多函数调用
  - 并行函数调用
  - 函数相关性判断
  - 幻觉检测（调用不存在的函数）
- 评测方法：精确匹配 + AST匹配
- 局限性：静态测试集，不评估多轮交互

**TauBench**
- 来源：Sierra AI
- 评估维度：
  - 用户模拟（模拟真实用户与Agent交互）
  - 多轮工具调用
  - 任务完成率
  - 对话效率
- 评测方法：模拟用户+评判员双角色
- 与真实体验的关联：通过用户模拟更接近真实场景

**Toolathlon**
- 来源：清华等
- 评估维度：
  - 多工具组合任务
  - 工具链式调用
  - 复杂推理+工具使用
- 评估指标：任务完成率、步骤效率

**BrowseComp**
- 来源：OpenAI
- 评估维度：
  - 浏览器操作能力
  - 网页信息检索与推理
  - 多步浏览+信息整合
- 评估方法：真实浏览器环境中的任务完成

**Claw-Eval**
- 评估维度：
  - 综合Agent能力评估
  - 动态环境评测
  - 长程任务评测
- 特点：支持动态环境，更接近真实部署

**其他重要榜单**
- **AgentBench**：多场景Agent能力评估（OS、DB、Web、KG等8个场景）
- **SWE-bench**：软件工程任务评估（真实GitHub Issue修复）
- **WebArena**：网页交互任务评估
- **GAIA**：通用AI助手评估（需要多步推理+工具使用）
- **ToolBench**：大规模工具使用评估（16K+ API）
- **MLE-bench**：机器学习工程能力评估
- **AndroidWorld**：移动端Agent评估

#### 3.4.3 评估局限性分析

**静态榜单的过拟合风险**
- 模型可能在特定榜单上过拟合
- 榜单数据可能被泄露到训练数据中
- 刷榜行为导致评估失真

**评估与真实体验的Gap**
- 榜单任务与真实用户任务的分布差异
- 评估环境过于理想化
- 真实场景的多样性和不可预测性

**评估覆盖面不足**
- 现有榜单主要覆盖英文场景
- 长程任务评估不足
- 多模态Agent评估不足
- 安全性评估不足

**数据污染问题**
- 评测数据泄露到训练集
- 模型记忆评测答案
- 数据污染检测方法

#### 3.4.4 高级评估方向

**动态环境评测**
- 环境实时变化，任务答案不固定
- 需要Agent实时适应
- 评估基础设施的挑战

**超长程任务评测**
- 几十甚至上百步的任务
- 长程信用分配评估
- 记忆与状态管理评估

**多智能体协作评测**
- 多Agent协作完成任务
- 通信效率评估
- 角色分工与协调

**安全性与对抗性评测**
- 恶意指令注入防护
- 越狱攻击鲁棒性
- 工具滥用防护
- 隐私保护评估

---

### 3.5 通用型Agent工具使用（ClaudeCode/OpenClaw等）

#### 3.5.1 ClaudeCode

**基本使用与配置**
- 安装与初始化
- 项目配置（CLAUDE.md / AGENTS.md）
- 模型选择与参数配置
- 工具权限配置

**上下文工程**
- 上下文窗口的高效利用
- 系统提示设计
- 上下文压缩与摘要
- 多文件上下文管理

**Skill系统**
- Skill定义与编写
- Skill加载与触发机制
- Skill组合与编排
- 自定义Skill开发

**MCP协议集成**
- MCP Server配置
- Tool/Resource/Prompt的使用
- 自定义MCP Server开发

**高效使用模式**
- 任务分解策略
- 多轮迭代开发
- 代码审查与重构
- 测试驱动开发

#### 3.5.2 Cursor

**AI辅助编程工作流**
- 代码补全（Tab补全）
- 内联编辑（Cmd+K）
- Chat模式
- Agent模式

**代码补全与重构**
- 多行补全
- 跨文件重构
- 代码解释

**多文件编辑与理解**
- 代码库索引
- @file / @folder引用
- 上下文感知编辑

#### 3.5.3 Codex / OpenAI Codex

**代码生成与理解**
- Codex CLI使用
- 代码生成能力
- 代码理解与解释

**集成开发流程**
- 与IDE集成
- 与CI/CD集成
- 代码审查集成

#### 3.5.4 OpenClaw

**开源Agent框架使用**
- 框架安装与配置
- Agent创建与配置
- 工具集成

**工具集成与扩展**
- 内置工具使用
- 自定义工具开发
- MCP工具集成

**自定义Agent开发**
- Agent行为定制
- 提示工程
- 工作流定制

#### 3.5.5 生产级Agent工作流

**全周期开发实践**
- 需求分析 → 设计 → 编码 → 测试 → 部署
- Agent辅助各阶段的具体实践
- 效率提升度量

**Agent协作模式**
- 多Agent分工协作
- Agent间通信
- 任务分发与结果聚合

**效率最大化策略**
- 上下文窗口优化
- 工具链优化
- 自动化流水线设计

---

### 3.6 面向基模的智能体Harness搭建

#### 3.6.1 Harness基础概念

**Harness定义与作用**
- Harness：包裹在大模型外部的"框架/线束"，将基模转化为可执行任务的智能体
- 作用：管理上下文、调度工具、控制执行循环、管理记忆
- Harness vs Agent Framework：Harness更底层，关注runtime和基础设施

**上下文工程（Context Engineering）**
- **上下文窗口管理**：
  - Token计数与预算管理
  - 上下文溢出处理策略
  - 上下文优先级队列
- **上下文压缩与摘要**：
  - 对话历史压缩
  - 工具结果压缩
  - 渐进式摘要
- **上下文优先级排序**：
  - 系统指令 > 当前任务 > 相关记忆 > 对话历史
  - 动态优先级调整
- **动态上下文组装**：
  - 根据任务阶段组装不同上下文
  - 上下文模板系统

**Skill系统**
- **Skill定义与注册**：Skill是Agent的能力单元，包含触发条件、执行逻辑、输出格式
- **Skill加载与卸载**：按需加载Skill，减少上下文占用
- **Skill组合与编排**：多Skill组合完成复杂任务

**ReAct范式**
- **Reasoning + Acting循环**：
  ```
  Thought: 我需要查询用户的订单状态
  Action: call_tool(order_lookup, {user_id: "123"})
  Observation: 订单状态为"已发货"
  Thought: 用户问的是物流信息，我需要进一步查询物流
  Action: call_tool(logistics_track, {order_id: "456"})
  Observation: 预计明天到达
  Answer: 您的订单已发货，预计明天到达。
  ```
- **Thought-Action-Observation**：ReAct的核心三元组
- **ReACT的变体与改进**：
  - Reflexion：在ReAct基础上增加自我反思
  - Plan-and-Execute：先规划再执行
  - Tree of Thoughts：树形推理

**MCP（Model Context Protocol）**
- **协议规范**：
  - JSON-RPC 2.0基础
  - Tool / Resource / Prompt 三种能力类型
  - 能力发现与协商
- **Tool/Resource/Prompt定义**：
  - Tool：可执行的函数（名称、描述、参数schema、执行handler）
  - Resource：可读取的数据源（文件、数据库、API）
  - Prompt：预定义的提示模板
- **Server与Client架构**：
  - MCP Server：提供能力（工具、资源、提示）
  - MCP Client：消费能力（通常是Agent/Harness）
  - 多Server多Client架构
- **传输层**：
  - stdio：标准输入输出（本地进程通信）
  - SSE（Server-Sent Events）：单向流式
  - Streamable HTTP：双向流式HTTP

#### 3.6.2 主流Harness框架

**OpenClaw**
- **架构设计**：
  - 模块化组件设计
  - 插件化工具系统
  - 可配置的runtime loop
- **工具基建**：
  - 工具注册中心
  - 工具执行引擎
  - 工具结果处理管线
- **Runtime Loop实现**：
  - 主循环：感知 → 推理 → 行动 → 观察
  - 中断处理
  - 超时管理
- **上下文管理**：
  - 上下文窗口监控
  - 自动压缩策略
  - 上下文持久化
- **记忆管理**：
  - 短期记忆：对话历史
  - 长期记忆：外部存储
  - 记忆检索与注入

**Hermes**
- **架构设计**：
  - 多智能体支持
  - 消息总线架构
- **多智能体支持**：
  - Agent角色定义
  - Agent间通信协议
  - 任务分发与协调
- **异步通信机制**：
  - 异步消息队列
  - 事件驱动架构
  - 回调机制

**LangGraph**
- **图结构Agent编排**：
  - 节点（Node）：Agent或工具
  - 边（Edge）：控制流
  - 条件边：基于状态的路由
- **状态管理**：
  - 全局状态对象
  - 状态更新与传播
  - 状态持久化与恢复
- **条件路由**：
  - 基于状态的条件分支
  - 循环与跳转
  - 并行执行

**其他框架**
- **AutoGPT / BabyAGI**：早期自主Agent框架，展示了自主任务分解和执行的概念
- **CrewAI**：多Agent协作框架，角色分工明确
- **MetaGPT**：模拟软件团队的多Agent框架

#### 3.6.3 Harness核心组件实现

**Runtime Loop（运行时循环）**
- **主循环设计**：
  ```python
  while not task_complete and not timeout:
      context = assemble_context(task, memory, history)
      response = llm.generate(context)
      if response.has_action:
          result = execute_tool(response.action)
          update_memory(result)
          update_history(response, result)
      else:
          task_complete = True
  ```
- **中断与恢复机制**：
  - 保存执行状态
  - 从断点恢复
  - 异常处理与重试
- **错误处理与重试**：
  - 工具执行失败的重试策略
  - LLM调用失败的处理
  - 超时处理

**工具基建（Tool Infrastructure）**
- **工具注册与发现**：
  - 工具元数据管理（名称、描述、参数schema）
  - 工具自动发现（扫描注册）
  - 工具版本管理
- **工具执行引擎**：
  - 同步执行 vs 异步执行
  - 超时控制
  - 结果格式化
- **工具权限控制**：
  - 工具白名单/黑名单
  - 参数验证与沙箱化
  - 敏感操作确认
- **工具结果处理**：
  - 结果截断（长结果处理）
  - 结果格式化
  - 错误信息处理

**上下文管理器**
- **上下文窗口监控**：实时Token计数
- **上下文压缩策略**：
  - 对话历史摘要
  - 工具结果压缩
  - 旧消息丢弃策略
- **上下文持久化**：跨会话上下文保存与恢复

**记忆管理器**
- **记忆读写接口**：统一的记忆操作API
- **记忆索引与检索**：向量索引、关键词索引
- **记忆生命周期管理**：创建 → 更新 → 遗忘

#### 3.6.4 高阶特性

**多智能体同步机制**
- **共享状态管理**：多Agent共享的全局状态
- **同步屏障与协调**：等待所有Agent完成某阶段
- **冲突解决**：多Agent产生冲突结果时的解决策略

**多智能体异步机制**
- **消息队列与事件驱动**：Agent间通过消息队列通信
- **异步任务调度**：任务队列、优先级调度
- **结果聚合**：收集多Agent结果并聚合

**自演化（Self-evolution）**
- **自我改进循环**：
  - 执行任务 → 评估结果 → 提取经验 → 更新策略 → 再次执行
- **经验积累与复用**：
  - 成功经验模板化
  - 失败经验避坑
- **在线学习与适应**：
  - 运行时策略调整
  - 工具使用模式优化
  - 上下文管理策略优化

#### 3.6.5 Harness二次开发与独立搭建

**基于已有框架的二次开发**
- 扩展OpenClaw/Hermes的插件系统
- 自定义工具开发与集成
- 自定义记忆后端实现
- 自定义评估指标集成

**从零搭建Harness的架构设计**
- 分层架构设计：
  - 基础层：LLM接口、工具执行引擎、存储后端
  - 核心层：Runtime Loop、上下文管理、记忆管理
  - 应用层：任务编排、多Agent协调、自演化
- 模块化设计原则
- 接口抽象与依赖注入

**模块化与可扩展性设计**
- 插件化架构
- 配置驱动设计
- 热加载与热更新

**性能优化与生产化**
- 异步IO优化
- 缓存策略
- 并发控制
- 监控与日志
- 错误恢复与容错

---

### 3.7 沙箱环境工程

#### 3.7.1 沙箱基础

**沙箱概念与作用**
- 沙箱：隔离的代码执行环境，限制程序对系统资源的访问
- 在Agent中的作用：
  - 安全执行Agent生成的代码
  - 隔离不同Agent的执行环境
  - 防止恶意代码影响宿主系统
  - 提供可复现的执行环境

**隔离级别与安全模型**
- 进程级隔离（最低）：seccomp、namespace
- 容器级隔离（中等）：Docker、Podman
- 微VM级隔离（高）：Firecracker、Kata Containers
- VM级隔离（最高）：完整虚拟机

**常见沙箱技术对比**

| 技术 | 隔离级别 | 启动速度 | 资源开销 | 安全性 | 适用场景 |
|------|----------|----------|----------|--------|----------|
| Docker | 容器级 | 秒级 | 低 | 中 | 通用代码执行 |
| gVisor | 容器级 | 秒级 | 中 | 高 | 不信任代码执行 |
| Firecracker | 微VM | 毫秒级 | 中低 | 高 | 高并发沙箱 |
| Kata Containers | 微VM | 秒级 | 中 | 高 | K8s集成 |
| WASM | 进程级 | 毫秒级 | 极低 | 高 | 轻量代码执行 |
| nsjail | 进程级 | 毫秒级 | 极低 | 中 | 快速隔离 |

#### 3.7.2 单实例沙箱搭建

**Docker容器沙箱**
- **Dockerfile编写**：
  - 基础镜像选择（python:slim, node:alpine等）
  - 依赖安装（pip, npm, apt）
  - 环境变量配置
  - 非root用户运行
  - 健康检查配置
- **容器启停管理**：
  - docker run/stop/start命令
  - 容器生命周期管理脚本
  - 容器状态监控
- **依赖管理**：
  - 依赖预装 vs 运行时安装
  - 依赖版本锁定
  - 依赖冲突解决
- **基础配置**：
  - 网络配置（网络隔离、端口映射）
  - 存储配置（Volume挂载、tmpfs）
  - 资源限制（CPU、内存、磁盘）

**进程级沙箱**
- **seccomp / AppArmor**：
  - seccomp：系统调用过滤
  - AppArmor：基于路径的权限控制
  - 配置文件编写
- **namespace隔离**：
  - PID namespace：进程隔离
  - Network namespace：网络隔离
  - Mount namespace：文件系统隔离
  - User namespace：用户隔离
- **cgroup资源限制**：
  - CPU配额限制
  - 内存限制
  - 磁盘IO限制
  - 设备访问控制

**故障排查**
- **依赖缺失诊断**：检查容器内安装的包、路径配置
- **权限不足排查**：检查用户权限、文件权限、capability
- **网络问题排查**：检查网络配置、DNS、防火墙

#### 3.7.3 多实例隔离沙箱

**容器编排**
- **Docker Compose多实例**：
  - 多容器定义与编排
  - 服务间依赖管理
  - 网络配置
- **Kubernetes Pod隔离**：
  - Pod设计与部署
  - Namespace隔离
  - Service网络配置
  - ConfigMap与Secret管理
- **容器网络隔离**：
  - 每实例独立网络namespace
  - 网络策略（NetworkPolicy）
  - 出口流量控制

**精细化权限管控**
- **RBAC权限模型**：
  - Role / ClusterRole定义
  - RoleBinding / ClusterRoleBinding
  - ServiceAccount管理
- **文件系统权限**：
  - 只读根文件系统
  - 临时文件系统（tmpfs）
  - 文件权限掩码
- **网络策略**：
  - 入站/出站流量控制
  - 端口级访问控制
  - 域名级访问控制

**资源管控**
- **CPU/内存限制**：
  - Requests与Limits设置
  - 超限行为（OOM Kill、Throttle）
  - 资源配额管理
- **磁盘IO限制**：
  - IOPS限制
  - 带宽限制
  - 存储配额
- **网络带宽限制**：
  - 流量整形（Traffic Shaping）
  - 带宽配额
- **资源监控与告警**：
  - 实时资源使用监控
  - 阈值告警
  - 资源使用日志

**跨实例通信**
- **共享存储**：
  - NFS共享存储
  - 分布式文件系统
  - 对象存储（S3兼容）
- **消息队列**：
  - Redis / RabbitMQ / Kafka
  - 异步消息传递
- **gRPC/REST API**：
  - 实例间API调用
  - 服务发现与负载均衡

#### 3.7.4 高级沙箱工程

**高并发沙箱运行**
- **并发调度策略**：
  - 任务队列与工作池
  - 优先级调度
  - 公平调度
- **资源池管理**：
  - 预热沙箱池
  - 沙箱复用策略
  - 空闲沙箱回收
- **负载均衡**：
  - 基于资源利用率的负载均衡
  - 亲和性调度
  - 反亲和性（分散调度）
- **弹性伸缩**：
  - 自动扩缩容
  - 基于队列长度的伸缩策略
  - 预测性伸缩

**动态隔离技术**
- **微VM（Firecracker / Kata）**：
  - Firecracker：AWS开源的轻量级微VM
    - 极快启动（~125ms）
    - 极小内存开销（~5MB）
    - 适合高并发沙箱
  - Kata Containers：兼容K8s的微VM
    - 强隔离性
    - 与容器生态兼容
- **gVisor沙箱**：
  - Google开源的应用内核
  - 拦截系统调用
  - 提供容器级体验+VM级安全
- **WASM沙箱**：
  - WebAssembly作为沙箱运行时
  - 极快启动、极小开销
  - 适合函数级代码执行
  - WASI（WebAssembly System Interface）

**安全防护**
- **恶意代码注入防护**：
  - 输入消毒（Input Sanitization）
  - 代码静态分析
  - 危险API检测
- **代码执行审计**：
  - 执行日志记录
  - 系统调用追踪
  - 网络流量记录
- **沙箱逃逸检测**：
  - 异常行为检测
  - 容器逃逸防护
  - 内核漏洞利用防护
- **安全基线与合规**：
  - CIS Benchmark合规
  - 安全扫描（Trivy、Clair）
  - 镜像签名验证

#### 3.7.5 可复用框架与规范

**沙箱即服务（Sandbox-as-a-Service）**
- 统一沙箱API接口
- 多租户沙箱管理
- 沙箱生命周期管理服务

**自动化部署管线**
- Infrastructure as Code（Terraform / Pulumi）
- CI/CD集成
- 蓝绿部署与金丝雀发布

**监控与可观测性**
- 指标收集（Prometheus + Grafana）
- 日志聚合（ELK / Loki）
- 分布式追踪（Jaeger / OpenTelemetry）
- 告警系统

**最佳实践与规范输出**
- 沙箱配置规范
- 安全基线文档
- 运维手册
- 故障排查指南

---

## 4. 分领域学习资料规划

### 4.1 智能体记忆能力 — 学习资料

#### 4.1.1 基础理论

| 序号 | 资料名称 | 作者/来源 | 类型 | 难度 | 推荐理由 |
|------|----------|-----------|------|------|----------|
| 1 | 《大语言模型》第5章：使用大语言模型 | 赵鑫等 | 书籍 | 入门 | 系统介绍LLM基础，包含上下文窗口和prompt工程 |
| 2 | "Memory in Artificial Intelligence: A Survey" | 多位作者 | 论文综述 | 中级 | 全面综述AI记忆系统，涵盖记忆类型分类与构建方法 |
| 3 | "A Survey on the Memory Mechanism of Large Language Model based Agents" | Zeyu Zhang等 | 论文综述 | 中级 | 专门针对LLM Agent记忆机制的综述，覆盖短期/长期记忆、记忆操作 |
| 4 | "Cognitive Architectures for Language Agents"（CoALA） | S. Sumers等 | 论文 | 中级 | 提出了语言Agent的认知架构框架，将记忆作为核心组件 |
| 5 | 《认知心理学》相关章节 — 工作记忆与长时记忆 | 多种教材 | 书籍 | 入门 | 理解人类记忆系统的分类机制，为AI记忆设计提供认知科学基础 |

#### 4.1.2 记忆构建与Refine方法

| 序号 | 资料名称 | 作者/来源 | 类型 | 难度 | 推荐理由 |
|------|----------|-----------|------|------|----------|
| 6 | "MemGPT: Towards LLMs as Operating Systems" | C. Packer等 | 论文+开源项目 | 中级 | 提出了受OS虚拟内存启发的记忆管理方案，核心参考 |
| 7 | MemGPT / Letta 开源项目 | UC Berkeley | 开源项目 | 中高级 | MemGPT的工程实现，包含记忆分页、管理、检索的完整代码 |
| 8 | LangChain Memory模块文档与源码 | LangChain | 开源项目+文档 | 中级 | 工业级记忆系统实现，包含多种记忆类型（ConversationBufferMemory等） |
| 9 | "Generative Agents: Interactive Simulacra of Human Behavior" | J. Park等 | 论文+开源项目 | 中级 | Stanford小镇项目，展示了记忆流(Memory Stream)和反思(Reflection)机制 |
| 10 | "ExpeL: LLM Agents Are Experiential Learners" | A. Zhao等 | 论文 | 中高级 | 经验学习与记忆复用的Agent框架 |
| 11 | "Reflexion: Language Agents with Verbal Reinforcement Learning" | N. Shinn等 | 论文+开源项目 | 中级 | 语言反思机制，将失败经验转化为记忆 |
| 12 | "Voyager: An Open-Ended Embodied Agent with LLMs" | G. Wang等 | 论文+开源项目 | 中高级 | Minecraft中的技能库记忆与课程学习 |

#### 4.1.3 向量检索与RAG（长期记忆技术基础）

| 序号 | 资料名称 | 作者/来源 | 类型 | 难度 | 推荐理由 |
|------|----------|-----------|------|------|----------|
| 13 | "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" | Lewis等（Facebook） | 论文 | 中级 | RAG的开创性论文，理解检索增强生成的基础 |
| 14 | Pinecone / Weaviate / Chroma 官方文档与教程 | 各官方 | 文档+教程 | 入门-中级 | 主流向量数据库的使用，长期记忆的存储后端 |
| 15 | "Dense Passage Retrieval for Open-Domain QA" | Karpukhin等 | 论文 | 中级 | 密集检索的基础，理解向量化检索原理 |
| 16 | "Lost in the Middle: How Language Models Use Long Contexts" | Liu等 | 论文 | 中级 | 理解LLM在长上下文中的信息利用模式，指导记忆加载策略 |

#### 4.1.4 个性化与长时任务记忆

| 序号 | 资料名称 | 作者/来源 | 类型 | 难度 | 推荐理由 |
|------|----------|-----------|------|------|----------|
| 17 | "ChatDB: A Database Augmented LLM for Natural Language Interaction with Databases" | 多位作者 | 论文 | 中高级 | 数据库增强的记忆管理方案 |
| 18 | "Think Before You Speak: Training Language Models With Pause Tokens" | S. Goyal等 | 论文 | 中级 | 工作记忆的token级别实现 |
| 19 | "LongAgent: Scaling LLM Agents to 128K Tokens" | 多位作者 | 论文 | 高级 | 长上下文Agent的记忆管理方案 |

#### 4.1.5 实践项目建议

| 序号 | 项目名称 | 类型 | 难度 | 说明 |
|------|----------|------|------|------|
| P1 | 搭建带记忆的对话Agent | 实践项目 | 中级 | 使用LangChain Memory + 向量数据库实现多轮记忆 |
| P2 | 实现MemGPT风格的记忆分页系统 | 实践项目 | 高级 | 参考MemGPT实现记忆的加载/卸载/分页机制 |
| P3 | 构建经验记忆系统 | 实践项目 | 高级 | 实现任务执行→经验提取→记忆存储→经验复用的完整链路 |

---

### 4.2 智能体轨迹数据合成与微调训练 — 学习资料

#### 4.2.1 大模型训练流程基础

| 序号 | 资料名称 | 作者/来源 | 类型 | 难度 | 推荐理由 |
|------|----------|-----------|------|------|----------|
| 1 | 《大语言模型》第3-4章：模型训练 | 赵鑫等 | 书籍 | 入门-中级 | 系统介绍预训练、SFT等训练流程 |
| 2 | Stanford CS224N: Natural Language Processing with Deep Learning | Stanford | 大学课程 | 中级 | NLP与深度学习经典课程，包含Transformer和语言模型 |
| 3 | Stanford CS25: Transformers United | Stanford | 大学课程 | 中级 | 专题课程，深入Transformer架构和训练 |
| 4 | "LLM Data Pipelines: Unlocking the Foundation Model Era" | L. Tang等 | 书籍 | 中级 | 专门讲LLM数据管线，从数据收集到处理的全流程 |
| 5 | Hugging Face NLP Course（Chapter on Fine-tuning） | Hugging Face | 教程 | 入门-中级 | 实操导向的微调教程，包含SFT实践 |
| 6 | "Scaling Laws for Neural Language Models" | Kaplan等（OpenAI） | 论文 | 中级 | 理解模型规模、数据量、计算量的关系 |
| 7 | "Training Compute-Optimal Large Language Models"（Chinchilla） | Hoffmann等 | 论文 | 中级 | 修正了Scaling Laws，指导数据配比 |

#### 4.2.2 SFT训练实践

| 序号 | 资料名称 | 作者/来源 | 类型 | 难度 | 推荐理由 |
|------|----------|-----------|------|------|----------|
| 8 | "Fine-tuning Language Models from Human Preferences" | D. Ziegler等 | 论文 | 中级 | 从人类偏好微调的基础方法 |
| 9 | LLaMA-Factory 开源项目 | hiyouga | 开源项目 | 入门-中级 | 统一的大模型微调框架，支持SFT/DPO/RLHF等，上手快 |
| 10 | Axolotl 开源项目 | OpenAccess AI Collective | 开源项目 | 中级 | 高性能大模型微调工具，支持多种训练方法 |
| 11 | "LoRA: Low-Rank Adaptation of Large Language Models" | E. Hu等 | 论文+开源 | 中级 | 参数高效微调方法，降低SFT计算成本 |
| 12 | DeepSpeed-Chat 开源项目 | Microsoft | 开源项目 | 中高级 | 微软的RLHF/SFT训练框架，包含完整的三阶段训练流程 |
| 13 | "Direct Preference Optimization: Your Language Model is Secretly a Reward Model" | R. Rafailov等 | 论文 | 中级 | DPO方法论文，简化RLHF流程 |

#### 4.2.3 轨迹数据合成方法

| 序号 | 资料名称 | 作者/来源 | 类型 | 难度 | 推荐理由 |
|------|----------|-----------|------|------|----------|
| 14 | "Self-Instruct: Aligning Language Models with Self-Generated Instructions" | Y. Wang等 | 论文+开源 | 中级 | 自指令生成方法，轨迹合成的基础方法 |
| 15 | "Evol-Instruct: Evolving Complex Instructions for LLMs" | 多位作者 | 论文 | 中级 | 指令进化方法，逐步复杂化生成更难的任务 |
| 16 | "Toolformer: Language Models Can Teach Themselves to Use Tools" | T. Schick等 | 论文 | 中级 | 模型自主学习工具使用的方法，轨迹自动生成 |
| 17 | "Gorilla: Large Language Model Connected with Massive APIs" | S. Patil等 | 论文+开源 | 中级 | 大规模API使用训练，包含轨迹数据合成 |
| 18 | "API-Bank: A Comprehensive Benchmark for Tool-Augmented LLMs" | M. Li等 | 论文 | 中级 | API使用基准，包含轨迹数据格式 |
| 19 | "AgentTuning: Enabling Generalized Agent Abilities for LLMs" | A. Zeng等 | 论文+开源 | 中高级 | 面向Agent能力的微调方法，包含轨迹数据构建 |
| 20 | "FireAct: Toward Language Agent Fine-tuning" | B. Chen等 | 论文 | 中级 | 语言Agent微调方法，轨迹数据合成实践 |
| 21 | "Agent-FLAN: Designing Data and Methods of Effective Agent Tuning" | Z. Chen等 | 论文+开源 | 中高级 | Agent微调的数据设计与方法 |

#### 4.2.4 轨迹数据质量筛选

| 序号 | 资料名称 | 作者/来源 | 类型 | 难度 | 推荐理由 |
|------|----------|-----------|------|------|----------|
| 22 | "Let's Verify Step by Step" | OpenAI | 论文 | 中高级 | 过程奖励模型（PRM）的基础论文，步骤级质量评估 |
| 23 | "PRM800K: Process Reward Model Dataset" | OpenAI | 数据集 | 中高级 | PRM训练数据集，理解步骤级评分 |
| 24 | "Scaling Relationship Between Learning and Reasoning in LLMs" | 多位作者 | 论文 | 高级 | 学习与推理的Scaling关系，数据质量vs数量的权衡 |
| 25 | "Textbooks Are All You Need"（Phi系列） | Microsoft | 论文 | 中级 | 高质量数据的重要性，"教科书质量"数据理念 |
| 26 | "Magpie: Alignment Data Synthesis by Augmenting and Repurposing" | 多位作者 | 论文 | 中高级 | 数据合成与对齐方法 |

#### 4.2.5 模仿学习局限与改进

| 序号 | 资料名称 | 作者/来源 | 类型 | 难度 | 推荐理由 |
|------|----------|-----------|------|------|----------|
| 27 | "DAGGER: Dataset Aggregation for Imitation Learning" | S. Ross等 | 论文 | 中级 | 解决模仿学习分布偏移的经典方法 |
| 28 | "Understanding the Weaknesses of LLM-based Agents" | 多位作者 | 论文 | 中级 | 分析Agent的弱点，理解模仿学习的局限 |
| 29 | "Reinforced Self-Training (ReST) for Language Modeling" | G. Gulcehre等 | 论文 | 中高级 | 自训练+强化学习，克服模仿学习局限 |

#### 4.2.6 实践项目建议

| 序号 | 项目名称 | 类型 | 难度 | 说明 |
|------|----------|------|------|------|
| P4 | 使用LLaMA-Factory进行Agent SFT | 实践项目 | 中级 | 收集/合成Agent轨迹数据，使用LLaMA-Factory进行SFT训练 |
| P5 | 构建轨迹数据合成管线 | 实践项目 | 高级 | 实现从任务模板→轨迹生成→质量筛选→SFT数据的完整管线 |
| P6 | 实现PRM-based轨迹筛选 | 实践项目 | 高级 | 训练过程奖励模型，用于步骤级轨迹质量评估 |

---

### 4.3 智能体强化学习 — 学习资料

#### 4.3.1 强化学习基础

| 序号 | 资料名称 | 作者/来源 | 类型 | 难度 | 推荐理由 |
|------|----------|-----------|------|------|----------|
| 1 | 《Reinforcement Learning: An Introduction》 | R. Sutton & A. Barto | 书籍 | 入门-中级 | 强化学习圣经，系统全面，覆盖MDP到现代RL方法 |
| 2 | David Silver's RL Course | DeepMind | 大学课程 | 入门-中级 | DeepMind强化学习课程，经典入门 |
| 3 | Stanford CS234: Reinforcement Learning | Stanford | 大学课程 | 中级 | Stanford强化学习课程，理论与代码并重 |
| 4 | OpenAI Spinning Up in Deep RL | OpenAI | 教程+代码 | 入门-中级 | 深度强化学习入门资源，从基础到实现 |
| 5 | Hugging Face Deep RL Course | Hugging Face | 教程 | 入门-中级 | 实操导向的深度强化学习课程 |
| 6 | Berkeley CS285: Deep Reinforcement Learning | UC Berkeley | 大学课程 | 高级 | 深度强化学习高级课程，Sergey Levine主讲 |

#### 4.3.2 PPO与LLM对齐

| 序号 | 资料名称 | 作者/来源 | 类型 | 难度 | 推荐理由 |
|------|----------|-----------|------|------|----------|
| 7 | "Proximal Policy Optimization Algorithms" | J. Schulman等（OpenAI） | 论文 | 中级 | PPO原论文，理解裁剪目标函数和GAE |
| 8 | "Training Language Models to Follow Instructions with Human Feedback"（InstructGPT） | L. Ouyang等 | 论文 | 中高级 | RLHF三阶段流程的经典论文 |
| 9 | "Constitutional AI: Harmlessness from AI Feedback" | Anthropic | 论文 | 中高级 | RLAIF方法，AI反馈替代人类反馈 |
| 10 | TRL（Transformer Reinforcement Learning）库 | Hugging Face | 开源项目 | 中级 | HF的RL训练库，PPO/DPO实现 |
| 11 | DeepSpeed-Chat | Microsoft | 开源项目 | 中高级 | 完整的RLHF训练流程实现 |
| 12 | "Direct Preference Optimization" | R. Rafailov等 | 论文 | 中级 | DPO方法，简化对齐训练 |
| 13 | "Trust Region Policy Optimization"（TRPO） | J. Schulman等 | 论文 | 高级 | PPO的前身，理解信任域优化 |

#### 4.3.3 GRPO与最新RL方法

| 序号 | 资料名称 | 作者/来源 | 类型 | 难度 | 推荐理由 |
|------|----------|-----------|------|------|----------|
| 14 | "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning" | DeepSeek | 论文+开源 | 高级 | GRPO的实际应用，推理能力的RL训练 |
| 15 | "DeepSeekMath: Pushing the Limits of Math Reasoning" | DeepSeek | 论文 | 高级 | GRPO算法的详细描述与数学推导 |
| 16 | "Group Relative Policy Optimization" | DeepSeek | 论文 | 高级 | GRPO算法论文，理解组内相对优势估计 |
| 17 | OpenRLHF 开源项目 | OpenRLHF | 开源项目 | 中高级 | 开源RLHF框架，支持PPO/DPO/GRPO等 |
| 18 | verl 开源项目 | Volcano Engine | 开源项目 | 高级 | 字节跳动的RL训练框架，支持大规模RL |
| 19 | "Rejection Sampling Fine-Tuning" | 多位作者 | 论文 | 中级 | RFT方法，拒绝采样微调 |

#### 4.3.4 智能体RL特有挑战

| 序号 | 资料名称 | 作者/来源 | 类型 | 难度 | 推荐理由 |
|------|----------|-----------|------|------|----------|
| 20 | "RLHF + Agent: Reinforcement Learning from Human Feedback for Agent" | 多位作者 | 论文 | 高级 | Agent场景下的RLHF，讨论Agent RL的特有挑战 |
| 21 | "Agent-R: Training Language Model Agents via Self-Reflection" | 多位作者 | 论文 | 高级 | 通过自我反思训练Agent |
| 22 | "Agent-RewardBench: Benchmarking Reward Models for Agent" | 多位作者 | 论文 | 高级 | Agent场景的奖励模型评估 |
| 23 | "Echo: Exploring the Integration of Reinforcement Learning and Agent Fine-Tuning" | 多位作者 | 论文 | 高级 | RL与Agent微调的集成 |
| 24 | "Online RL for LLM Agents: Challenges and Solutions" | 多位作者 | 论文 | 高级 | 在线RL在Agent场景的挑战与方案 |
| 25 | "Learning to Use Tools via Reinforcement Learning" | 多位作者 | 论文 | 高级 | 通过RL学习工具使用 |

#### 4.3.5 Task合成与环境模拟

| 序号 | 资料名称 | 作者/来源 | 类型 | 难度 | 推荐理由 |
|------|----------|-----------|------|------|----------|
| 26 | "AgentGym: Evolving Large Language Model-Based Agents" | 多位作者 | 论文+开源 | 高级 | Agent训练环境与任务合成 |
| 27 | "AgentTrek: Agent Trajectory Synthesis via Guiding Replay" | 多位作者 | 论文 | 高级 | 轨迹合成与回放引导 |
| 28 | OpenAI Gym / Gymnasium | OpenAI / Farama | 开源项目 | 入门-中级 | 标准化RL环境接口，理解环境设计 |
| 29 | "Environment Design for Language Agent Training" | 多位作者 | 论文 | 高级 | Agent训练环境设计方法论 |
| 30 | "CraftBase: A Sandbox Environment for Agent Training" | 多位作者 | 论文 | 高级 | 沙箱环境中的Agent训练 |

#### 4.3.6 实践项目建议

| 序号 | 项目名称 | 类型 | 难度 | 说明 |
|------|----------|------|------|------|
| P7 | 使用TRL跑通PPO训练 | 实践项目 | 中级 | 使用HuggingFace TRL库在小模型上跑通PPO |
| P8 | 使用GRPO训练Agent | 实践项目 | 高级 | 使用OpenRLHF或verl实现GRPO训练Agent |
| P9 | 构建Agent RL训练环境 | 实践项目 | 高级 | 实现任务合成+环境模拟+奖励函数的完整RL训练环境 |
| P10 | Agent RL超参调优实验 | 实践项目 | 高级 | 系统实验KL系数、entropy系数等超参的影响 |

---

### 4.4 智能体评估 — 学习资料

#### 4.4.1 评估基础理论

| 序号 | 资料名称 | 作者/来源 | 类型 | 难度 | 推荐理由 |
|------|----------|-----------|------|------|----------|
| 1 | "Holistic Evaluation of Language Models"（HELM） | Stanford CRFM | 论文+开源 | 中级 | 语言模型全面评估框架，评估方法论参考 |
| 2 | "Evaluating Large Language Models: A Survey" | 多位作者 | 综述论文 | 中级 | LLM评估方法综述 |
| 3 | "LLM-as-a-Judge: Evaluating LLMs with LLMs" | L. Zheng等 | 论文 | 中级 | 使用LLM作为评估者的方法论 |
| 4 | "On the Limitations of Reasoning LLM Benchmarks" | 多位作者 | 论文 | 中级 | 分析现有基准的局限性 |

#### 4.4.2 主流评测榜单

| 序号 | 资料名称 | 作者/来源 | 类型 | 难度 | 推荐理由 |
|------|----------|-----------|------|------|----------|
| 5 | Berkeley Function Calling Leaderboard (BFCL) | UC Berkeley | 开源项目+榜单 | 中级 | 函数调用评估的标杆，理解工具使用评估 |
| 6 | "tau-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains" | Sierra AI | 论文+开源 | 中高级 | 用户模拟评估方法，多轮工具调用评估 |
| 7 | Toolathlon 评测 | 清华等 | 开源项目 | 中级 | 多工具组合任务评估 |
| 8 | BrowseComp | OpenAI | 榜单 | 中级 | 浏览器操作评估 |
| 9 | Claw-Eval | 多位作者 | 开源项目 | 中高级 | 动态环境Agent评估 |
| 10 | "AgentBench: Evaluating LLMs as Agents" | X. Liu等 | 论文+开源 | 中级 | 多场景Agent能力评估，8个场景 |
| 11 | "SWE-bench: Can Language Models Resolve Real-World GitHub Issues?" | C. Jimenez等 | 论文+开源 | 中高级 | 软件工程任务评估，真实GitHub Issue |
| 12 | "WebArena: A Realistic Web Environment for Building Autonomous Agents" | S. Zhou等 | 论文+开源 | 中高级 | 网页交互任务评估 |
| 13 | "GAIA: A Benchmark for General AI Assistants" | G. Mialon等 | 论文+开源 | 中高级 | 通用AI助手评估 |
| 14 | "ToolBench: Facilitating LLMs to Master 16000+ Real-world APIs" | Y. Qin等 | 论文+开源 | 中级 | 大规模工具使用评估 |
| 15 | "MLE-bench: Evaluating Machine Learning Agents on ML Engineering" | OpenAI | 论文+开源 | 高级 | ML工程能力评估 |

#### 4.4.3 评估局限性与高级方向

| 序号 | 资料名称 | 作者/来源 | 类型 | 难度 | 推荐理由 |
|------|----------|-----------|------|------|----------|
| 16 | "Dynamic Evaluation of LLMs" | 多位作者 | 论文 | 高级 | 动态环境评估方法 |
| 17 | "Contamination-Free Evaluation" | 多位作者 | 论文 | 中级 | 数据污染检测与无污染评估 |
| 18 | "Multi-Agent Collaboration Evaluation" | 多位作者 | 论文 | 高级 | 多智能体协作评估 |
| 19 | "Safety Evaluation of LLM Agents" | 多位作者 | 论文 | 高级 | Agent安全性评估 |

#### 4.4.4 实践项目建议

| 序号 | 项目名称 | 类型 | 难度 | 说明 |
|------|----------|------|------|------|
| P11 | 在BFCL上评测自己的Agent模型 | 实践项目 | 中级 | 使用BFCL框架评测自训练的Agent模型 |
| P12 | 搭建动态评估环境 | 实践项目 | 高级 | 实现一个支持动态环境响应的Agent评估系统 |
| P13 | 分析评测榜单局限性 | 实践项目 | 中级 | 选择2-3个榜单，深入分析其与真实体验的Gap |

---

### 4.5 通用型Agent工具使用 — 学习资料

#### 4.5.1 ClaudeCode

| 序号 | 资料名称 | 作者/来源 | 类型 | 难度 | 推荐理由 |
|------|----------|-----------|------|------|----------|
| 1 | Claude Code 官方文档 | Anthropic | 文档 | 入门 | ClaudeCode的官方使用文档，涵盖安装、配置、使用 |
| 2 | Anthropic Cookbook | Anthropic | 开源项目+教程 | 入门-中级 | 官方示例代码集，包含Agent构建最佳实践 |
| 3 | "Building Effective Agents" | Anthropic Blog | 博客 | 入门-中级 | Anthropic关于构建有效Agent的博客文章，核心概念 |
| 4 | Claude Code 最佳实践指南 | 社区/博客 | 博客 | 中级 | 社区总结的ClaudeCode高效使用技巧 |
| 5 | "Context Engineering for AI Coding Agents" | 多位作者 | 博客 | 中级 | 上下文工程在编码Agent中的应用 |

#### 4.5.2 Cursor

| 序号 | 资料名称 | 作者/来源 | 类型 | 难度 | 推荐理由 |
|------|----------|-----------|------|------|----------|
| 6 | Cursor 官方文档与教程 | Cursor | 文档+教程 | 入门 | Cursor的官方使用指南 |
| 7 | "How to Use Cursor AI: Complete Guide" | 社区博客 | 博客 | 入门-中级 | Cursor的完整使用指南 |
| 8 | Cursor Rules 最佳实践 | 社区 | 博客 | 中级 | Cursor规则文件的最佳实践 |

#### 4.5.3 Codex / OpenAI

| 序号 | 资料名称 | 作者/来源 | 类型 | 难度 | 推荐理由 |
|------|----------|-----------|------|------|----------|
| 9 | OpenAI Codex CLI 文档 | OpenAI | 文档 | 入门 | Codex CLI的官方文档 |
| 10 | OpenAI Agents SDK 文档 | OpenAI | 文档+开源 | 中级 | OpenAI的Agent SDK，理解Agent构建范式 |
| 11 | "OpenAI Agents SDK: A Guide" | 社区 | 博客 | 中级 | OpenAI Agents SDK的使用指南 |

#### 4.5.4 OpenClaw与其他

| 序号 | 资料名称 | 作者/来源 | 类型 | 难度 | 推荐理由 |
|------|----------|-----------|------|------|----------|
| 12 | OpenClaw 官方文档与源码 | OpenClaw | 开源项目 | 中级 | 开源Agent框架，理解Agent构建 |
| 13 | "The AI Coding Assistant Landscape: Claude Code vs Cursor vs Copilot vs Codex" | 社区博客 | 博客 | 入门 | 各工具对比，理解不同工具的优劣 |
| 14 | "One Person Company with AI Agents" | 社区博客 | 博客 | 中级 | 使用AI Agent实现"一人公司"的实践分享 |

#### 4.5.5 实践项目建议

| 序号 | 项目名称 | 类型 | 难度 | 说明 |
|------|----------|------|------|------|
| P14 | 使用ClaudeCode完成完整项目开发 | 实践项目 | 中级 | 从需求到部署使用CC完成一个完整项目 |
| P15 | 使用Cursor进行大规模代码重构 | 实践项目 | 中级 | 使用Cursor完成跨文件重构任务 |
| P16 | 对比不同Agent工具的效率 | 实践项目 | 中级 | 系统对比CC/Cursor/Codex在同类任务上的表现 |

---

### 4.6 面向基模的智能体Harness搭建 — 学习资料

#### 4.6.1 Harness基础概念

| 序号 | 资料名称 | 作者/来源 | 类型 | 难度 | 推荐理由 |
|------|----------|-----------|------|------|----------|
| 1 | "ReAct: Synergizing Reasoning and Acting in Language Models" | S. Yao等 | 论文 | 入门-中级 | ReAct范式的开创性论文，Agent核心概念 |
| 2 | "Toolformer: Language Models Can Teach Themselves to Use Tools" | T. Schick等 | 论文 | 中级 | 工具使用的开创性工作 |
| 3 | "A Survey on Large Language Model based Autonomous Agents" | L. Wang等 | 综述论文 | 中级 | LLM Agent全面综述，覆盖架构、能力、应用 |
| 4 | "The Rise and Potential of Large Language Model Based Agents" | Z. Xi等 | 综述论文 | 中级 | Agent发展综述，理解Agent技术全景 |
| 5 | "LLM Powered Autonomous Agents" | Lilian Weng | 博客 | 入门-中级 | Lilian Weng的经典Agent博客，概念清晰 |
| 6 | "Cognitive Architectures for Language Agents"（CoALA） | S. Sumers等 | 论文 | 中级 | Agent认知架构框架，系统化理解Agent组件 |

#### 4.6.2 上下文工程与Skill系统

| 序号 | 资料名称 | 作者/来源 | 类型 | 难度 | 推荐理由 |
|------|----------|-----------|------|------|----------|
| 7 | "Lost in the Middle: How Language Models Use Long Contexts" | Liu等 | 论文 | 中级 | 理解LLM在长上下文中的行为，指导上下文工程 |
| 8 | "Prompt Engineering Guide" | DAIR.AI | 教程 | 入门-中级 | 全面的Prompt工程指南 |
| 9 | "Chain-of-Thought Prompting Elicits Reasoning in LLMs" | J. Wei等 | 论文 | 入门-中级 | CoT提示，推理链的基础 |
| 10 | "Tree of Thoughts: Deliberate Problem Solving with LLMs" | S. Yao等 | 论文 | 中级 | 树形推理，ReAct的扩展 |
| 11 | "Reflexion: Language Agents with Verbal Reinforcement Learning" | N. Shinn等 | 论文+开源 | 中级 | 反思机制，Agent自我改进 |

#### 4.6.3 MCP协议

| 序号 | 资料名称 | 作者/来源 | 类型 | 难度 | 推荐理由 |
|------|----------|-----------|------|------|----------|
| 12 | Model Context Protocol 官方规范 | Anthropic | 规范文档 | 入门-中级 | MCP协议的官方规范，理解Tool/Resource/Prompt |
| 13 | MCP TypeScript SDK | Anthropic | 开源项目 | 中级 | MCP的TypeScript SDK实现 |
| 14 | MCP Python SDK | Anthropic | 开源项目 | 中级 | MCP的Python SDK实现 |
| 15 | "Awesome MCP Servers" | 社区 | 开源项目列表 | 入门 | MCP Server集合，学习各种工具集成 |
| 16 | "Building MCP Servers: A Guide" | 社区博客 | 博客 | 中级 | MCP Server开发实践指南 |

#### 4.6.4 主流Harness框架源码学习

| 序号 | 资料名称 | 作者/来源 | 类型 | 难度 | 推荐理由 |
|------|----------|-----------|------|------|----------|
| 17 | OpenClaw 源码与文档 | OpenClaw | 开源项目 | 中高级 | 主流开源Harness，学习runtime loop、工具基建、上下文管理 |
| 18 | Hermes 源码与文档 | Hermes | 开源项目 | 中高级 | 多智能体Harness，学习多Agent同步/异步机制 |
| 19 | LangGraph 源码与文档 | LangChain | 开源项目 | 中级 | 图结构Agent编排，学习状态管理与条件路由 |
| 20 | LangChain Agent 模块源码 | LangChain | 开源项目 | 中级 | 工业级Agent框架，学习Agent执行循环 |
| 21 | AutoGPT 源码 | Significant Gravitas | 开源项目 | 中级 | 早期自主Agent，理解Agent基本架构 |
| 22 | CrewAI 源码与文档 | CrewAI | 开源项目 | 中级 | 多Agent协作框架 |
| 23 | MetaGPT 源码与文档 | DeepWisdom | 开源项目 | 中高级 | 多Agent软件团队模拟 |

#### 4.6.5 多智能体与自演化

| 序号 | 资料名称 | 作者/来源 | 类型 | 难度 | 推荐理由 |
|------|----------|-----------|------|------|----------|
| 24 | "MetaGPT: Meta Programming for Multi-Agent Collaborative Framework" | S. Hong等 | 论文+开源 | 中高级 | 多Agent协作框架论文 |
| 25 | "CAMEL: Communicative Agents for Mind Exploration" | 多位作者 | 论文+开源 | 中高级 | 多Agent通信与协作 |
| 26 | "AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation" | Microsoft | 论文+开源 | 中级 | 微软的多Agent对话框架 |
| 27 | "Voyager: An Open-Ended Embodied Agent with LLMs" | G. Wang等 | 论文+开源 | 高级 | 自演化Agent，技能库+课程学习 |
| 28 | "ExpeL: LLM Agents Are Experiential Learners" | A. Zhao等 | 论文 | 高级 | 经验学习与自演化 |

#### 4.6.6 实践项目建议

| 序号 | 项目名称 | 类型 | 难度 | 说明 |
|------|----------|------|------|------|
| P17 | 跑通OpenClaw/Hermes并分析其架构 | 实践项目 | 中级 | 部署运行主流Harness，深入分析其内部实现 |
| P18 | 基于已有Harness开发自定义Agent | 实践项目 | 中高级 | 在OpenClaw或LangGraph上开发自定义Agent功能 |
| P19 | 从零搭建一个Mini Harness | 实践项目 | 高级 | 实现runtime loop+工具基建+上下文管理+记忆管理的完整Harness |
| P20 | 实现多智能体协作系统 | 实践项目 | 高级 | 实现多Agent同步/异步通信与协作 |

---

### 4.7 沙箱环境工程 — 学习资料

#### 4.7.1 沙箱基础

| 序号 | 资料名称 | 作者/来源 | 类型 | 难度 | 推荐理由 |
|------|----------|-----------|------|------|----------|
| 1 | Docker 官方文档与教程 | Docker | 文档+教程 | 入门 | Docker基础，沙箱搭建的第一步 |
| 2 | 《Docker Deep Dive》 | Nigel Poulton | 书籍 | 入门-中级 | Docker深入指南，从基础到高级 |
| 3 | "Linux Containers and Virtualization: A Practical Guide" | 多位作者 | 书籍 | 中级 | Linux容器与虚拟化技术全面指南 |
| 4 | "Understanding Linux Kernel" 相关章节 — namespace与cgroup | 多位作者 | 书籍 | 高级 | 理解Linux内核的隔离机制 |
| 5 | "Playing with seccomp" | Julia Evans | 博客 | 中级 | seccomp系统调用过滤的实践指南 |

#### 4.7.2 容器与编排

| 序号 | 资料名称 | 作者/来源 | 类型 | 难度 | 推荐理由 |
|------|----------|-----------|------|------|----------|
| 6 | Kubernetes 官方文档与教程 | CNCF/Kubernetes | 文档+教程 | 中级 | K8s官方文档，多实例沙箱编排 |
| 7 | 《Kubernetes in Action》 | Marko Lukša | 书籍 | 中级 | K8s实践指南，从入门到生产 |
| 8 | 《Kubernetes Patterns》 | B. Ibragimov等 | 书籍 | 中高级 | K8s设计模式，最佳实践 |
| 9 | Docker Compose 官方文档 | Docker | 文档 | 入门 | 多容器编排，多实例沙箱入门 |
| 10 | "Kubernetes Security: Operating Kubernetes Clusters and Applications Safely" | A. Stennett等 | 书籍 | 中高级 | K8s安全实践，沙箱安全防护 |

#### 4.7.3 高级隔离技术

| 序号 | 资料名称 | 作者/来源 | 类型 | 难度 | 推荐理由 |
|------|----------|-----------|------|------|----------|
| 11 | Firecracker 官方文档与源码 | AWS | 开源项目+文档 | 高级 | AWS微VM，高并发沙箱核心技术 |
| 12 | "Firecracker: Lightweight Virtualization for Serverless Applications" | A. Agache等 | 论文 | 高级 | Firecracker设计论文，理解微VM架构 |
| 13 | Kata Containers 官方文档 | Kata Containers | 开源项目+文档 | 高级 | 兼容K8s的微VM，强隔离沙箱 |
| 14 | gVisor 官方文档与源码 | Google | 开源项目+文档 | 高级 | 应用内核沙箱，系统调用拦截 |
| 15 | "gVisor: A User-Space Kernel for Container Security" | Google | 论文/技术文档 | 高级 | gVisor架构设计，理解系统调用沙箱 |
| 16 | WebAssembly (WASM) 沙箱相关 | Wasmtime / Wasmer | 开源项目 | 中高级 | WASM沙箱运行时，轻量级代码执行 |
| 17 | "WASI: WebAssembly System Interface" | Bytecode Alliance | 规范文档 | 中高级 | WASM系统接口规范 |
| 18 | nsjail 开源项目 | Google | 开源项目 | 中级 | 进程级沙箱工具，快速隔离 |

#### 4.7.4 安全与监控

| 序号 | 资料名称 | 作者/来源 | 类型 | 难度 | 推荐理由 |
|------|----------|-----------|------|------|----------|
| 19 | 《Container Security: Fundamental Technology Concepts That Protect Containerized Applications》 | L. Liz Rice | 书籍 | 中级 | 容器安全基础，沙箱安全防护 |
| 20 | Trivy / Clair 安全扫描工具 | Aquasec / Clair | 开源项目 | 入门-中级 | 容器镜像安全扫描 |
| 21 | Prometheus + Grafana 监控栈 | CNCF | 开源项目 | 中级 | 沙箱监控与可观测性 |
| 22 | OpenTelemetry 分布式追踪 | CNCF | 开源项目 | 中高级 | 分布式追踪，多实例沙箱可观测性 |
| 23 | "Container Escape: Attacking and Defending Docker" | 多位作者 | 博客/论文 | 高级 | 容器逃逸攻击与防护 |
| 24 | Falco 运行时安全监控 | CNCF | 开源项目 | 中高级 | 容器运行时安全监控 |

#### 4.7.5 Agent沙箱专项

| 序号 | 资料名称 | 作者/来源 | 类型 | 难度 | 推荐理由 |
|------|----------|-----------|------|------|----------|
| 25 | E2B (Code Interpreter Sandbox) | E2B | 开源项目 | 中高级 | 专为AI Agent设计的代码执行沙箱 |
| 26 | Daytona 开发环境管理 | Daytona | 开源项目 | 中级 | 开发环境管理，可参考沙箱管理设计 |
| 27 | Modal Serverless Sandbox | Modal | 文档+平台 | 中高级 | Serverless沙箱，高并发代码执行 |
| 28 | "Sandboxing AI Agents: Best Practices" | 社区博客 | 博客 | 中级 | AI Agent沙箱最佳实践 |
| 29 | "Running Claude Code / OpenClaw in Sandboxes at Scale" | 社区博客 | 博客 | 高级 | 高并发运行Agent Harness的沙箱工程实践 |

#### 4.7.6 实践项目建议

| 序号 | 项目名称 | 类型 | 难度 | 说明 |
|------|----------|------|------|------|
| P21 | 搭建单实例Docker沙箱 | 实践项目 | 入门 | 使用Docker搭建Agent代码执行沙箱 |
| P22 | 搭建多实例隔离沙箱 | 实践项目 | 中级 | 使用K8s/Docker Compose搭建多实例隔离沙箱 |
| P23 | 使用Firecracker搭建高并发沙箱 | 实践项目 | 高级 | 使用Firecracker微VM搭建高并发Agent执行环境 |
| P24 | 构建沙箱即服务平台 | 实践项目 | 高级 | 实现统一API+多租户+自动扩缩容的沙箱服务平台 |
| P25 | Agent沙箱安全加固 | 实践项目 | 高级 | 实现恶意代码检测+沙箱逃逸防护+执行审计 |

---

## 5. 学习路径建议

### 5.1 总体学习路线（6个阶段，预计6-9个月）

```
阶段1: 基础筑基（1-2月）
  ├── LLM基础理论
  ├── 强化学习基础
  └── Agent概念入门

阶段2: Agent核心技术（1.5-2月）
  ├── 记忆系统
  ├── 轨迹数据合成
  └── SFT训练实践

阶段3: 强化学习深入（1-2月）
  ├── PPO/GRPO算法
  ├── Agent RL实践
  └── 超参调优

阶段4: Harness与工具链（1-1.5月）
  ├── Harness框架学习
  ├── MCP协议
  └── 自建Harness

阶段5: 沙箱工程（1月）
  ├── Docker/K8s基础
  ├── 高级隔离技术
  └── 高并发沙箱

阶段6: 评估与综合实战（1月）
  ├── 评测榜单实践
  ├── 综合项目
  └── 开源贡献
```

### 5.2 详细阶段规划

#### 阶段1：基础筑基（第1-2月）

**目标**：建立LLM和RL的理论基础，理解Agent基本概念

**学习内容**：
1. LLM基础
   - 阅读《大语言模型》前4章
   - 学习Stanford CS224N课程核心讲座
   - 理解Transformer架构和训练流程

2. 强化学习基础
   - 阅读《RL: An Introduction》前6章
   - 完成OpenAI Spinning Up教程
   - 理解MDP、策略梯度、价值函数

3. Agent概念入门
   - 阅读Lilian Weng的"LLM Powered Autonomous Agents"
   - 阅读"ReAct"论文
   - 阅读"A Survey on LLM based Autonomous Agents"

**里程碑**：
- 能解释LLM训练流程（预训练→SFT→RLHF）
- 能解释PPO算法的基本原理
- 能描述Agent的基本架构（感知→推理→行动→观察）

**推荐资料优先级**：
- [基础] 《大语言模型》→ CS224N → Spinning Up → ReAct论文 → Agent综述

---

#### 阶段2：Agent核心技术（第3-4.5月）

**目标**：掌握记忆系统、轨迹数据合成和SFT训练

**学习内容**：
1. 记忆系统（2周）
   - 阅读"MemGPT"论文和源码
   - 学习LangChain Memory模块
   - 阅读"Generative Agents"论文（记忆流+反思）
   - 理解RAG基础和向量数据库

2. 轨迹数据合成（2周）
   - 阅读"Self-Instruct"、"Toolformer"、"Agent-FLAN"论文
   - 理解轨迹数据格式和合成方法
   - 理解模仿学习的局限性

3. SFT训练实践（2-3周）
   - 使用LLaMA-Factory/Axolotl进行SFT
   - 学习LoRA/QLoRA参数高效微调
   - 实践Agent SFT（工具调用格式SFT）

4. 数据质量筛选（1-2周）
   - 学习PRM（"Let's Verify Step by Step"）
   - 理解turn-level和step-level筛选
   - 实践轨迹数据curation

**实践项目**：
- P1: 搭建带记忆的对话Agent
- P4: 使用LLaMA-Factory进行Agent SFT
- P5: 构建轨迹数据合成管线

**里程碑**：
- 能搭建完整的Agent记忆系统
- 能合成Agent轨迹数据并进行SFT训练
- 能实现轨迹数据质量筛选

**推荐资料优先级**：
- [记忆] MemGPT论文+源码 → LangChain Memory → Generative Agents → RAG论文
- [轨迹] Self-Instruct → Toolformer → Agent-FLAN → PRM论文
- [SFT] LLaMA-Factory → LoRA论文 → DeepSpeed-Chat

---

#### 阶段3：强化学习深入（第5-6.5月）

**目标**：掌握PPO/GRPO算法，具备Agent RL实操经验

**学习内容**：
1. PPO与RLHF（2周）
   - 深入学习PPO论文
   - 学习InstructGPT论文（RLHF三阶段）
   - 使用TRL库跑通PPO训练

2. GRPO与最新方法（2周）
   - 学习DeepSeek-R1和DeepSeekMath论文
   - 理解GRPO算法原理
   - 使用OpenRLHF/verl实践GRPO

3. Agent RL特有挑战（2周）
   - 理解Agent RL vs Reasoning RL的差异
   - 学习Task合成技术
   - 学习环境模拟与沙箱构建

4. 超参调优（1-2周）
   - 理解KL、entropy等超参含义
   - 实践超参调优实验

**实践项目**：
- P7: 使用TRL跑通PPO训练
- P8: 使用GRPO训练Agent
- P9: 构建Agent RL训练环境
- P10: Agent RL超参调优实验

**里程碑**：
- 能跑通PPO和GRPO训练
- 能构建Agent RL训练环境（任务合成+环境模拟+奖励函数）
- 能调节RL训练超参（KL、entropy等）

**推荐资料优先级**：
- [PPO] PPO论文 → InstructGPT → TRL库 → DeepSpeed-Chat
- [GRPO] DeepSeek-R1 → DeepSeekMath → OpenRLHF → verl
- [Agent RL] Agent-R → AgentGym → AgentTrek

---

#### 阶段4：Harness与工具链（第7-8.5月）

**目标**：掌握Harness架构，能独立搭建Harness

**学习内容**：
1. Harness基础概念（1周）
   - 复习ReAct、CoT
   - 学习上下文工程
   - 理解Skill系统

2. MCP协议（1周）
   - 学习MCP官方规范
   - 实践MCP Server开发
   - 理解Tool/Resource/Prompt

3. 主流Harness框架学习（2-3周）
   - 深入学习OpenClaw源码（runtime loop、工具基建、上下文管理）
   - 学习Hermes（多智能体机制）
   - 学习LangGraph（图结构编排）

4. Harness搭建实践（2-3周）
   - 基于已有框架二次开发
   - 从零搭建Mini Harness
   - 实现多智能体协作

5. Agent工具使用（持续）
   - 熟练使用ClaudeCode
   - 熟练使用Cursor
   - 使用Agent完成全周期开发

**实践项目**：
- P14: 使用ClaudeCode完成完整项目开发
- P17: 跑通OpenClaw/Hermes并分析架构
- P18: 基于已有Harness开发自定义Agent
- P19: 从零搭建Mini Harness
- P20: 实现多智能体协作系统

**里程碑**：
- 能独立分析主流Harness的内部实现
- 能基于已有Harness进行二次开发
- 能从零搭建一个功能完整的Harness

**推荐资料优先级**：
- [概念] ReAct论文 → CoALA → Lilian Weng博客 → 上下文工程
- [MCP] MCP规范 → MCP SDK → Awesome MCP Servers
- [框架] OpenClaw源码 → Hermes源码 → LangGraph源码

---

#### 阶段5：沙箱工程（第9-10月）

**目标**：掌握沙箱搭建与高并发运行

**学习内容**：
1. Docker与容器基础（1周）
   - Docker基础与Dockerfile
   - 单实例沙箱搭建
   - 故障排查

2. K8s与多实例沙箱（2周）
   - K8s基础
   - 多实例隔离沙箱
   - 精细化权限与资源管控

3. 高级隔离技术（2周）
   - Firecracker微VM
   - gVisor沙箱
   - WASM沙箱

4. 高并发与安全（2周）
   - 高并发调度
   - 安全防护
   - 监控与可观测性

**实践项目**：
- P21: 搭建单实例Docker沙箱
- P22: 搭建多实例隔离沙箱
- P23: 使用Firecracker搭建高并发沙箱
- P24: 构建沙箱即服务平台
- P25: Agent沙箱安全加固

**里程碑**：
- 能搭建多实例隔离沙箱
- 能使用Firecracker实现高并发沙箱
- 能设计完整的沙箱工程方案

**推荐资料优先级**：
- [基础] Docker文档 → Docker Deep Dive → K8s文档
- [高级] Firecracker → gVisor → WASM → nsjail
- [安全] Container Security → Trivy → Falco

---

#### 阶段6：评估与综合实战（第11-12月）

**目标**：掌握Agent评估，完成综合实战项目

**学习内容**：
1. 评估基础理论（1周）
   - 学习评估方法论
   - 理解LLM-as-Judge
   - 学习HELM

2. 主流评测榜单实践（2周）
   - 在BFCL上评测Agent模型
   - 了解TauBench、Toolathlon、BrowseComp等
   - 分析评测局限性

3. 综合实战项目（2-3周）
   - 整合记忆+训练+RL+Harness+沙箱
   - 完成端到端Agent系统

4. 开源贡献（持续）
   - 向开源项目贡献代码
   - 发布技术博客
   - 参与社区讨论

**实践项目**：
- P11: 在BFCL上评测自己的Agent模型
- P12: 搭建动态评估环境
- P13: 分析评测榜单局限性
- 综合项目：端到端Agent系统（记忆+训练+RL+Harness+沙箱+评估）

**里程碑**：
- 能在主流榜单上评测Agent模型
- 能搭建动态评估环境
- 能完成端到端Agent系统

---

### 5.3 学习资料优先级总览

#### 必读核心资料（Top 20）

| 优先级 | 资料名称 | 类型 | 对应知识点 |
|--------|----------|------|------------|
| 1 | 《大语言模型》赵鑫等 | 书籍 | LLM基础 |
| 2 | 《RL: An Introduction》Sutton & Barto | 书籍 | RL基础 |
| 3 | "ReAct" 论文 | 论文 | Agent基础 |
| 4 | "MemGPT" 论文+源码 | 论文+项目 | 记忆系统 |
| 5 | "Generative Agents" 论文 | 论文 | 记忆系统 |
| 6 | "Self-Instruct" 论文 | 论文 | 轨迹合成 |
| 7 | "Agent-FLAN" 论文 | 论文 | 轨迹合成+SFT |
| 8 | "PPO" 论文 | 论文 | RL算法 |
| 9 | "DeepSeek-R1" 论文 | 论文 | GRPO |
| 10 | "Let's Verify Step by Step" 论文 | 论文 | 数据质量 |
| 11 | LLaMA-Factory 开源项目 | 项目 | SFT实践 |
| 12 | OpenRLHF / verl 开源项目 | 项目 | RL训练 |
| 13 | OpenClaw 源码 | 项目 | Harness |
| 14 | LangGraph 源码 | 项目 | Harness |
| 15 | MCP 官方规范 | 规范 | MCP协议 |
| 16 | Docker 官方文档 | 文档 | 沙箱基础 |
| 17 | Firecracker 文档与源码 | 项目 | 高级沙箱 |
| 18 | BFCL 评测 | 榜单 | Agent评估 |
| 19 | SWE-bench 评测 | 榜单 | Agent评估 |
| 20 | Lilian Weng "LLM Powered Autonomous Agents" | 博客 | Agent综述 |

#### 进阶提升资料（Top 15）

| 优先级 | 资料名称 | 类型 | 对应知识点 |
|--------|----------|------|------------|
| 21 | Stanford CS224N 课程 | 课程 | LLM基础 |
| 22 | Berkeley CS285 课程 | 课程 | 深度RL |
| 23 | "InstructGPT" 论文 | 论文 | RLHF |
| 24 | "DPO" 论文 | 论文 | 对齐训练 |
| 25 | "Reflexion" 论文 | 论文 | Agent反思 |
| 26 | "Voyager" 论文 | 论文 | 自演化 |
| 27 | "AgentBench" 论文 | 论文 | Agent评估 |
| 28 | "GAIA" 论文 | 论文 | Agent评估 |
| 29 | Hermes 源码 | 项目 | 多Agent Harness |
| 30 | AutoGen 源码 | 项目 | 多Agent协作 |
| 31 | E2B 沙箱项目 | 项目 | Agent沙箱 |
| 32 | gVisor 文档与源码 | 项目 | 沙箱隔离 |
| 33 | 《Kubernetes in Action》 | 书籍 | 容器编排 |
| 34 | "tau-bench" 论文 | 论文 | Agent评估 |
| 35 | Anthropic "Building Effective Agents" | 博客 | Agent最佳实践 |

---

### 5.4 掌握程度达成路径

| 目标等级 | 需要完成的学习与实践 | 预计时间 |
|----------|----------------------|----------|
| **基础级（了解概念）** | 完成阶段1 | 1-2月 |
| **进阶级（跑通开源项目）** | 完成阶段1-3 + 对应实践项目 | 4-5月 |
| **精通级（实操经验）** | 完成阶段1-6 + 全部核心实践项目 | 6-9月 |
| **专家级（业界影响力）** | 在精通级基础上，开源贡献+论文发表+产品落地 | 持续 |

### 5.5 综合复习要点

针对AI模型-Agent技术岗位的，建议重点关注：

1. **记忆系统**：能画出完整的Agent记忆系统架构图，解释短期/长期记忆的构建、refine和加载方法
2. **轨迹数据合成**：能描述完整的轨迹合成管线，解释质量筛选策略，有SFT实操经验
3. **强化学习**：能解释PPO/GRPO的算法原理，理解Agent RL的特有挑战，有超参调优经验
4. **评估**：熟悉主流榜单，能分析其局限性，有实测经验
5. **Harness搭建**：能描述Harness的核心组件，有从零搭建或二次开发经验
6. **沙箱工程**：能设计完整的沙箱方案，有高并发运行经验
7. **工具使用**：熟练使用ClaudeCode/Cursor等工具，能完成全周期开发

---

### 5.6 持续学习资源

| 资源类型 | 具体资源 | 说明 |
|----------|----------|------|
| 论文追踪 | arXiv (cs.AI, cs.CL, cs.LG) | 每日追踪最新Agent论文 |
| 论文追踪 | Papers With Code - Agents | 带代码的论文集合 |
| 社区 | Hugging Face Daily Papers | HF每日精选论文 |
| 社区 | Reddit r/LocalLLaMA | 开源LLM社区 |
| 社区 | X/Twitter #LLMAgent | 关注Agent领域研究者 |
| 博客 | Anthropic Blog | Anthropic技术博客 |
| 博客 | OpenAI Blog | OpenAI技术博客 |
| 博客 | Lilian Weng's Blog | 深度技术博客 |
| 博客 | Sebastian Raschka's Blog | LLM训练相关深度文章 |
| Newsletter | The Batch (DeepLearning.AI) | AI领域新闻周刊 |
| Newsletter | TLDR AI | AI每日新闻 |
| 开源追踪 | GitHub Trending (Python/AI) | 追踪热门AI开源项目 |
| 课程 | MIT 6.S191: Introduction to Deep Learning | MIT深度学习课程 |
| 课程 | CMU 11-868: LLM Systems | CMU的LLM系统课程 |
| 课程 | Berkeley LLM Agents MOOC | Berkeley的Agent课程 |

---

> **文档说明**：本学习计划针对AI模型-Agent技术岗位编制，覆盖全部7个知识/技能领域的详细知识点梳理和学习资料规划。建议按照阶段路径系统学习，并在每个阶段完成对应的实践项目以巩固知识。所有推荐资料均来自公开可获取的书籍、课程、论文、博客和开源项目。
