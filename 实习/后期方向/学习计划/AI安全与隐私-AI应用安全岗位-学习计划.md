## 目录

- [1. 岗位概述](#1-岗位概述)
- [2. 知识体系全景图（知识树）](#2-知识体系全景图知识树)
- [3. 分领域详细知识点梳理](#3-分领域详细知识点梳理)
  - [3.1 Agent与LLM漏洞机理](#31-agent与llm漏洞机理)
  - [3.2 动态上下文安全与隐私数据架构](#32-动态上下文安全与隐私数据架构)
  - [3.3 工具调用与逻辑安全](#33-工具调用与逻辑安全)
  - [3.4 AI护栏与对齐理论](#34-ai护栏与对齐理论)
  - [3.5 Agent红队攻防实战（技能项1）](#35-agent红队攻防实战技能项1)
  - [3.6 防御组件研发与落地（技能项2）](#36-防御组件研发与落地技能项2)
  - [3.7 业务风险建模能力（技能项3）](#37-业务风险建模能力技能项3)
  - [3.8 隐私追踪与策略强制执行（技能项4）](#38-隐私追踪与策略强制执行技能项4)
- [4. 分领域学习资料规划](#4-分领域学习资料规划)
  - [4.1 AI/LLM安全基础](#41-aillm安全基础)
  - [4.2 Agent与LLM漏洞机理](#42-agent与llm漏洞机理)
  - [4.3 动态上下文安全与隐私数据架构](#43-动态上下文安全与隐私数据架构)
  - [4.4 工具调用与逻辑安全](#44-工具调用与逻辑安全)
  - [4.5 AI护栏与对齐理论](#45-ai护栏与对齐理论)
  - [4.6 Agent红队攻防实战](#46-agent红队攻防实战)
  - [4.7 防御组件研发与落地](#47-防御组件研发与落地)
  - [4.8 业务风险建模能力](#48-业务风险建模能力)
  - [4.9 隐私追踪与策略强制执行](#49-隐私追踪与策略强制执行)
- [5. 学习路径建议](#5-学习路径建议)

---

## 1. 岗位概述

### 1.1 岗位定位

AI安全与隐私-AI应用安全岗位是AI安全方向的核心研发岗位，聚焦于大语言模型（LLM）和AI Agent应用层面的安全攻防。该岗位要求候选人具备从漏洞机理分析、攻防实战、防御组件研发到隐私保护全链路的能力，覆盖AI应用安全的完整生命周期。

### 1.2 核心能力要求

| 维度 | 核心能力 | 对应掌握程度目标 |
|------|----------|------------------|
| 专业知识 | Agent与LLM漏洞机理 | 达到3-专家级 |
| 专业知识 | 动态上下文安全与隐私数据架构 | 达到3-专家级 |
| 专业知识 | 工具调用与逻辑安全 | 达到3-专家级 |
| 专业知识 | AI护栏与对齐理论 | 达到3-专家级 |
| 关键技能 | Agent红队攻防实战 | 达到3-专家级 |
| 关键技能 | 防御组件研发与落地 | 达到3-专家级 |
| 关键技能 | 业务风险建模能力 | 达到3-专家级 |
| 关键技能 | 隐私追踪与策略强制执行 | 达到3-专家级 |

### 1.3 岗位能力全景

```
AI应用安全人才能力模型
├── 攻：红队攻防 → 发现漏洞 → 构造payload → 自动化扫描
├── 防：防御组件 → 护栏系统 → 微调加固 → 架构设计
├── 策：风险建模 → 威胁建模 → 博弈分析 → 系统性风险评估
└── 隐：隐私保护 → 数据脱敏 → 污点追踪 → TEE/密码学
```

---

## 2. 知识体系全景图（知识树）

```
AI安全与隐私-AI应用安全
│
├── A. Agent与LLM漏洞机理
│   ├── A1. LLM安全基础
│   │   ├── A1.1 大模型架构基础（Transformer/注意力机制）
│   │   ├── A1.2 LLM训练流程（预训练→SFT→RLHF→对齐）
│   │   └── A1.3 LLM推理流程（Tokenization→采样解码→上下文窗口）
│   ├── A2. OWASP Top 10 for LLM
│   │   ├── A2.1 提示词注入（Prompt Injection）
│   │   ├── A2.2 不安全的输出处理
│   │   ├── A2.3 训练数据投毒
│   │   ├── A2.4 模型DoS
│   │   ├── A2.5 供应链漏洞
│   │   ├── A2.6 敏感信息泄露
│   │   ├── A2.7 不安全的插件设计
│   │   ├── A2.8 过度代理（Excessive Agency）
│   │   ├── A2.9 过度依赖
│   │   └── A2.10 模型盗窃
│   ├── A3. 注入攻击深度
│   │   ├── A3.1 直接提示词注入
│   │   ├── A3.2 间接提示词注入
│   │   ├── A3.3 多模态注入（图片隐藏载荷/音频隐写/视频）
│   │   └── A3.4 跨会话注入
│   ├── A4. 提权与越权
│   │   ├── A4.1 角色越权（Role Escalation）
│   │   ├── A4.2 权限提升（Privilege Escalation）
│   │   └── A4.3 插件调用越权
│   ├── A5. 模型注意力机制与攻击
│   │   ├── A5.1 注意力权重与攻击载荷关联
│   │   ├── A5.2 采样参数（Temperature/Top-p/Top-k）对攻击成功率影响
│   │   └── A5.3 上下文窗口利用与绕过
│   └── A6. 高级攻击理论
│       ├── A6.1 长链路决策中的攻击向量演变
│       ├── A6.2 规划失效（Planning Failure）根源分析
│       ├── A6.3 对抗性 suffix 攻击（GCG等）
│       └── A6.4 第一性原理分析模型行为
│
├── B. 动态上下文安全与隐私数据架构
│   ├── B1. LLM外部数据获取机制
│   │   ├── B1.1 搜索插件与实时检索
│   │   ├── B1.2 向量数据库与RAG
│   │   ├── B1.3 长上下文窗口注入
│   │   └── B1.4 非参数化知识处理
│   ├── B2. 多租户/多角色环境安全
│   │   ├── B2.1 多租户隔离机制
│   │   ├── B2.2 多角色权限模型
│   │   ├── B2.3 外部动态注入数据的权限隔离
│   │   └── B2.4 知识库/网络流的权限控制
│   ├── B3. 向量逃逸与上下文注入防御
│   │   ├── B3.1 向量数据库逃逸攻击
│   │   ├── B3.2 上下文间接注入防御
│   │   └── B3.3 RAG安全架构
│   ├── B4. 数据脱敏
│   │   ├── B4.1 静态数据脱敏（Data Masking）
│   │   ├── B4.2 动态数据脱敏
│   │   └── B4.3 脱敏算法与落地方法
│   ├── B5. 多模态/长上下文信息泄露
│   │   ├── B5.1 逆向成员推理攻击（Membership Inference Attack）
│   │   ├── B5.2 模型逆向攻击
│   │   └── B5.3 长上下文信息泄露机理
│   ├── B6. 隐私增强技术
│   │   ├── B6.1 差分隐私（Differential Privacy）
│   │   ├── B6.2 安全信息流分析（Information Flow Analysis）
│   │   └── B6.3 隐私量化方法
│   └── B7. 零信任AI数据安全
│       ├── B7.1 零信任架构原则
│       ├── B7.2 AI数据全生命周期安全
│       ├── B7.3 端到端隐私保护框架
│       └── B7.4 AGI级动态多源数据流安全
│
├── C. 工具调用与逻辑安全
│   ├── C1. Agent工具调用基础
│   │   ├── C1.1 API调用机制
│   │   ├── C1.2 数据库访问
│   │   ├── C1.3 Shell命令执行
│   │   └── C1.4 Function Calling协议
│   ├── C2. Agent场景下的传统安全威胁
│   │   ├── C2.1 SSRF（服务端请求伪造）在Agent中的演变
│   │   ├── C2.2 命令注入在Agent自动调用中的演变
│   │   ├── C2.3 API越权在Agent场景的演变
│   │   └── C2.4 防御策略
│   ├── C3. 形式化方法
│   │   ├── C3.1 线性时序逻辑（LTL）
│   │   ├── C3.2 Agent动作安全边界描述
│   │   ├── C3.3 模型检测（Model Checking）
│   │   └── C3.4 安全属性规约
│   └── C4. 自然语言到数学约束
│       ├── C4.1 自然语言指令的形式化转换
│       ├── C4.2 数学约束建模
│       ├── C4.3 长链路任务决策一致性
│       └── C4.4 决策可验证性
│
├── D. AI护栏与对齐理论
│   ├── D1. 安全基础
│   │   ├── D1.1 敏感词过滤
│   │   ├── D1.2 黑名单机制
│   │   └── D1.3 内容安全策略
│   ├── D2. RLHF基础
│   │   ├── D2.1 人类反馈强化学习定义
│   │   ├── D2.2 奖励模型训练
│   │   ├── D2.3 PPO策略优化
│   │   └── D2.4 RLHF对安全基线的影响
│   ├── D3. 安全护栏框架
│   │   ├── D3.1 NeMo Guardrails原理
│   │   ├── D3.2 有害意图拦截
│   │   ├── D3.3 意图重定向
│   │   └── D3.4 动态护栏策略
│   ├── D4. 后训练对齐方法
│   │   ├── D4.1 DPO（Direct Preference Optimization）
│   │   ├── D4.2 IPO/SimPO等变体
│   │   ├── D4.3 偏好对齐逻辑
│   │   └── D4.4 对齐税（Alignment Tax）分析
│   ├── D5. 对齐算法深度
│   │   ├── D5.1 损失函数视角分析对齐
│   │   ├── D5.2 对齐对模型安全基线的影响
│   │   ├── D5.3 对齐税的量化与缓解
│   │   └── D5.4 过度对齐风险
│   └── D6. 超级对齐前沿
│       ├── D6.1 Superalignment理论
│       ├── D6.2 多Agent对抗博弈
│       ├── D6.3 安全审计机制设计
│       └── D6.4 可扩展监督（Scalable Oversight）
│
├── E. Agent红队攻防实战（技能项1）
│   ├── E1. 越狱（Jailbreak）复现
│   │   ├── E1.1 公开Jailbreak模板复现
│   │   ├── E1.2 直接提示词注入模板
│   │   └── E1.3 经典越狱技术（DAN/Roleplay/Encoding等）
│   ├── E2. 间接注入Payload构造
│   │   ├── E2.1 针对业务场景的复杂间接注入
│   │   ├── E2.2 越权漏洞挖掘
│   │   └── E2.3 提权漏洞挖掘
│   ├── E3. 自动化红队
│   │   ├── E3.1 自动化脚本编写
│   │   ├── E3.2 LLM辅助工具进行决策路径探测
│   │   ├── E3.3 脆弱性扫描
│   │   └── E3.4 大规模攻击测试
│   └── E4. 新型攻击范式发现
│       ├── E4.1 跨模态注入
│       ├── E4.2 共谋绕过
│       ├── E4.3 自动化红队评估引擎设计
│       └── E4.4 从零发现新型攻击
│
├── F. 防御组件研发与落地（技能项2）
│   ├── F1. 基础输入/输出扫描
│   │   ├── F1.1 现成接口调用
│   │   ├── F1.2 输入过滤
│   │   └── F1.3 输出检测
│   ├── F2. 生产级护栏系统开发
│   │   ├── F2.1 LangChain框架
│   │   ├── F2.2 动态敏感信息拦截
│   │   ├── F2.3 意图重定向实现
│   │   └── F2.4 生产级部署
│   ├── F3. 微调与模型蒸馏防御
│   │   ├── F3.1 SFT（监督微调）
│   │   ├── F3.2 LoRA（低秩适配）
│   │   ├── F3.3 模型蒸馏
│   │   └── F3.4 推理性能与鲁棒性平衡
│   └── F4. 从0到1防御架构
│       ├── F4.1 神经符号监控器
│       ├── F4.2 创造性防御架构设计
│       └── F4.3 端到端防御体系
│
├── G. 业务风险建模能力（技能项3）
│   ├── G1. AI业务基础风险
│   │   ├── G1.1 AI助手风险点
│   │   ├── G1.2 自动化办公场景风险
│   │   └── G1.3 基本风险列举
│   ├── G2. 威胁建模
│   │   ├── G2.1 STRIDE方法
│   │   ├── G2.2 AI业务威胁建模
│   │   ├── G2.3 Agent行为安全基线
│   │   └── G2.4 审计策略制定
│   ├── G3. 多Agent博弈分析
│   │   ├── G3.1 博弈论模型
│   │   ├── G3.2 多Agent协作环境共谋风险
│   │   ├── G3.3 去中心化审计方案
│   │   └── G3.4 多Agent安全博弈
│   └── G4. AGI级风险评估
│       ├── G4.1 动态风险评估模型
│       ├── G4.2 策略漂移（Policy Drift）
│       ├── G4.3 系统性风险预测
│       └── G4.4 系统性风险防御
│
└── H. 隐私追踪与策略强制执行（技能项4）
    ├── H1. 静态脱敏
    │   ├── H1.1 手动脱敏
    │   ├── H1.2 配置化脱敏
    │   └── H1.3 敏感数据识别
    ├── H2. 动态隐私访问控制
    │   ├── H2.1 上下文识别
    │   ├── H2.2 动态流控制
    │   ├── H2.3 多角色Agent隐私策略
    │   └── H2.4 策略强制执行
    ├── H3. 污点追踪
    │   ├── H3.1 Taint Tracking原理
    │   ├── H3.2 细粒度动态隐私追踪
    │   ├── H3.3 跨应用数据流向监控
    │   └── H3.4 数据溯源
    └── H4. 高级隐私防御架构
        ├── H4.1 分布式Agent隐私防御
        ├── H4.2 TEE（可信执行环境）
        ├── H4.3 同态加密
        ├── H4.4 MPC（安全多方计算）
        └── H4.5 硬件可信根
```

---

## 3. 分领域详细知识点梳理

### 3.1 Agent与LLM漏洞机理

#### 3.1.1 LLM安全基础
- **大模型架构基础**：Transformer架构、自注意力机制（Self-Attention）、多头注意力、位置编码、前馈网络、残差连接与层归一化
- **LLM训练全流程**：预训练（Pre-training）→ 有监督微调（SFT）→ 人类反馈强化学习（RLHF）→ 安全对齐（Safety Alignment）
- **LLM推理流程**：Tokenization（BPE/SentencePiece）→ 自回归生成 → 采样策略（Greedy/Beam Search/Top-k/Top-p/Temperature）→ 上下文窗口管理（KV Cache/滑动窗口）
- **LLM安全特殊性**：非确定性输出、概率性推理、上下文依赖性、对自然语言的开放性

#### 3.1.2 OWASP Top 10 for LLM Applications（2023/2025版）
- **LLM01: Prompt Injection（提示词注入）**：直接注入、间接注入、通过外部数据源注入
- **LLM02: Insecure Output Handling（不安全的输出处理）**：LLM输出未经验证直接传递给后端执行
- **LLM03: Training Data Poisoning（训练数据投毒）**：在训练数据中植入后门触发器
- **LLM04: Model DoS（模型拒绝服务）**：通过精心构造输入消耗过多资源
- **LLM05: Supply Chain Vulnerabilities（供应链漏洞）**：第三方模型、数据集、组件的漏洞
- **LLM06: Sensitive Information Disclosure（敏感信息泄露）**：模型泄露训练数据中的隐私信息
- **LLM07: Insecure Plugin Design（不安全的插件设计）**：插件缺乏输入验证、过度权限
- **LLM08: Excessive Agency（过度代理）**：Agent被授予过多权限，可执行未预期操作
- **LLM09: Overreliance（过度依赖）**：系统或用户过度信任LLM输出
- **LLM10: Model Theft（模型盗窃）**：通过API查询窃取模型参数

#### 3.1.3 注入攻击深度解析
- **直接提示词注入**：攻击者在用户输入中直接嵌入恶意指令，覆盖系统提示
  - 指令覆盖（Instruction Override）
  - 角色劫持（Role Hijacking）
  - 虚构权限（Fabricated Authority）
- **间接提示词注入**：攻击载荷通过外部数据源（网页、文档、图片）间接进入LLM上下文
  - 网页内容注入
  - 文档元数据注入
  - 搜索结果注入
- **多模态注入**：
  - 图片隐藏载荷（Steganography/隐写术、像素级编码、不可见文本层）
  - 音频隐写（频域隐藏、超声波载荷）
  - 视频帧嵌入
- **跨会话注入**：利用持久化记忆或共享上下文跨会话传播攻击

#### 3.1.4 提权与越权
- **角色越权**：通过注入让模型扮演更高权限角色
- **权限提升**：利用Agent的多步骤决策链逐步获取更高权限
- **插件调用越权**：诱导Agent调用超出授权范围的插件或以超出预期的方式调用插件

#### 3.1.5 模型注意力机制与攻击
- **注意力权重与攻击载荷关联**：攻击载荷如何操纵注意力分布，使模型关注恶意指令而忽略安全约束
- **采样参数影响**：
  - Temperature：高温度增加随机性，可能绕过安全过滤；低温度更确定但可能更易被确定性攻击
  - Top-p/Top-k：截断策略对攻击成功率的影响
  - 重复惩罚（Repetition Penalty）：对重复型攻击的影响
- **上下文窗口利用**：长上下文中的"迷失在中间"（Lost in the Middle）现象及其在攻击中的利用

#### 3.1.6 高级攻击理论
- **长链路决策攻击向量演变**：在多步骤Agent决策链中，攻击如何从一个步骤传播到下一个步骤，逐步累积影响
- **规划失效（Planning Failure）根源**：
  - 目标偏移（Goal Drift）：Agent在长链路任务中偏离原始目标
  - 约束遗忘（Constraint Forgetting）：安全约束在长上下文中被"遗忘"
  - 子目标冲突（Sub-goal Conflict）：多个子目标之间的冲突导致安全边界被突破
- **对抗性后缀攻击**：GCG（Greedy Coordinate Gradient）等基于梯度的对抗性攻击方法
- **第一性原理分析**：从模型的基本工作原理（概率生成、注意力机制）出发分析安全失效的根本原因

---

### 3.2 动态上下文安全与隐私数据架构

#### 3.2.1 LLM外部数据获取机制
- **搜索插件与实时检索**：Agent通过搜索引擎获取实时信息的安全风险
- **向量数据库与RAG**：
  - 向量嵌入（Embedding）与相似性搜索
  - RAG（Retrieval-Augmented Generation）架构
  - 向量数据库（Pinecone/Weaviate/Chroma/Faiss）
  - RAG中的安全边界
- **长上下文窗口注入**：大上下文窗口（128K/1M tokens）带来的安全挑战
- **非参数化知识处理**：模型如何区分参数化知识（训练获得）与非参数化知识（外部输入），以及由此产生的安全风险

#### 3.2.2 多租户/多角色环境安全
- **多租户隔离机制**：不同租户之间的数据隔离、上下文隔离、向量库隔离
- **多角色权限模型**：RBAC（基于角色的访问控制）在AI Agent中的适配
- **外部动态注入数据的权限隔离**：知识库、网络流等动态数据源的权限控制
- **知识库权限控制**：不同权限级别的知识库条目访问控制

#### 3.2.3 向量逃逸与上下文注入防御
- **向量数据库逃逸攻击**：
  - 嵌入空间攻击（Embedding Space Attack）
  - 相似性操纵
  - 向量投毒
- **上下文间接注入防御**：
  - 输入消毒（Input Sanitization）
  - 上下文隔离标记
  - 指令层级（Instruction Hierarchy）
- **RAG安全架构**：
  - 检索结果验证
  - 来源可信度评估
  - 检索内容沙箱化

#### 3.2.4 数据脱敏
- **静态数据脱敏（Data Masking）**：
  - 替换/掩码（Masking）
  - 泛化（Generalization）
  - 扰动（Perturbation）
  - 假名化（Pseudonymization）
- **动态数据脱敏**：根据访问者角色和上下文动态决定脱敏策略
- **脱敏算法**：k-匿名、l-多样性、t-接近性
- **落地方法**：在数据管道中集成脱敏组件，Pipeline级别的脱敏策略

#### 3.2.5 多模态/长上下文信息泄露
- **逆向成员推理攻击（Membership Inference Attack）**：判断特定数据是否在训练集中
- **模型逆向攻击（Model Inversion Attack）**：从模型输出反推训练数据
- **属性推断攻击（Attribute Inference Attack）**：推断训练数据的属性
- **长上下文信息泄露**：在超长上下文中，早期输入的敏感信息可能在后续输出中泄露

#### 3.2.6 隐私增强技术
- **差分隐私（Differential Privacy）**：
  - ε-差分隐私定义
  - Laplace机制、Gaussian机制
  - DP-SGD（差分隐私随机梯度下降）
  - 隐私预算管理
- **安全信息流分析（Information Flow Analysis）**：
  - 信息流控制模型
  - 安全类型系统
  - 非干扰性（Non-interference）
  - 动态信息流追踪
- **隐私量化方法**：
  - 隐私度量指标
  - 风险-效用权衡
  - 隐私预算分配

#### 3.2.7 零信任AI数据安全
- **零信任架构原则**：永不信任、始终验证、最小权限、假设被入侵
- **AI数据全生命周期安全**：采集→存储→处理→传输→销毁各阶段的安全控制
- **端到端隐私保护框架**：从数据输入到模型输出的全链路隐私保护
- **AGI级动态多源数据流安全**：面向未来AGI系统处理多源异构数据流的安全架构

---

### 3.3 工具调用与逻辑安全

#### 3.3.1 Agent工具调用基础
- **API调用机制**：REST/GraphQL/gRPC API调用、OpenAPI规范、Function Calling协议
- **数据库访问**：SQL/NoSQL查询生成与执行、参数化查询
- **Shell命令执行**：命令构造、参数传递、权限管理
- **Function Calling协议**：OpenAI Function Calling、Anthropic Tool Use、MCP（Model Context Protocol）

#### 3.3.2 Agent场景下的传统安全威胁
- **SSRF在Agent中的演变**：
  - Agent可自主决定访问的URL，扩大SSRF攻击面
  - 内网探测、云元数据服务访问
  - 防御：URL白名单、网络隔离、响应验证
- **命令注入在Agent自动调用中的演变**：
  - Agent构造命令时的注入风险
  - 参数拼接注入
  - 防御：参数化执行、命令白名单、沙箱化
- **API越权在Agent场景的演变**：
  - 水平越权：Agent访问其他用户数据
  - 垂直越权：Agent执行超出权限级别的操作
  - 防御：权限校验中间件、最小权限原则、操作审计

#### 3.3.3 形式化方法
- **线性时序逻辑（LTL）**：
  - LTL语法与语义
  - 时序算子（□ always, ◇ eventually, ○ next, U until）
  - 安全属性规约（Safety Properties）
  - 活性属性规约（Liveness Properties）
- **Agent动作安全边界描述**：
  - 用LTL公式描述Agent允许/禁止的行为序列
  - 安全不变式（Safety Invariants）
  - 动作前置条件与后置条件
- **模型检测（Model Checking）**：
  - SPIN/NuSMV工具
  - 状态空间爆炸问题
  - 抽象与符号化方法
- **安全属性规约**：
  - 不可达性（Unreachability）：危险状态不可达
  - 互斥性（Mutual Exclusion）：冲突操作不可同时执行
  - 活性（Liveness）：安全检查最终被执行

#### 3.3.4 自然语言到数学约束
- **自然语言指令的形式化转换**：
  - 语义解析（Semantic Parsing）
  - 意图形式化表示
  - 约束提取
- **数学约束建模**：
  - 线性约束、非线性约束
  - 不等式系统
  - 优化目标
- **长链路任务决策一致性**：
  - 决策树一致性验证
  - 多步骤约束传播
  - 回溯与冲突解决
- **决策可验证性**：
  - 决策证明（Decision Proofs）
  - 可审计日志
  - 形式化验证

---

### 3.4 AI护栏与对齐理论

#### 3.4.1 安全基础
- **敏感词过滤**：基于关键词/正则表达式的过滤、基于语义的过滤
- **黑名单机制**：IP黑名单、用户黑名单、内容黑名单
- **内容安全策略**：多级内容分类、违规内容识别、分级响应

#### 3.4.2 RLHF基础
- **RLHF定义与流程**：
  - 步骤1：SFT（有监督微调）—— 在高质量数据上微调基座模型
  - 步骤2：奖励模型训练 —— 训练一个预测人类偏好的模型
  - 步骤3：PPO强化学习 —— 使用奖励模型优化策略模型
- **奖励模型训练**：
  - 偏好数据收集（Pairwise Comparison）
  - Bradley-Terry模型
  - 奖励模型损失函数
- **PPO策略优化**：
  - 策略梯度（Policy Gradient）
  - 优势函数估计（GAE）
  - KL散度惩罚（防止策略偏离过远）
  - 裁剪目标函数
- **RLHF对安全基线的影响**：RLHF如何让模型学会拒绝有害请求、遵循安全准则

#### 3.4.3 安全护栏框架
- **NeMo Guardrails原理**：
  - Colang编程语言
  - 输入/输出/对话/检索护栏
  - 基于规则与基于模型的混合护栏
  - 护栏流程编排
- **有害意图拦截**：
  - 意图分类器
  - 安全分类器
  - 多级拦截策略
- **意图重定向**：
  - 检测到有害意图后安全重定向
  - 对话主题引导
  - 拒绝策略设计
- **动态护栏策略**：
  - 基于上下文的动态护栏调整
  - 用户画像驱动的个性化安全策略
  - 风险评分驱动的分级响应

#### 3.4.4 后训练对齐方法
- **DPO（Direct Preference Optimization）**：
  - 直接偏好优化原理
  - DPO损失函数：$L_{DPO} = -\log\sigma(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)})$
  - DPO vs RLHF：无需训练奖励模型，直接优化策略
- **DPO变体**：
  - IPO（Identity Preference Optimization）
  - SimPO（Simple Preference Optimization）
  - KTO（Kahneman-Tversky Optimization）
  - ORPO
- **偏好对齐逻辑**：如何从人类偏好数据中学习安全行为准则
- **对齐税（Alignment Tax）分析**：
  - 对齐导致的能力下降
  - 损失函数视角：对齐目标与能力目标的冲突
  - 对齐税的量化指标
  - 缓解策略（如混合训练、能力恢复微调）

#### 3.4.5 对齐算法深度
- **损失函数视角分析对齐**：
  - 对齐目标的数学表达
  - 梯度冲突分析
  - 多目标优化
- **对齐对模型安全基线的影响**：
  - 对齐强度与安全性的关系
  - 过度对齐导致的过度拒绝
  - 安全性与有用性的权衡
- **对齐税的量化与缓解**：
  - 能力评估基准上的性能差异
  - 能力恢复技术
  - 对齐税的数学建模
- **过度对齐风险**：
  - Sycophancy（谄媚）
  - Over-refusal（过度拒绝）
  - 对齐税对下游应用的影响

#### 3.4.6 超级对齐前沿
- **Superalignment理论**：
  - OpenAI Superalignment计划
  - 弱到强监督（Weak-to-Strong Supervision）
  - 可扩展监督（Scalable Oversight）
  - 自动化对齐研究
- **多Agent对抗博弈**：
  - 红队Agent vs 蓝队Agent
  - 对抗训练
  - 博弈论均衡分析
- **安全审计机制设计**：
  - 自动化安全审计
  - 行为一致性监控
  - 异常检测
- **可扩展监督（Scalable Oversight）**：
  - 辩论（Debate）
  - 递归奖励建模（Recursive Reward Modeling）
  - AI安全 via 辩论

---

### 3.5 Agent红队攻防实战（技能项1）

#### 3.5.1 越狱（Jailbreak）复现
- **经典越狱技术**：
  - DAN（Do Anything Now）系列
  - 角色扮演越狱（Roleplay Jailbreak）
  - 编码绕过（Base64/Unicode/ROT13）
  - 翻译攻击（多语言绕过）
  - 前缀注入（Prefix Injection）
  - 拒绝抑制（Refusal Suppression）
- **公开Jailbreak模板复现**：从JailbreakDB、PromptInject等开源项目复现
- **直接提示词注入模板**：覆盖系统提示、伪造系统消息

#### 3.5.2 间接注入Payload构造
- **业务场景定向构造**：针对特定业务流程设计间接注入
- **越权漏洞挖掘**：通过注入测试Agent是否执行超出权限的操作
- **提权漏洞挖掘**：通过多步骤交互逐步提升权限

#### 3.5.3 自动化红队
- **自动化脚本编写**：批量生成测试用例、自动化执行攻击流程
- **LLM辅助决策路径探测**：使用LLM作为攻击助手，自动探索Agent决策树
- **脆弱性扫描**：系统化扫描Agent的所有输入面和工具调用面
- **大规模攻击测试**：Fuzzing、变异测试、对抗性生成

#### 3.5.4 新型攻击范式发现
- **跨模态注入**：图片/音频/视频中的隐藏攻击载荷
- **共谋绕过**：多个Agent协作时通过共谋绕过安全限制
- **自动化红队评估引擎设计**：从攻击生成→执行→效果评估→报告生成的全自动化流程

---

### 3.6 防御组件研发与落地（技能项2）

#### 3.6.1 基础输入/输出扫描
- **现成接口调用**：OpenAI Moderation API、Azure Content Safety、Perspective API
- **输入过滤**：关键词过滤、语义相似度检测、注入特征检测
- **输出检测**：有害内容检测、PII泄露检测、越权操作检测

#### 3.6.2 生产级护栏系统开发
- **LangChain框架**：
  - LangChain安全组件
  - LLMGuard集成
  - 自定义Guardrail Chain
- **动态敏感信息拦截**：
  - 实时PII检测（NER-based）
  - 正则模式匹配
  - 上下文感知的敏感信息识别
- **意图重定向实现**：
  - 安全意图分类
  - 重定向对话流
  - 优雅降级策略
- **生产级部署**：
  - 高可用架构
  - 延迟优化
  - 监控与告警

#### 3.6.3 微调与模型蒸馏防御
- **SFT（监督微调）**：在安全示例上微调模型以增强拒绝能力
- **LoRA（低秩适配）**：
  - 低秩分解原理
  - 安全导向的LoRA训练
  - 多LoRA融合
- **模型蒸馏**：
  - Teacher-Student蒸馏
  - 安全能力蒸馏
  - 蒸馏效率优化
- **推理性能与鲁棒性平衡**：在不显著增加延迟的前提下提升防御能力

#### 3.6.4 从0到1防御架构
- **神经符号监控器（Neuro-Symbolic Monitor）**：
  - 神经网络（模式识别）+ 符号推理（规则验证）的混合架构
  - 实时动作验证
  - 安全属性检查
- **创造性防御架构设计**：结合形式化验证、运行时监控、动态策略的端到端防御体系

---

### 3.7 业务风险建模能力（技能项3）

#### 3.7.1 AI业务基础风险
- **AI助手风险点**：信息泄露、误导性输出、权限滥用、幻觉风险
- **自动化办公场景风险**：邮件自动发送风险、文档操作风险、日程管理风险
- **基本风险列举与分类**：按CIA三要素（机密性/完整性/可用性）分类AI风险

#### 3.7.2 威胁建模
- **STRIDE方法**：
  - Spoofing（仿冒）：伪造Agent身份
  - Tampering（篡改）：篡改Agent输入/输出/知识库
  - Repudiation（抵赖）：Agent操作不可追溯
  - Information Disclosure（信息泄露）：敏感信息通过Agent泄露
  - Denial of Service（拒绝服务）：Agent资源耗尽
  - Elevation of Privilege（提权）：Agent获取未授权权限
- **AI业务威胁建模**：将STRIDE适配到AI/Agent场景
- **Agent行为安全基线**：定义Agent允许的行为边界、操作白名单
- **审计策略制定**：操作日志、决策追溯、异常行为告警

#### 3.7.3 多Agent博弈分析
- **博弈论模型**：
  - 静态博弈与动态博弈
  - 完全信息与不完全信息博弈
  - 纳什均衡
  - 重复博弈
- **多Agent协作环境共谋风险**：
  - 隐式共谋：Agent通过观察彼此行为协调策略
  - 显式共谋：Agent通过通信信道协商
  - 共谋绕过安全限制
- **去中心化审计方案**：
  - 分布式审计节点
  - 共识机制
  - 防篡改审计日志

#### 3.7.4 AGI级风险评估
- **动态风险评估模型**：实时监控AI系统的风险水平，动态调整安全策略
- **策略漂移（Policy Drift）**：
  - 在线学习导致的策略偏移
  - 微调/更新导致的行为变化
  - 漂移检测与度量
- **系统性风险预测**：预测AI系统在大规模部署中的系统性风险
- **系统性风险防御**：熔断机制、回滚策略、隔离方案

---

### 3.8 隐私追踪与策略强制执行（技能项4）

#### 3.8.1 静态脱敏
- **手动脱敏**：人工识别并处理敏感数据
- **配置化脱敏**：基于规则配置的自动脱敏（正则匹配、字段映射）
- **敏感数据识别**：NER（命名实体识别）驱动的PII检测、数据分类分级

#### 3.8.2 动态隐私访问控制
- **上下文识别**：根据对话上下文、用户角色、任务类型动态判断数据访问权限
- **动态流控制**：运行时数据流控制，基于标签的安全策略
- **多角色Agent隐私策略**：不同角色Agent的数据访问差异化策略
- **策略强制执行**：Policy Enforcement Point（PEP）、Policy Decision Point（PDP）

#### 3.8.3 污点追踪
- **Taint Tracking原理**：
  - 污点源（Taint Source）：敏感数据输入点
  - 污点传播（Taint Propagation）：数据在计算过程中的传播
  - 污点汇（Taint Sink）：敏感数据输出点
  - 污点策略（Taint Policy）：何时触发告警/阻断
- **细粒度动态隐私追踪**：
  - Token级别追踪
  - 跨函数/跨模块追踪
  - 上下文感知的污点传播
- **跨应用数据流向监控**：
  - 数据血缘（Data Lineage）
  - 数据流图（Data Flow Graph）
  - 实时流向审计
- **数据溯源**：追溯数据的来源、变换历史、使用记录

#### 3.8.4 高级隐私防御架构
- **分布式Agent隐私防御**：
  - 去中心化隐私保护
  - Agent间数据最小化原则
  - 零知识验证
- **TEE（可信执行环境）**：
  - Intel SGX
  - ARM TrustZone
  - AMD SEV
  - TEE中的模型推理
  - 远程证明（Remote Attestation）
- **同态加密**：
  - 全同态加密（FHE）
  - 部分同态加密（PHE）
  - CKKS/BGV方案
  - 同态加密在隐私推理中的应用
- **MPC（安全多方计算）**：
  - 秘密共享（Secret Sharing）
  - 混淆电路（Garbled Circuits）
  - 不经意传输（Oblivious Transfer）
  - MPC在联邦学习中的应用
- **硬件可信根**：
  - TPM（可信平台模块）
  - 硬件安全模块（HSM）
  - 安全启动链

---

## 4. 分领域学习资料规划

### 4.1 AI/LLM安全基础

> 这是进入AI安全领域的前置基础，建议首先学习。

#### 4.1.1 书籍

| # | 名称 | 作者/来源 | 类型 | 难度 | 推荐理由 |
|---|------|----------|------|------|----------|
| 1 | **《深度学习》** | Ian Goodfellow, Yoshua Bengio, Aaron Courville | 书籍（"花书"） | 中-高 | AI安全必读基础，第3章概率论、第6章深度前馈网络、第8章优化、第14章自编码器，理解模型内部机制是分析漏洞的前提。Goodfellow本人就是对抗样本研究的开创者 |
| 2 | **《Natural Language Processing with Transformers》** | Lewis Tunstall, Leandro von Werra, Thomas Wolf | 书籍 | 中 | O'Reilly出版，系统讲解Transformer架构和HuggingFace生态，理解LLM工作原理的实践指南 |
| 3 | **《Build a Large Language Model (From Scratch)》** | Sebastian Raschka | 书籍 | 中 | 从零构建GPT模型，深入理解Tokenization、注意力机制、训练流程，是理解LLM漏洞机理的根基 |
| 4 | **《Hands-On Large Language Models》** | Jay Alammar, Maarten Grootendorst | 书籍 | 中 | 2024年出版，系统讲解LLM的架构、训练、微调、对齐全流程，含可视化讲解 |
| 5 | **《AI Safety》** | Dan Hendrycks (主编) | 在线书籍/教材 | 中-高 | Center for AI Safety出品，涵盖AI安全的核心理念、风险分类、对齐方法，是AI安全领域的入门权威读物 |

#### 4.1.2 大学课程

| # | 名称 | 来源 | 难度 | 推荐理由 |
|---|------|------|------|----------|
| 1 | **CS224N: Natural Language Processing with Deep Learning** | Stanford University | 中-高 | NLP领域标杆课程，系统讲解Transformer、注意力机制、LLM训练，理解LLM内部机制的必修课。课程网址：https://web.stanford.edu/class/cs224n/ |
| 2 | **CS324: Large Language Models** | Stanford University | 高 | 专门讲解大语言模型，涵盖模型能力、训练、对齐、安全等主题。课程网址：https://stanford-cs324.github.io/ |
| 3 | **MIT 6.S191: Introduction to Deep Learning** | MIT | 中 | MIT深度学习入门课程，涵盖Transformer、生成模型等基础，含对抗鲁棒性内容。课程网址：http://introtodeeplearning.com/ |
| 4 | **Safe AI: Alignment and Governance** | UC Berkeley | 高 | 专注于AI安全与对齐的课程，涵盖RLHF、可扩展监督、AI治理等前沿主题 |

#### 4.1.3 博客与在线资源

| # | 名称 | 来源 | 难度 | 推荐理由 |
|---|------|------|------|----------|
| 1 | **Lilian Weng's Blog: "Prompt Injection"** | Lilian Weng (OpenAI) | 中-高 | AI安全领域最有影响力的个人技术博客之一，系统梳理提示词注入攻击的全貌。文章网址：https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/ |
| 2 | **Lilian Weng's Blog: "LLM Powered Autonomous Agents"** | Lilian Weng (OpenAI) | 中 | 系统讲解Agent架构、工具调用、记忆机制，理解Agent安全的基础。文章网址：https://lilianweng.github.io/posts/2023-06-23-agent/ |
| 3 | **Hugging Face NLP Course** | Hugging Face | 入门-中 | 免费在线课程，系统学习Transformer和LLM生态，实践导向。课程网址：https://huggingface.co/learn/nlp-course |
| 4 | **Illustrated Transformer** | Jay Alammar | 入门-中 | 经典可视化讲解Transformer架构，理解注意力机制的最佳入门资料。文章网址：https://jalammar.github.io/illustrated-transformer/ |
| 5 | **OpenAI Blog: "Alignment"** | OpenAI | 中-高 | OpenAI对齐研究博客系列，涵盖RLHF、Superalignment等前沿主题。网址：https://openai.com/research/alignment |

#### 4.1.4 开源项目/代码库

| # | 名称 | 来源 | 难度 | 推荐理由 |
|---|------|------|------|----------|
| 1 | **Hugging Face Transformers** | Hugging Face | 中 | LLM生态核心库，理解模型架构、训练、推理的最佳实践代码库。GitHub: https://github.com/huggingface/transformers |
| 2 | **nanoGPT** | Andrej Karpathy | 入门-中 | 最简洁的GPT实现，约300行代码，理解LLM核心原理的最佳学习项目。GitHub: https://github.com/karpathy/nanoGPT |
| 3 | **LLM Security Hub** | 社区维护 | 中 | LLM安全资源汇总，含攻击案例、防御工具、研究论文索引。GitHub: https://github.com/tldrsec/llm-security-hub |

---

### 4.2 Agent与LLM漏洞机理

#### 4.2.1 书籍

| # | 名称 | 作者/来源 | 类型 | 难度 | 推荐理由 |
|---|------|----------|------|------|----------|
| 1 | **《OWASP Top 10 for LLM Applications》** | OWASP | 在线文档/标准 | 入门-中 | LLM安全领域的权威标准，系统列举LLM应用的十大安全风险，是知识体系的核心框架。文档网址：https://owasp.org/www-project-top-10-for-large-language-model-applications/ |
| 2 | **《The Art of Prompt Injection》** | Kai Greshake et al. | 论文/博客系列 | 中-高 | 间接提示词注入攻击的系统性论述，作者是该领域的开创性研究者 |
| 3 | **《Adversarial Machine Learning》** | Anthony D. Joseph et al. | 书籍 | 高 | MIT Press出版，系统讲解对抗性机器学习的理论与实践，包含对抗样本、投毒攻击、模型窃取等 |
| 4 | **《Red Teaming for LLMs》** | 社区编撰 | 电子书 | 中 | 系统讲解LLM红队测试方法论，含攻击模板、自动化工具、评估框架 |

#### 4.2.2 大学课程

| # | 名称 | 来源 | 难度 | 推荐理由 |
|---|------|------|------|----------|
| 1 | **CS 294-165: AI Safety (UC Berkeley)** | UC Berkeley | 高 | UC Berkeley的AI安全课程，涵盖对齐、对抗鲁棒性、红队测试等，含大量前沿论文阅读。课程网址：https://aisafety.dance/ |
| 2 | **Safe and Robust AI (CMU 11-785)** | Carnegie Mellon University | 高 | CMU的AI安全课程，涵盖对抗攻击、形式化验证、安全强化学习 |
| 3 | **AI Safety Fundamentals** | BlueDot Impact | 中-高 | 免费在线AI安全课程，系统化学习AI安全理论与实践，含对齐、治理、技术风险等模块。课程网址：https://aisafetyfundamentals.com/ |

#### 4.2.3 博客与论文

| # | 名称 | 作者/来源 | 难度 | 推荐理由 |
|---|------|------|------|----------|
| 1 | **论文："Universal and Transferable Adversarial Attacks on Aligned Language Models"** | Andy Zou et al. | 高 | GCG攻击的开创性论文，提出基于梯度的通用对抗后缀攻击方法，是对抗性LLM攻击的里程碑。论文：https://arxiv.org/abs/2307.15043 |
| 2 | **论文："Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection"** | Kai Greshake et al. | 中-高 | 间接提示词注入攻击的奠基论文，展示通过网页/文档间接注入攻击LLM应用的方法。论文：https://arxiv.org/abs/2302.12173 |
| 3 | **论文："Jailbroken: How Does LLM Safety Training Fail?"** | Wei et al. (NeurIPS 2023) | 高 | 系统分析越狱攻击为何能成功，从理论和实践两个维度分析安全训练的失效原因。论文：https://arxiv.org/abs/2307.02483 |
| 4 | **论文："Ignore This Title and HackAPrompt: Exposing Systemic Weaknesses of LLMs through a Global Scale Prompt Hacking Competition"** | Schulhoff et al. | 中 | 大规模越狱竞赛的分析报告，系统化分类越狱技术。论文：https://arxiv.org/abs/2311.16119 |
| 5 | **论文："Lost in the Middle: How Language Models Use Long Contexts"** | Liu et al. | 中 | 分析LLM在长上下文中的性能衰减现象，对理解长上下文安全至关重要。论文：https://arxiv.org/abs/2307.03172 |
| 6 | **博客："Prompt Injection attack against LLM-integrated Applications"** | Liu et al. | 中 | 系统梳理提示词注入攻击的分类与防御，含攻击树模型。论文：https://arxiv.org/abs/2306.05499 |
| 7 | **论文："How Johnny Can Persuade LLMs to Jailbreak Them: Rethinking Persuasion to Challenge AI Safety by Humanizing LLMs"** | Zeng et al. | 中 | 从社会工程学角度分析越狱，提出 persuasion-based 攻击分类。论文：https://arxiv.org/abs/2311.13548 |
| 8 | **博客："LLM Attack & Defense"** | Simon Willison | 中 | 著名AI安全研究者Simon Willison的博客，持续追踪LLM安全事件与漏洞分析。网址：https://simonwillison.net/ |
| 9 | **论文："Image Hijacks: Adversarial Images can Control Generative AI Systems"** | Bailey et al. | 高 | 多模态注入攻击的前沿论文，展示如何通过图片劫持LLM生成。论文：https://arxiv.org/abs/2309.10270 |
| 10 | **论文："Adversarial Attacks on LLM Agents"** | 赵天雨 et al. | 中-高 | 专门针对Agent系统的对抗攻击论文，涵盖工具调用劫持、决策路径操纵等。论文：https://arxiv.org/abs/2402.11276 |

#### 4.2.4 开源项目/代码库

| # | 名称 | 来源 | 难度 | 推荐理由 |
|---|------|------|------|----------|
| 1 | **PyRIT (Python Risk Identification Toolkit)** | Microsoft | 中 | 微软开源的LLM红队测试框架，支持自动化生成攻击payload、多轮对话攻击、评分评估。是Agent红队实战的核心工具。GitHub: https://github.com/Azure/PyRIT |
| 2 | **Garak (LLM Vulnerability Scanner)** | NVIDIA | 中 | NVIDIA开源的LLM漏洞扫描器，支持多种攻击类型（提示注入、越狱、数据泄露等），可自动化扫描。GitHub: https://github.com/leondz/garak |
| 3 | **llm-attacks** | Andy Zou et al. | 高 | GCG对抗攻击的官方实现代码，学习梯度基对抗攻击的最佳代码库。GitHub: https://github.com/llm-attacks/llm-attacks |
| 4 | **PromptInject** | Liu et al. | 中 | 提示词注入攻击框架，系统化实现多种注入攻击方法。GitHub: https://github.com/agencyenterprise/PromptInject |
| 5 | **JailbreakBench** | Neel Patel et al. | 中 | 越狱攻击的标准化基准测试平台，含攻击/防御排行榜、评估指标。GitHub: https://github.com/JailbreakBench/jailbreakbench |
| 6 | **AdvBench** | Andy Zou et al. | 中 | 对抗行为基准数据集，用于评估LLM的安全性和鲁棒性。GitHub: https://github.com/llm-attacks/llm-attacks |
| 7 | **Tensor Trust** | UC Berkeley | 中 | 在线LLM越狱攻击/防御游戏平台，收集了大量真实攻击数据。网址：https://tensortrust.ai/ |
| 8 | **WMDP (Weapons of Mass Destruction Proxy)** | Center for AI Safety | 高 | 评估LLM危险知识遗忘的基准，用于安全对齐研究。GitHub: https://github.com/centerforaisafety/wmdp |

---

### 4.3 动态上下文安全与隐私数据架构

#### 4.3.1 书籍

| # | 名称 | 作者/来源 | 类型 | 难度 | 推荐理由 |
|---|------|----------|------|------|----------|
| 1 | **《The Algorithmic Foundations of Differential Privacy》** | Cynthia Dwork, Aaron Roth | 书籍 | 高 | 差分隐私领域的奠基性著作，由差分隐私创始人撰写，系统讲解DP的理论基础、机制设计和隐私组合定理。在线免费：https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf |
| 2 | **《Differential Privacy: A Primer for Non-Technical Audiences》** | Kobbi Nissim et al. | 在线文档 | 入门-中 | 差分隐私的通俗入门介绍，适合先读此书建立直觉再深入理论 |
| 3 | **《Information Flow Analysis: Theory and Practice》** | Sebastian Hunt | 书籍 | 高 | 信息流分析的系统教材，涵盖安全类型系统、非干扰性、静态与动态分析 |
| 4 | **《Zero Trust Networks: Building Secure Systems in Untrusted Networks》** | Evan Gilman, Doug Barth | 书籍 | 中 | O'Reilly出版的零信任网络权威指南，讲解零信任架构的核心原则与实践方法 |
| 5 | **《RAG with Production LLMs》** | LangChain | 在线文档 | 中 | LangChain的RAG生产实践文档，含RAG安全架构设计 |

#### 4.3.2 大学课程

| # | 名称 | 来源 | 难度 | 推荐理由 |
|---|------|------|------|----------|
| 1 | **CS 265: Cryptography and Network Security** | Stanford University | 高 | 密码学与网络安全基础课程，涵盖零知识证明、安全多方计算、同态加密等隐私技术基础 |
| 2 | **Privacy Technologies (CMU 17-858)** | Carnegie Mellon University | 高 | 专门讲解隐私技术的课程，涵盖差分隐私、联邦学习、匿名化技术 |
| 3 | **Secure and Private Systems (MIT 6.S982)** | MIT | 高 | MIT的隐私保护系统课程，涵盖信息流控制、差分隐私、TEE等 |

#### 4.3.3 博客与论文

| # | 名称 | 作者/来源 | 难度 | 推荐理由 |
|---|------|------|------|----------|
| 1 | **论文："Membership Inference Attacks Against Machine Learning Models"** | Reza Shokri et al. | 中-高 | 成员推理攻击的奠基论文，系统提出攻击框架。论文：https://arxiv.org/abs/1610.01242 |
| 2 | **论文："Deep Learning with Differential Privacy"** | Martin Abadi et al. (Google) | 高 | DP-SGD的开创性论文，将差分隐私应用于深度学习训练。论文：https://arxiv.org/abs/1607.00133 |
| 3 | **论文："Extracting Training Data from Large Language Models"** | Nicholas Carlini et al. | 高 | 从LLM中提取训练数据的开创性论文，展示LLM的隐私泄露风险。论文：https://arxiv.org/abs/2012.07805 |
| 4 | **论文："Scalable Extraction of Training Data from (Production) Language Models"** | Carlini et al. | 高 | 扩展到生产级LLM的训练数据提取攻击。论文：https://arxiv.org/abs/2311.17035 |
| 5 | **论文："Vector Database Security: Attack Vectors and Defenses"** | 社区研究 | 中 | 向量数据库安全分析，涵盖向量逃逸、投毒等攻击 |
| 6 | **博客："RAG Security: Risks and Mitigations"** | Lakera | 中 | Lakera安全公司的RAG安全分析博客，系统讲解RAG架构的安全风险与防御。网址：https://lakera.ai/ |
| 7 | **论文："Privacy-Preserving RAG: A Comprehensive Survey"** | 多位作者 | 中 | 隐私保护RAG的综合调研，涵盖差分隐私RAG、联邦RAG等方向 |
| 8 | **博客："Zero Trust Architecture for AI"** | NIST | 中 | NIST的零信任架构在AI系统中的应用指南 |

#### 4.3.4 开源项目/代码库

| # | 名称 | 来源 | 难度 | 推荐理由 |
|---|------|------|------|----------|
| 1 | **Opacus** | Meta (Facebook) | 中 | Meta开源的差分隐私训练库，支持PyTorch，是DP-SGD实践的核心工具。GitHub: https://github.com/pytorch/opacus |
| 2 | **TextAttack** | QData Lab | 中 | NLP对抗攻击框架，支持多种攻击方法，可用于测试LLM鲁棒性。GitHub: https://github.com/QData/TextAttack |
| 3 | **Privacy Meter** | NUS Data Privacy Lab | 中 | 隐私风险评估工具，量化机器学习模型的隐私泄露风险。GitHub: https://github.com/privacy-tech-lab/privacy-meter |
| 4 | **ChromaDB** | Chroma | 入门-中 | 开源向量数据库，学习向量数据库安全架构的最佳实践项目。GitHub: https://github.com/chroma-core/chroma |
| 5 | **Weaviate** | Weaviate | 中 | 开源向量搜索引擎，含安全模块，学习RAG安全架构。GitHub: https://github.com/weaviate/weaviate |
| 6 | **PrivateGPT** | Zylon | 中 | 私有化GPT实现，展示如何在隐私保护下运行LLM，含数据隔离架构。GitHub: https://github.com/zylon-ai/private-gpt |

---

### 4.4 工具调用与逻辑安全

#### 4.4.1 书籍

| # | 名称 | 作者/来源 | 类型 | 难度 | 推荐理由 |
|---|------|----------|------|------|----------|
| 1 | **《Principles of Model Checking》** | Christel Baier, Joost-Pieter Katoen | 书籍 | 高 | 模型检测领域的权威教材，系统讲解LTL、CTL、模型检测算法，是形式化验证Agent安全的基础 |
| 2 | **《Logic in Computer Science: Modelling and Reasoning about Systems》** | Michael Huth, Mark Ryan | 书籍 | 中-高 | 计算机逻辑的经典教材，涵盖命题逻辑、时序逻辑、模型检测，适合建立形式化方法基础 |
| 3 | **《Web Security for Developers》** | Malcolm McDonald | 书籍 | 入门-中 | Web安全基础，理解SSRF、命令注入、API越权等传统安全威胁的前提 |
| 4 | **《The Web Application Hacker's Handbook》** | Dafydd Stuttard, Marcus Pinto | 书籍 | 中 | Web安全攻防圣经，系统讲解OWASP Top 10传统Web漏洞，Agent安全需要在此基础上适配 |
| 5 | **《Software Verification and Analysis using Z3** | Leonardo de Moura et al. | 在线文档 | 高 | Z3 SMT求解器的使用指南，用于Agent动作的形式化验证 |

#### 4.4.2 大学课程

| # | 名称 | 来源 | 难度 | 推荐理由 |
|---|------|------|------|----------|
| 1 | **CS 254: Program Verification** | Stanford University | 高 | Stanford程序验证课程，涵盖形式化规约、模型检测、定理证明，是Agent逻辑安全的形式化方法基础 |
| 2 | **MIT 6.045J: Automata, Computability, and Complexity** | MIT | 中 | 自动机与可计算性理论，理解形式化方法的理论基础 |
| 3 | **CMU 15-414: Bug Catching** | Carnegie Mellon University | 中 | CMU的软件验证课程，涵盖形式化方法、模型检测、定理证明的实践应用 |
| 4 | **Web Security (Stanford CS142)** | Stanford University | 中 | Web安全基础课程，理解传统Web安全威胁 |

#### 4.4.3 博客与论文

| # | 名称 | 作者/来源 | 难度 | 推荐理由 |
|---|------|------|------|------|
| 1 | **论文："AgentDojo: A Dynamic Environment to Evaluate Prompt Injection Attacks and Defenses for LLM Agents"** | Edoardo Debenedetti et al. | 中-高 | 专门评估Agent工具调用安全的基准环境，含多种间接注入攻击场景。论文：https://arxiv.org/abs/2406.13352 |
| 2 | **论文："ToolEmu: Identifying the Risks of LLM Agents with an LLM Emulator"** | Yang et al. | 中 | 模拟Agent工具调用风险的开创性论文，系统化分析工具调用安全。论文：https://arxiv.org/abs/2309.15817 |
| 3 | **论文："Breaking Agents: Compromising Autonomous LLM Agents Through Malfunction Amplification"** | Andriushchenko et al. | 中-高 | 分析Agent故障放大的安全风险。论文：https://arxiv.org/abs/2403.02631 |
| 4 | **博客："MCP (Model Context Protocol) Security Analysis"** | Anthropic/MCP社区 | 中 | MCP协议的安全分析，理解Agent工具调用协议的安全设计。网址：https://modelcontextprotocol.io/ |
| 5 | **论文："Formal Verification of AI Agents: A Survey"** | 多位作者 | 高 | AI Agent形式化验证的综合调研，涵盖LTL应用、模型检测在AI系统中的实践 |
| 6 | **博客："SSRF in AI Agents: New Attack Surfaces"** | PortSwigger/安全社区 | 中 | 分析SSRF在AI Agent场景下的新攻击面，含实际案例 |

#### 4.4.4 开源项目/代码库

| # | 名称 | 来源 | 难度 | 推荐理由 |
|---|------|------|------|----------|
| 1 | **SPIN Model Checker** | Holzmann et al. | 高 | LTL模型检测的经典工具，用于验证Agent行为是否满足安全属性。GitHub: https://github.com/nimble-code/Spin |
| 2 | **NuSMV** | FBK-IRST | 高 | 符号模型检测器，支持LTL/CTL属性验证，用于Agent安全边界验证。网址：https://nusmv.fbk.eu/ |
| 3 | **Z3 SMT Solver** | Microsoft Research | 高 | 微软开源的SMT求解器，用于Agent动作约束的自动验证。GitHub: https://github.com/Z3Prover/z3 |
| 4 | **AgentDojo** | Edoardo Debenedetti | 中 | Agent安全评估环境，支持间接注入攻击测试和防御评估。GitHub: https://github.com/ethz-spylab/agentdojo |
| 5 | **ToolEmu** | Yang et al. | 中 | Agent工具调用风险模拟器，用于安全测试。GitHub: https://github.com/ryoungj/toolEmu |
| 6 | **MCP (Model Context Protocol)** | Anthropic | 中 | Anthropic开源的模型上下文协议，理解Agent工具调用安全的标准协议。GitHub: https://github.com/modelcontextprotocol |
| 7 | **LangGraph** | LangChain | 中 | LangChain的Agent状态图框架，支持构建可验证的Agent工作流。GitHub: https://github.com/langchain-ai/langgraph |

---

### 4.5 AI护栏与对齐理论

#### 4.5.1 书籍

| # | 名称 | 作者/来源 | 类型 | 难度 | 推荐理由 |
|---|------|----------|------|------|----------|
| 1 | **《Reinforcement Learning: An Introduction》(2nd Edition)** | Richard S. Sutton, Andrew G. Barto | 书籍 | 中-高 | 强化学习圣经（"RL圣经"），系统讲解RL理论基础，RLHF/DPO的理论根基。第13章Policy Gradient、第16章重要采样是RLHF核心 |
| 2 | **《Alignment of Language Models》** | 多位作者 | 在线书籍 | 中-高 | 系统讲解LLM对齐方法，涵盖SFT→RLHF→DPO全流程 |
| 3 | **《Human Feedback for Machine Learning》** | 多位作者 | 综述 | 中 | 人类反馈在机器学习中的系统应用，RLHF的理论基础 |
| 4 | **《NeMo Guardrails Documentation》** | NVIDIA | 在线文档 | 入门-中 | NeMo Guardrails官方文档，学习安全护栏框架的第一手资料。文档网址：https://docs.nvidia.com/nemo/guardrails/ |
| 5 | **《Superalignment: Overview and Open Problems》** | OpenAI Superalignment Team | 在线文档 | 高 | OpenAI超级对齐团队的公开研究文档，涵盖弱到强监督、可扩展监督等前沿理论 |

#### 4.5.2 大学课程

| # | 名称 | 来源 | 难度 | 推荐理由 |
|---|------|------|------|----------|
| 1 | **Deep RL Bootcamp** | UC Berkeley | 中-高 | Berkeley深度强化学习训练营，含RLHF相关内容，理解PPO等RLHF核心算法。课程网址：https://deeprlbootcamp.github.io/ |
| 2 | **CS 285: Deep Reinforcement Learning** | UC Berkeley | 高 | Sergey Levine的深度RL课程，涵盖策略梯度、PPO等RLHF核心算法。课程网址：https://rail.eecs.berkeley.edu/deeprlcourse/ |
| 3 | **Alignment, Safety, and Governance of AI** | Stanford HAI | 高 | Stanford HAI的对齐与安全课程，涵盖RLHF、DPO、Superalignment等前沿主题 |
| 4 | **CS 324: Large Language Models (Alignment Week)** | Stanford University | 中-高 | CS324的对齐模块，专门讲解RLHF、DPO等对齐方法 |

#### 4.5.3 博客与论文

| # | 名称 | 作者/来源 | 难度 | 推荐理由 |
|---|------|------|------|----------|
| 1 | **论文："Training language models to follow instructions with human feedback (InstructGPT)"** | Long Ouyang et al. (OpenAI) | 高 | RLHF的奠基论文，系统讲解SFT→奖励模型→PPO三步训练流程。论文：https://arxiv.org/abs/2203.02155 |
| 2 | **论文："Direct Preference Optimization: Your Language Model is Secretly a Reward Model"** | Rafael Rafailov et al. | 高 | DPO的开创性论文，提出无需奖励模型的对齐方法，改变了RLHF范式。论文：https://arxiv.org/abs/2305.18290 |
| 3 | **论文："Constitutional AI: Harmlessness from AI Feedback"** | Anthropic | 高 | Anthropic的Constitutional AI论文，提出基于AI反馈的对齐方法（RLAIF），是RLHF的重要替代方案。论文：https://arxiv.org/abs/2212.08073 |
| 4 | **论文："Weak-to-Strong Generalization"** | OpenAI Superalignment Team | 高 | OpenAI超级对齐的核心论文，提出弱模型监督强模型的范式。论文：https://arxiv.org/abs/2312.09390 |
| 5 | **论文："AI Safety via Debate"** | Irving et al. (DeepMind) | 高 | 辩论方法的奠基论文，提出通过AI辩论实现可扩展监督。论文：https://arxiv.org/abs/1805.00899 |
| 6 | **论文："Scaling Monosemanticity"** | Anthropic | 高 | Anthropic的可解释性研究，理解模型内部表示对安全分析至关重要。论文：https://transformer-circuits.pub/2024/scaling-monosemanticity/ |
| 7 | **博客："The Alignment Problem from a Deep Learning Perspective"** | Richard Ngo | 中-高 | 从深度学习视角分析对齐问题的系统性博客 |
| 8 | **论文："Sycophancy in Language Models"** | Anthropic | 中 | 分析LLM的谄媚问题，过度对齐的风险之一。论文：https://arxiv.org/abs/2310.13548 |
| 9 | **论文："SimPO: Simple Preference Optimization with a Reference-Free Reward"** | Meng et al. | 高 | SimPO论文，DPO的改进变体，无需参考模型。论文：https://arxiv.org/abs/2405.14734 |
| 10 | **博客："Illustrating Reinforcement Learning from Human Feedback (RLHF)"** | Hugging Face | 入门-中 | RLHF的最佳可视化入门教程，直观理解RLHF流程。网址：https://huggingface.co/blog/rlhf |
| 11 | **论文："Training a Helpful and Harmless Assistant with RLHF"** | Anthropic (HHH) | 高 | Anthropic的RLHF实践论文，详细讲解如何训练有用且无害的助手。论文：https://arxiv.org/abs/2204.05862 |
| 12 | **论文："Discovering Language Model Behaviors with Model-Written Evaluations"** | Perez et al. (Anthropic) | 中-高 | 使用LLM自动生成评估的方法，对自动化安全审计有重要参考价值。论文：https://arxiv.org/abs/2212.09251 |

#### 4.5.4 开源项目/代码库

| # | 名称 | 来源 | 难度 | 推荐理由 |
|---|------|------|------|----------|
| 1 | **NeMo Guardrails** | NVIDIA | 中 | NVIDIA开源的LLM护栏框架，支持输入/输出/对话/检索护栏，是学习护栏系统的核心项目。GitHub: https://github.com/NVIDIA/NeMo-Guardrails |
| 2 | **TRL (Transformer Reinforcement Learning)** | Hugging Face | 中 | HuggingFace的RLHF/DPO训练库，支持SFT、PPO、DPO等对齐训练。GitHub: https://github.com/huggingface/trl |
| 3 | **LLMGuard** | Protect AI | 入门-中 | LLM输入/输出安全检测库，支持PII检测、注入检测、有害内容检测。GitHub: https://github.com/protectai/llm-guard |
| 4 | **Guardrails AI** | Guardrails AI | 中 | 另一个流行的LLM护栏框架，支持结构化输出验证和安全检查。GitHub: https://github.com/guardrails-ai/guardrails |
| 5 | **ConstitutionalAI-Harmlessness** | Anthropic (社区复现) | 高 | Constitutional AI方法的社区复现实现，学习RLAIF的实践代码 |
| 6 | **OpenRLHF** | OpenLLMAI | 中-高 | 开源的RLHF训练框架，支持PPO、DPO、Reward Model训练全流程。GitHub: https://github.com/OpenRLHF/OpenRLHF |
| 7 | **Llama Guard** | Meta | 中 | Meta开源的LLM安全分类器，基于Llama微调，用于输入/输出安全检测。GitHub: https://github.com/meta-llama/llama-guard |

---

### 4.6 Agent红队攻防实战

#### 4.6.1 书籍

| # | 名称 | 作者/来源 | 类型 | 难度 | 推荐理由 |
|---|------|----------|------|------|----------|
| 1 | **《The Hacker's Playbook 3》** | Peter Kim | 书籍 | 中 | 渗透测试实战手册，红队测试方法论可迁移到AI Agent红队 |
| 2 | **《Red Team Development and Operations》** | Joe Vest, James Tubberville | 书籍 | 中-高 | 红队开发与运营指南，系统讲解红队工具开发、自动化测试方法论 |
| 3 | **《Hacking AI: A Guide to Attacking and Defending LLMs** | 社区编撰 | 电子书 | 中 | 专门针对AI/LLM的攻防实战指南 |

#### 4.6.2 博客与论文

| # | 名称 | 作者/来源 | 难度 | 推荐理由 |
|---|------|------|------|----------|
| 1 | **论文："Automated Red Teaming with ART: Assessing Robustness of LLMs"** | Mathem et al. | 中 | 自动化红队测试论文，提出使用LLM自动生成攻击的方法。论文：https://arxiv.org/abs/2311.08392 |
| 2 | **论文："Red Teaming Language Models to Reduce Harms: Methods, Scaling Behaviors, and Lessons Learned"** | Anthropic | 中 | Anthropic的红队测试实践论文，详细讲解红队方法论和经验教训。论文：https://arxiv.org/abs/2209.07858 |
| 3 | **论文："Large Language Models Can Be Used to Effectively Scale Up Adversarial Examples"** | Sadasivan et al. | 中-高 | 使用LLM自动生成对抗样本的方法论 |
| 4 | **博客："How to Red Team an LLM Application"** | Microsoft | 中 | 微软的LLM红队测试指南，系统讲解红队测试流程。网址：https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/red-teaming |
| 5 | **博客："AI Red Teaming Guide"** | NIST AI Safety Institute | 中 | NIST的AI红队测试指南，官方权威方法论 |
| 6 | **论文："Multi-turn Attack and Defense for LLM Agents"** | 多位作者 | 中 | 多轮对话攻击论文，系统化分析Agent在多轮交互中的脆弱性 |
| 7 | **博客："Jailbreaking LLMs: A Comprehensive Taxonomy"** | Lakera | 中 | Lakera的越狱技术分类法，系统化梳理所有已知越狱技术。网址：https://lakera.ai/ |

#### 4.6.3 开源项目/代码库

| # | 名称 | 来源 | 难度 | 推荐理由 |
|---|------|------|------|----------|
| 1 | **PyRIT** | Microsoft | 中 | 微软LLM红队测试框架（已在4.2.4介绍），支持多轮攻击、自动化payload生成。GitHub: https://github.com/Azure/PyRIT |
| 2 | **Garak** | NVIDIA | 中 | LLM漏洞扫描器（已在4.2.4介绍），支持自动化扫描多种漏洞类型。GitHub: https://github.com/leondz/garak |
| 3 | **RedTeam-AI** | 社区 | 中 | AI红队测试工具集，含攻击模板库、自动化测试框架 |
| 4 | **JailbreakDB** | 社区维护 | 入门-中 | 越狱攻击模板数据库，收集了大量公开越狱模板，适合复现学习。GitHub: https://github.com/jailbreakdb |
| 5 | **PromptInject** | Liu et al. | 中 | 提示词注入攻击框架（已在4.2.4介绍），系统化实现多种注入攻击。GitHub: https://github.com/agencyenterprise/PromptInject |
| 6 | **PAIR (Prompt Automatic Iterative Refinement)** | Chao et al. | 中-高 | 使用LLM自动生成越狱攻击的方法实现，自动化红队的前沿方法。论文：https://arxiv.org/abs/2310.08419 |
| 7 | **AdvBench** | Andy Zou et al. | 中 | 对抗行为基准数据集，用于评估和开发攻击/防御方法 |
| 8 | **HackAPrompt** | Schulhoff et al. | 入门-中 | 全球规模的提示词黑客竞赛平台，含大量真实攻击数据。网址：https://hackaprompt.com/ |

---

### 4.7 防御组件研发与落地

#### 4.7.1 书籍

| # | 名称 | 作者/来源 | 类型 | 难度 | 推荐理由 |
|---|------|----------|------|------|----------|
| 1 | **《LangChain for LLM Application Development》** | LangChain团队 | 在线文档/书籍 | 入门-中 | LangChain官方文档和教程，开发生产级护栏系统的核心框架 |
| 2 | **《Fine-Tuning Large Language Models** | Maxime Labonne | 书籍 | 中 | 系统讲解LLM微调方法（SFT/LoRA/DPO），防御组件微调的实践指南 |
| 3 | **《Designing Machine Learning Systems》** | Chip Huyen | 书籍 | 中 | ML系统设计经典，理解生产级AI安全系统的架构设计 |
| 4 | **《Neuro-Symbolic AI: An Emerging Class of AI Workloads and their Characterization** | 多位作者 | 综述 | 高 | 神经符号AI综述，理解神经符号监控器的理论基础 |

#### 4.7.2 博客与论文

| # | 名称 | 作者/来源 | 难度 | 推荐理由 |
|---|------|------|------|----------|
| 1 | **论文："Defending Against Alignment-Breaking Attacks via Robustly Aligned LLM"** | Cao et al. | 中 | 通过微调增强LLM对齐鲁棒性的方法。论文：https://arxiv.org/abs/2309.14348 |
| 2 | **论文："SafeDecoding: Defending against Jailbreak Attacks via Safety-Aware Decoding"** | Xu et al. | 中 | 安全感知解码方法，在解码阶段防御越狱攻击。论文：https://arxiv.org/abs/2402.08954 |
| 3 | **论文："Baseline Defenses for Adversarial Attacks Against Aligned Language Models"** | Robey et al. | 中 | 针对对抗攻击的基础防御方法，含平滑、检测等。论文：https://arxiv.org/abs/2308.01827 |
| 4 | **博客："Building Production-Grade Guardrails for LLMs"** | NVIDIA | 中 | NVIDIA的生产级护栏开发实践指南 |
| 5 | **论文："Neuro-Symbolic Guardrails for Autonomous Agents"** | 多位作者 | 高 | 神经符号护栏的前沿研究，结合符号推理和神经网络的Agent安全监控 |
| 6 | **博客："LoRA: Low-Rank Adaptation Explained"** | Sebastian Raschka | 中 | LoRA的详细技术解释，防御组件微调的核心技术。网址：https://magazine.sebastianraschka.com/p/lora-and-dora-from-scratch |
| 7 | **论文："SmoothLLM: Defending Large Language Models Against Jailbreaking Attacks"** | Robey et al. | 中 | 通过输入扰动防御越狱攻击的方法。论文：https://arxiv.org/abs/2310.03684 |
| 8 | **论文："Llama Guard: LLM-based Input-Output Safeguard for Human-AI Conversations"** | Meta | 中 | Meta的Llama Guard论文，基于LLM的安全分类器设计。论文：https://arxiv.org/abs/2312.06674 |

#### 4.7.3 开源项目/代码库

| # | 名称 | 来源 | 难度 | 推荐理由 |
|---|------|------|------|----------|
| 1 | **LangChain** | LangChain | 入门-中 | LLM应用开发框架，含安全组件、Guardrail集成。GitHub: https://github.com/langchain-ai/langchain |
| 2 | **NeMo Guardrails** | NVIDIA | 中 | 生产级护栏框架（已在4.5.4介绍），支持多种护栏类型。GitHub: https://github.com/NVIDIA/NeMo-Guardrails |
| 3 | **LLMGuard** | Protect AI | 入门-中 | LLM安全检测库（已在4.5.4介绍），支持PII检测、注入检测。GitHub: https://github.com/protectai/llm-guard |
| 4 | **PEFT (Parameter-Efficient Fine-Tuning)** | Hugging Face | 中 | HuggingFace的参数高效微调库，支持LoRA/AdaLoRA等，防御组件微调的核心工具。GitHub: https://github.com/huggingface/peft |
| 5 | **Llama Guard** | Meta | 中 | Meta的安全分类器（已在4.5.4介绍），可直接用于输入/输出安全检测。GitHub: https://github.com/meta-llama/llama-guard |
| 6 | **Guardrails AI** | Guardrails AI | 中 | 结构化输出验证和安全检查框架（已在4.5.4介绍）。GitHub: https://github.com/guardrails-ai/guardrails |
| 7 | **LLM Guard API** | Laiyer AI | 入门-中 | LLMGuard的API服务版本，可直接部署为微服务。GitHub: https://github.com/laiyer-ai/llm-guard-api |
| 8 | **Rebuff** | Protect AI | 中 | 提示词注入检测框架，结合启发式规则和ML模型。GitHub: https://github.com/protectai/rebuff |

---

### 4.8 业务风险建模能力

#### 4.8.1 书籍

| # | 名称 | 作者/来源 | 类型 | 难度 | 推荐理由 |
|---|------|----------|------|------|----------|
| 1 | **《Threat Modeling: Designing for Security》** | Adam Shostack | 书籍 | 中 | 威胁建模权威指南，系统讲解STRIDE等威胁建模方法，作者是微软威胁建模团队负责人 |
| 2 | **《Threat Modeling: A Practical Guide for Security Teams》** | Izar Tarandach, Mike Coates | 书籍 | 中 | 威胁建模实践指南，含敏捷威胁建模方法 |
| 3 | **《Game Theory: An Introduction》** | Steven Tadelis | 书籍 | 中 | 博弈论入门教材，理解多Agent博弈分析的理论基础 |
| 4 | **《The Art of Strategy: A Game Theorist's Guide to Success in Business and Life》** | Avinash Dixit, Barry Nalebuff | 书籍 | 入门-中 | 博弈论的通俗入门读物，建立博弈分析直觉 |
| 5 | **《Security Engineering: A Guide to Building Dependable Distributed Systems》** | Ross Anderson | 书籍 | 高 | 安全工程圣经（第三版免费在线），系统讲解安全工程方法论。在线网址：https://www.cl.cam.ac.uk/~rja14/book.html |

#### 4.8.2 大学课程

| # | 名称 | 来源 | 难度 | 推荐理由 |
|---|------|------|------|----------|
| 1 | **CS 155: Computer and Network Security** | Stanford University | 中 | Stanford计算机与网络安全课程，涵盖威胁建模、安全架构设计 |
| 2 | **Game Theory (Coursera, Stanford)** | Matthew Jackson, Kevin Leyton-Brown, Yoav Shoham | 入门-中 | Stanford博弈论在线课程，理解多Agent博弈分析的基础。课程网址：https://www.coursera.org/learn/game-theory-1 |
| 3 | **ECON 159: Game Theory** | Yale University | 入门-中 | Yale博弈论公开课，经典博弈论入门。课程网址：https://oyc.yale.edu/economics/econ-159 |
| 4 | **MIT 6.858: Computer Systems Security** | MIT | 中-高 | MIT计算机系统安全课程，涵盖安全设计原则、威胁建模 |

#### 4.8.3 博客与论文

| # | 名称 | 作者/来源 | 难度 | 推荐理由 |
|---|------|------|------|----------|
| 1 | **论文："Threat Modeling LLM-Integrated Applications"** | 多位作者 | 中 | 专门针对LLM应用的威胁建模论文，将STRIDE适配到AI场景 |
| 2 | **论文："Multi-Agent Collusion: Risks and Mitigations in LLM Systems"** | 多位作者 | 高 | 多Agent共谋风险分析论文，系统讲解共谋机理与防御 |
| 3 | **博客："AI Threat Modeling with STRIDE"** | OWASP | 中 | OWASP的AI威胁建模指南，将STRIDE方法适配到AI系统。网址：https://owasp.org/www-project-ai-security-and-privacy-guide/ |
| 4 | **论文："Cooperative AI: Machine Learning and Game Theory"** | Dafoe et al. | 高 | 合作AI的博弈论分析，理解多Agent协作与竞争 |
| 5 | **博客："NIST AI Risk Management Framework (AI RMF)"** | NIST | 中 | NIST的AI风险管理框架，AI业务风险建模的权威指南。网址：https://www.nist.gov/itl/ai-risk-management-framework |
| 6 | **论文："Evaluating and Mitigating Discrimination in Language Model Representations"** | Liang et al. | 中 | 评估和缓解LLM中的歧视问题，业务风险的重要维度 |
| 7 | **论文："AI Alignment and Safety: A Game-Theoretic Approach"** | 多位作者 | 高 | 从博弈论视角分析AI对齐与安全 |
| 8 | **博客:"EU AI Act Risk Classification Guide"** | EU AI Act | 中 | 欧盟AI法案的风险分类指南，了解合规视角的AI风险评估 |

#### 4.8.4 开源项目/代码库

| # | 名称 | 来源 | 难度 | 推荐理由 |
|---|------|------|------|----------|
| 1 | **OWASP AI Security and Privacy Guide** | OWASP | 入门-中 | OWASP的AI安全与隐私指南，含威胁建模模板和最佳实践。GitHub: https://github.com/OWASP/www-project-ai-security-and-privacy-guide |
| 2 | **Threat Dragon** | OWASP | 入门-中 | OWASP开源威胁建模工具，支持STRIDE等方法的可视化建模。GitHub: https://github.com/OWASP/threat-dragon |
| 3 | **NIST AI RMF Playbook** | NIST | 中 | NIST AI风险管理框架的实施手册，含风险评估模板。网址：https://airc.nist.gov/ |
| 4 | **AI Safety Benchmarks (MLCommons)** | MLCommons | 中 | MLCommons的AI安全基准测试，含风险评估方法论。网址：https://mlcommons.org/ |
| 5 | **Multi-Agent Security Framework** | 社区 | 高 | 多Agent安全分析框架，含博弈论模型实现 |

---

### 4.9 隐私追踪与策略强制执行

#### 4.9.1 书籍

| # | 名称 | 作者/来源 | 类型 | 难度 | 推荐理由 |
|---|------|----------|------|------|----------|
| 1 | **《A Pragmatic Introduction to Secure Multi-Party Computation》** | David Evans et al. | 在线书籍 | 高 | 安全多方计算（MPC）的实践入门教材，系统讲解秘密共享、混淆电路、OT等核心协议。在线免费：https://securecomputation.org/ |
| 2 | **《Fully Homomorphic Encryption: A Survey** | 多位作者 | 综述 | 高 | 全同态加密的综合调研，涵盖FHE方案、应用场景、性能优化 |
| 3 | **《Hardware-Assisted Security: TEE and Beyond》** | 多位作者 | 综述 | 高 | 硬件辅助安全综述，涵盖TEE、SGX、TrustZone等可信执行环境 |
| 4 | **《Taint Analysis for Software Security》** | 多位作者 | 书籍 | 中-高 | 污点分析在软件安全中的应用，含静态/动态污点分析方法 |
| 5 | **《Data Privacy: Principles and Practice》** | Katharina Morik et al. | 书籍 | 中 | 数据隐私原则与实践，涵盖匿名化、差分隐私、联邦学习 |

#### 4.9.2 大学课程

| # | 名称 | 来源 | 难度 | 推荐理由 |
|---|------|------|------|----------|
| 1 | **CS 255: Cryptography** | Stanford University | 高 | Stanford密码学课程，涵盖零知识证明、安全多方计算、同态加密等隐私技术基础 |
| 2 | **6.857: Network and Computer Security** | MIT | 中-高 | MIT网络安全课程，含硬件安全、TEE、信息流控制等内容 |
| 3 | **Privacy Enhancing Technologies (CMU)** | Carnegie Mellon University | 高 | CMU的隐私增强技术课程，涵盖匿名化、DP、TEE、MPC等 |
| 4 | **Secure Computation (BU CS 789)** | Boston University | 高 | 安全计算专门课程，深入讲解MPC、FHE、ZKP等密码学协议 |

#### 4.9.3 博客与论文

| # | 名称 | 作者/来源 | 难度 | 推荐理由 |
|---|------|------|------|----------|
| 1 | **论文："TaintRay: Understanding and Detecting Information Leakage in LLMs"** | 多位作者 | 高 | 专门针对LLM的污点追踪论文，将传统污点分析适配到大模型场景 |
| 2 | **论文:"A Survey of Privacy-Preserving Techniques for Large Language Models"** | 多位作者 | 中-高 | LLM隐私保护技术综述，涵盖DP、TEE、MPC在LLM中的应用 |
| 3 | **论文："Confidential Computing for AI: TEE-based Privacy Protection"** | 多位作者 | 高 | TEE在AI隐私保护中的应用论文 |
| 4 | **论文:"CrypTen: Secure Multi-Party Computation Meets Machine Learning"** | Facebook AI Research | 高 | Facebook的MPC-ML框架论文，将MPC应用于机器学习推理。论文：https://arxiv.org/abs/2109.00984 |
| 5 | **论文："BOLT: Privacy-Preserving, Accurate and Efficient Inference for Large Language Models"** | 多位作者 | 高 | 基于MPC的LLM隐私推理论文。论文：https://arxiv.org/abs/2405.01600 |
| 6 | **博客："Data Provenance and Lineage for AI Systems"** | 多位作者 | 中 | AI系统的数据溯源与血缘分析 |
| 7 | **论文："Information Flow Control for LLM Applications"** | 多位作者 | 高 | 信息流控制应用于LLM的前沿研究 |
| 8 | **博客:"Confidential AI with Intel SGX"** | Intel | 中 | Intel的SGX在AI隐私保护中的实践指南 |
| 9 | **论文:"On the Privacy Risks of Retrieval-Augmented Language Models"** | 多位作者 | 中 | RAG系统的隐私风险分析，含向量数据库的信息泄露 |

#### 4.9.4 开源项目/代码库

| # | 名称 | 来源 | 难度 | 推荐理由 |
|---|------|------|------|----------|
| 1 | **CrypTen** | Meta (Facebook AI Research) | 高 | Meta开源的MPC框架，支持安全多方计算在机器学习中的应用，学习MPC实践的最佳项目。GitHub: https://github.com/facebookresearch/CrypTen |
| 2 | **OpenFHE** | OpenFHE Consortium | 高 | 开源全同态加密库，支持多种FHE方案（BGV/CKKS/TFHE），学习同态加密实践的核心工具。GitHub: https://github.com/openfheorg/openfhe-development |
| 3 | **OpenSGX / SGX SDK** | Intel / 社区 | 高 | Intel SGX SDK，学习TEE编程实践的官方工具。GitHub: https://github.com/intel/linux-sgx |
| 4 | **Gramine** | Gramine Project | 高 | 基于Intel SGX的LibOS，可在TEE中运行LLM推理，学习TEE应用的实践项目。GitHub: https://github.com/gramineproject/gramine |
| 5 | **Opaque** | UC Berkeley RISELab | 高 | 基于SGX的安全数据分析框架，学习TEE在数据处理中的应用。GitHub: https://github.com/microsoft/Opaque |
| 6 | **TF-Encrypted (TFE)** | OpenMined | 高 | 基于TensorFlow的安全多方计算框架，支持隐私保护的ML推理。GitHub: https://github.com/tf-encrypted/tf-encrypted |
| 7 | **Presidio** | Microsoft | 入门-中 | 微软开源的数据保护和匿名化工具，支持PII检测、匿名化、脱敏，是隐私保护实践的入门项目。GitHub: https://github.com/microsoft/presidio |
| 8 | **FATE (Federated AI Technology Enabler)** | Webank AI | 中-高 | 联邦学习框架，含差分隐私、安全聚合等隐私保护机制。GitHub: https://github.com/FederatedAI/FATE |
| 9 | **Tumult Analytics** | Tumult Labs | 中 | 差分隐私数据分析框架，学习DP在数据分析中的实践。GitHub: https://github.com/tumult-labs/tumult-analytics |

---

## 5. 学习路径建议

### 5.1 阶段划分与时间规划

```
阶段0 (前置基础)     阶段1 (安全基础)      阶段2 (攻防实战)      阶段3 (高级进阶)
    │                   │                   │                   │
    ▼                   ▼                   ▼                   ▼
┌──────────┐     ┌──────────┐       ┌──────────┐       ┌──────────┐
│  LLM基础  │ ──→ │ LLM安全  │ ────→ │ 红队攻防  │ ────→ │ 前沿研究  │
│  RL基础   │     │ 漏洞机理  │       │ 防御组件  │       │ 架构设计  │
│  密码学   │     │ 隐私基础  │       │ 风险建模  │       │ 创新研究  │
└──────────┘     └──────────┘       └──────────┘       └──────────┘
  约4-6周           约6-8周            约8-10周           约6-8周
```

### 5.2 详细学习路线

#### 阶段0：前置基础（4-6周）

**目标**：建立LLM/RL/安全的基础认知，为后续AI安全学习打好根基。

| 周次 | 学习内容 | 核心资料 | 实践任务 |
|------|----------|----------|----------|
| 第1-2周 | Transformer与LLM架构 | 1. Illustrated Transformer (Jay Alammar)<br>2. Stanford CS224N (选看核心讲座)<br>3. nanoGPT项目 | 用nanoGPT理解GPT架构，阅读代码 |
| 第3周 | LLM训练全流程 | 1. 《Build a Large Language Model (From Scratch)》<br>2. HuggingFace NLP Course | 使用HuggingFace Transformers跑通一个模型推理 |
| 第4周 | 强化学习基础 | 1. 《RL: An Introduction》第13章<br>2. HuggingFace RLHF博客 | 理解Policy Gradient和PPO的基本原理 |
| 第5周 | 密码学与隐私基础 | 1. Stanford CS255 讲义（选读）<br>2. 差分隐私概念入门 | 理解DP、MPC、FHE的基本概念 |
| 第6周 | Web安全基础 | 1. 《Web Application Hacker's Handbook》选读<br>2. OWASP Top 10 (传统版) | 理解SSRF、命令注入、API越权等传统威胁 |

#### 阶段1：LLM安全基础与漏洞机理（6-8周）

**目标**：系统掌握LLM安全风险全貌，理解OWASP Top 10 for LLM，深入漏洞机理。

| 周次 | 学习内容 | 核心资料 | 实践任务 |
|------|----------|----------|----------|
| 第1-2周 | OWASP Top 10 for LLM | 1. OWASP Top 10 for LLM 官方文档<br>2. Lilian Weng's "Prompt Injection"博客 | 制作OWASP Top 10 LLM风险思维导图 |
| 第3周 | 提示词注入攻击 | 1. Greshake et al. 间接注入论文<br>2. PromptInject项目 | 用PromptInject复现3种注入攻击 |
| 第4周 | 越狱攻击 | 1. "Jailbroken"论文<br>2. JailbreakDB模板<br>3. HackAPrompt | 复现5种经典越狱技术 |
| 第5周 | 对抗性攻击 | 1. GCG论文 (Zou et al.)<br>2. llm-attacks代码库 | 运行GCG攻击实验 |
| 第6周 | 多模态注入与注意力机制 | 1. "Image Hijacks"论文<br>2. "Lost in the Middle"论文 | 分析注意力权重与攻击载荷的关联 |
| 第7-8周 | Agent漏洞与工具调用安全 | 1. AgentDojo论文/项目<br>2. ToolEmu论文<br>3. MCP协议文档 | 在AgentDojo中测试Agent工具调用安全 |

#### 阶段2：攻防实战与防御组件（8-10周）

**目标**：具备红队攻防实战能力，能开发生产级防御组件。

| 周次 | 学习内容 | 核心资料 | 实心任务 |
|------|----------|----------|----------|
| 第1-2周 | 自动化红队测试 | 1. PyRIT框架<br>2. Garak扫描器<br>3. Anthropic红队论文 | 使用PyRIT对开源LLM进行自动化红队测试 |
| 第3周 | 间接注入Payload构造 | 1. AgentDojo实战<br>2. 业务场景分析 | 针对RAG应用构造间接注入payload |
| 第4-5周 | 防御组件开发 | 1. NeMo Guardrails文档/代码<br>2. LangChain安全组件<br>3. LLMGuard | 开发一个生产级护栏系统原型 |
| 第6-7周 | 微调防御 | 1. TRL库<br>2. PEFT库<br>3. Llama Guard | 使用LoRA微调一个安全分类器 |
| 第8周 | 威胁建模 | 1. 《Threat Modeling》<br>2. STRIDE for AI<br>3. NIST AI RMF | 对一个AI助手应用进行STRIDE威胁建模 |
| 第9-10周 | 隐私保护实践 | 1. Presidio工具<br>2. Opacus (DP训练)<br>3. 差分隐私概念 | 实现静态/动态数据脱敏Pipeline |

#### 阶段3：高级进阶与前沿研究（6-8周）

**目标**：达到评估3-专家级水平，掌握形式化方法、超级对齐、高级隐私架构。

| 周次 | 学习内容 | 核心资料 | 实践任务 |
|------|----------|----------|----------|
| 第1-2周 | 形式化方法与Agent安全 | 1. 《Principles of Model Checking》选读<br>2. LTL基础<br>3. SPIN/NuSMV工具 | 用LTL描述Agent安全属性并用模型检测器验证 |
| 第3周 | 对齐理论深度 | 1. InstructGPT论文<br>2. DPO论文<br>3. Constitutional AI论文 | 分析对齐税，从损失函数角度分析RLHF vs DPO |
| 第4周 | 超级对齐前沿 | 1. Weak-to-Strong Generalization论文<br>2. AI Safety via Debate论文<br>3. OpenAI Superalignment文档 | 设计一个多Agent对抗博弈的安全审计方案 |
| 第5周 | 高级隐私技术 | 1. CrypTen框架<br>2. OpenFHE库<br>3. Gramine (TEE) | 在TEE中运行LLM推理的原型实验 |
| 第6周 | 多Agent博弈与共谋 | 1. 博弈论课程 (Coursera Stanford)<br>2. 多Agent共谋论文 | 分析多Agent协作场景的共谋风险 |
| 第7-8周 | 综合项目 | 全部知识综合 | 设计一个端到端的AI应用安全方案（含攻击面分析、防御架构、隐私保护、审计策略） |

### 5.3 学习优先级矩阵

按照**岗位考察权重**和**学习难度**排列优先级：

```
                    高优先级
                        │
         ┌──────────────┼──────────────┐
         │              │              │
    优先学          核心必学         尽量学
    (基础+高频)      (岗位核心)      (前沿加分)
         │              │              │
   ┌─────┴─────┐  ┌────┴────┐   ┌────┴────┐
   │ OWASP Top10│  │ 间接注入 │   │ 超级对齐 │
   │ 提示词注入  │  │ 自动化   │   │ 形式化   │
   │ 越狱复现   │  │ 红队    │   │ 验证    │
   │ RLHF基础  │  │ 护栏开发 │   │ 同态加密 │
   │ STRIDE    │  │ 微调防御 │   │ MPC    │
   │ 数据脱敏   │  │ 污点追踪 │   │ TEE    │
   │           │  │ 博弈分析  │   │ 零信任  │
   └───────────┘  └─────────┘   └────────┘
         │              │              │
         ▼              ▼              ▼
      入门-中          中-高          高
```

### 5.4 关键里程碑检查清单

#### 里程碑1：LLM安全基础（阶段0+1完成后）
- [ ] 能完整列举OWASP Top 10 for LLM并解释每种风险
- [ ] 能复现至少5种越狱技术并解释原理
- [ ] 理解直接注入与间接注入的区别
- [ ] 能解释注意力机制与攻击载荷的关联
- [ ] 理解GCG等对抗攻击的基本原理

#### 里程碑2：攻防实战能力（阶段2完成后）
- [ ] 能使用PyRIT/Garak进行自动化红队测试
- [ ] 能针对业务场景构造间接注入payload
- [ ] 能使用NeMo Guardrails/LangChain开发护栏系统
- [ ] 能使用LoRA微调安全分类器
- [ ] 能对AI应用进行STRIDE威胁建模
- [ ] 能实现数据脱敏Pipeline

#### 里程碑3：高级进阶（阶段3完成后）
- [ ] 能用LTL描述Agent安全属性
- [ ] 能从损失函数角度分析RLHF/DPO的对齐税
- [ ] 理解Superalignment和可扩展监督
- [ ] 能使用CrypTen/OpenFHE进行隐私计算
- [ ] 能分析多Agent共谋风险
- [ ] 能设计端到端AI应用安全方案

### 5.5 推荐学习节奏

| 日常 | 每周投入 | 建议安排 |
|------|----------|----------|
| 理论学习 | 8-10小时 | 阅读论文/书籍/博客，做笔记 |
| 代码实践 | 8-10小时 | 运行开源项目，编写攻击/防御代码 |
| 论文精读 | 3-4小时 | 每周精读1-2篇核心论文 |
| 综合项目 | 4-5小时 | 每阶段完成一个综合实践项目 |
| **总计** | **约23-29小时/周** | 全程约24-32周（6-8个月） |

### 5.6 核心论文阅读清单（按优先级排序）

| 优先级 | 论文 | 对应知识点 | 难度 |
|--------|------|-----------|------|
| ★★★★★ | OWASP Top 10 for LLM | A2 | 入门 |
| ★★★★★ | InstructGPT (Ouyang et al., 2022) | D2 | 高 |
| ★★★★★ | DPO (Rafailov et al., 2023) | D4 | 高 |
| ★★★★★ | Indirect Prompt Injection (Greshake et al., 2023) | A3 | 中-高 |
| ★★★★☆ | GCG Attack (Zou et al., 2023) | A6 | 高 |
| ★★★★☆ | Jailbroken (Wei et al., 2023) | A3 | 高 |
| ★★★★☆ | Constitutional AI (Anthropic, 2022) | D4 | 高 |
| ★★★★☆ | Extracting Training Data (Carlini et al., 2021) | B5 | 高 |
| ★★★★☆ | Deep Learning with DP (Abadi et al., 2016) | B6 | 高 |
| ★★★☆☆ | Weak-to-Strong Generalization (OpenAI, 2023) | D6 | 高 |
| ★★★☆☆ | AI Safety via Debate (Irving et al., 2018) | D6 | 高 |
| ★★★☆☆ | AgentDojo (Debenedetti et al., 2024) | C2 | 中-高 |
| ★★★☆☆ | ToolEmu (Yang et al., 2023) | C2 | 中 |
| ★★★☆☆ | Membership Inference (Shokri et al., 2017) | B5 | 中-高 |
| ★★★☆☆ | Llama Guard (Meta, 2023) | F1 | 中 |
| ★★☆☆☆ | Image Hijacks (Bailey et al., 2023) | A3 | 高 |
| ★★☆☆☆ | SmoothLLM (Robey et al., 2023) | F2 | 中 |
| ★★☆☆☆ | SimPO (Meng et al., 2024) | D4 | 高 |

### 5.7 核心开源项目实践清单

| 优先级 | 项目 | 对应技能 | 建议实践方式 |
|--------|------|----------|-------------|
| ★★★★★ | **PyRIT** | E3 (自动化红队) | 对GPT-4/Llama进行自动化红队测试 |
| ★★★★★ | **NeMo Guardrails** | F2 (护栏开发) | 为一个RAG应用开发完整护栏系统 |
| ★★★★★ | **TRL (HuggingFace)** | F3 (微调防御) | 使用DPO训练一个安全对齐的模型 |
| ★★★★☆ | **Garak** | E3 (漏洞扫描) | 扫描多个开源LLM的安全漏洞 |
| ★★★★☆ | **AgentDojo** | E2 (间接注入) | 在AgentDojo中测试和构造间接注入攻击 |
| ★★★★☆ | **PEFT (LoRA)** | F3 (微调防御) | 使用LoRA微调安全分类器 |
| ★★★★☆ | **Presidio** | H1 (数据脱敏) | 构建PII检测和脱敏Pipeline |
| ★★★☆☆ | **Opacus** | B6 (差分隐私) | 使用DP-SGD训练模型 |
| ★★★☆☆ | **CrypTen** | H4 (MPC) | 使用MPC进行隐私保护推理 |
| ★★★☆☆ | **llm-attacks (GCG)** | E4 (新型攻击) | 运行GCG攻击实验 |
| ★★☆☆☆ | **OpenFHE** | H4 (同态加密) | 运行FHE基础示例 |
| ★★☆☆☆ | **Gramine (SGX)** | H4 (TEE) | 在SGX中运行LLM推理 |
| ★★☆☆☆ | **SPIN/NuSMV** | C3 (形式化验证) | 用LTL验证Agent安全属性 |

---

## 附录：知识-资料映射速查表

| 知识领域 | 核心书籍 | 核心课程 | 核心论文 | 核心项目 |
|----------|----------|----------|----------|----------|
| A. Agent与LLM漏洞机理 | OWASP Top 10 for LLM | CS324 (Stanford) | GCG, Indirect Injection, Jailbroken | PyRIT, Garak, llm-attacks |
| B. 动态上下文与隐私 | Algorithmic Foundations of DP | CMU Privacy Tech | DP-SGD, Membership Inference | Opacus, Privacy Meter, ChromaDB |
| C. 工具调用与逻辑安全 | Principles of Model Checking | CS254 (Stanford) | AgentDojo, ToolEmu | SPIN, NuSMV, Z3, AgentDojo |
| D. AI护栏与对齐 | RL: An Introduction | CS285 (Berkeley) | InstructGPT, DPO, Constitutional AI | NeMo Guardrails, TRL, Llama Guard |
| E. 红队攻防 | Red Team Dev and Ops | AI Safety Fundamentals | Anthropic Red Teaming, ART | PyRIT, Garak, JailbreakDB |
| F. 防御组件 | LangChain文档 | — | Llama Guard, SmoothLLM | NeMo Guardrails, PEFT, LLMGuard |
| G. 风险建模 | Threat Modeling (Shostack) | Game Theory (Coursera) | NIST AI RMF, AI Threat Modeling | Threat Dragon, NIST AI RMF |
| H. 隐私追踪 | Pragmatic Intro to MPC | CS255 (Stanford) | TaintRay, CrypTen, BOLT | CrypTen, OpenFHE, Presidio, Gramine |

---

> **文档版本**：v1.0  
> **最后更新**：2026-10-12  
> **维护说明**：本学习计划将根据AI安全领域的最新进展持续更新。建议定期关注OWASP LLM Top 10更新、arXiv安全论文、以及开源安全工具的版本迭代。
