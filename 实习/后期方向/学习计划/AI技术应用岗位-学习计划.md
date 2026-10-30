## 目录

- [[#1. 岗位概述]]
- [[#2. 知识体系全景图（知识树）]]
- [[#3. 分领域详细知识点梳理]]
  - [[#3.1 AI算法原理与模型选型]]
  - [[#3.2 大模型应用开发与微调]]
  - [[#3.3 AI模型应用设计开发能力]]
  - [[#3.4 模型微调与性能优化能力]]
- [[#4. 分领域学习资料规划]]
  - [[#4.1 AI算法原理与模型选型 — 学习资料]]
  - [[#4.2 大模型应用开发与微调 — 学习资料]]
  - [[#4.3 AI模型应用设计开发能力 — 学习资料]]
  - [[#4.4 模型微调与性能优化能力 — 学习资料]]
- [[#5. 学习路径建议（推荐学习顺序与阶段划分）]]

---

## 1. 岗位概述

### 1.1 岗位定位

"AI技术应用"岗位聚焦于**将大模型（LLM）能力落地到实际业务场景**的工程化能力，涵盖从算法原理理解、模型选型、应用开发（RAG / Prompt工程 / Agent）、微调优化到推理部署的全链路技术栈。该岗位要求候选人同时具备**扎实的理论基础**和**较强的工程实践能力**，是连接AI算法研究与产品落地的桥梁。

### 1.2 核心能力要求

| 能力维度 | 核心要求 |
|----------|----------|
| 专业知识1 | AI算法原理与模型选型（Transformer / CNN / RNN / GNN / 大模型架构 / 注意力机制变体 / 模型选型方法论） |
| 专业知识2 | 大模型应用开发与微调（SFT / LoRA / RLHF / RAG / Prompt工程 / Agent / 端到端应用设计） |
| 关键技能1 | AI模型应用设计开发能力（架构设计 / 数据处理 / 模型集成 / RAG系统开发 / 多轮对话 / 输出格式控制 / 性能调优 / Agent系统） |
| 关键技能2 | 模型微调与性能优化能力（微调全流程 / vLLM/TGI推理引擎 / 量化压缩 / 动态批处理 / 投机解码 / 显存优化 / 评估体系） |

### 1.3 面试程度目标

- **最低达标线**：进阶级（能独立负责中等复杂度AI应用系统的架构设计与开发）
- **优秀线**：精通级（能进行深度优化、设计复杂Agent系统、主导技术选型与架构决策）
- **卓越线**：专家级（能提出创新性方案、主导企业级平台建设、具备技术体系规划能力）

---

## 2. 知识体系全景图（知识树）

```
AI技术应用岗位知识体系
│
├── A. AI算法原理与模型选型
│   ├── A1. 神经网络基础
│   │   ├── A1.1 前馈神经网络（MLP / 激活函数 / 反向传播）
│   │   ├── A1.2 损失函数与优化器（SGD / Adam / 学习率调度）
│   │   ├── A1.3 正则化技术（Dropout / BatchNorm / LayerNorm）
│   │   └── A1.4 梯度问题（梯度消失/爆炸 / 残差连接）
│   ├── A2. CNN（卷积神经网络）
│   │   ├── A2.1 卷积操作与感受野
│   │   ├── A2.2 经典架构（LeNet / AlexNet / VGG / ResNet）
│   │   └── A2.3 适用场景与局限性
│   ├── A3. RNN（循环神经网络）
│   │   ├── A3.1 基本RNN与梯度问题
│   │   ├── A3.2 LSTM与GRU
│   │   ├── A3.3 序列建模应用
│   │   └── A3.4 适用场景与局限性
│   ├── A4. GNN（图神经网络）
│   │   ├── A4.1 图卷积（GCN）与消息传递
│   │   ├── A4.2 GAT / GraphSAGE
│   │   └── A4.3 适用场景与局限性
│   ├── A5. Transformer架构（核心）
│   │   ├── A5.1 自注意力机制（Q/K/V计算 / 注意力权重生成 / 缩放点积）
│   │   ├── A5.2 多头注意力（Multi-Head Attention）
│   │   ├── A5.3 位置编码（正弦编码 / 可学习编码 / RoPE / ALiBi）
│   │   ├── A5.4 前馈网络（FFN / SwiGLU）
│   │   ├── A5.5 编码器-解码器架构
│   │   ├── A5.6 Layer Normalization（Pre-Norm vs Post-Norm）
│   │   └── A5.7 残差连接
│   ├── A6. 注意力机制变体
│   │   ├── A6.1 稀疏注意力（Sparse Attention）
│   │   ├── A6.2 线性注意力（Linear Attention）
│   │   ├── A6.3 多查询注意力（MQA）
│   │   ├── A6.4 分组查询注意力（GQA）
│   │   └── A6.5 滑动窗口注意力（SWA）
│   ├── A7. 大模型架构与原理
│   │   ├── A7.1 编码器型（BERT系列）
│   │   ├── A7.2 解码器型（GPT系列 / LLaMA / GLM）
│   │   ├── A7.3 编码器-解码器型（T5 / BART）
│   │   ├── A7.4 缩放定律（Scaling Law）
│   │   ├── A7.5 涌现能力（Emergent Abilities）
│   │   ├── A7.6 指令遵循（Instruction Following）
│   │   └── A7.7 大模型训练阶段（预训练 → SFT → RLHF/DPO）
│   ├── A8. 前沿算法技术
│   │   ├── A8.1 Mixture of Experts（MoE）
│   │   ├── A8.2 长上下文建模
│   │   ├── A8.3 多模态融合（CLIP / LLaVA / Flamingo）
│   │   └── A8.4 强化学习基础（RLHF / PPO / DPO）
│   └── A9. 模型选型方法论
│       ├── A9.1 业务场景分析（推理速度 / 显存 / 精度 / 成本）
│       ├── A9.2 开源模型对比（LLaMA / GLM / Qwen / Mistral / Phi）
│       ├── A9.3 模型规模选择（7B / 13B / 70B / 405B）
│       └── A9.4 选型技术依据与Trade-off分析
│
├── B. 大模型应用开发与微调
│   ├── B1. 大模型微调
│   │   ├── B1.1 全量微调（Full Fine-tuning）
│   │   ├── B1.2 参数高效微调（PEFT）
│   │   │   ├── B1.2.1 LoRA（低秩适配）
│   │   │   ├── B1.2.2 QLoRA（量化LoRA）
│   │   │   ├── B1.2.3 Adapter Tuning
│   │   │   ├── B1.2.4 Prefix Tuning
│   │   │   └── B1.2.5 P-Tuning
│   │   ├── B1.3 SFT（监督微调）
│   │   │   ├── B1.3.1 指令数据构造
│   │   │   ├── B1.3.2 对话格式模板
│   │   │   └── B1.3.3 训练超参数调优
│   │   ├── B1.4 RLHF（基于人类反馈的强化学习）
│   │   │   ├── B1.4.1 奖励模型训练
│   │   │   ├── B1.4.2 PPO策略优化
│   │   │   └── B1.4.3 DPO（直接偏好优化）
│   │   └── B1.5 微调全流程实践
│   │       ├── B1.5.1 数据准备与清洗
│   │       ├── B1.5.2 训练配置与分布式训练
│   │       ├── B1.5.3 效果评估
│   │       └── B1.5.4 常见问题排查
│   ├── B2. RAG（检索增强生成）
│   │   ├── B2.1 RAG基础流程
│   │   │   ├── B2.1.1 文档解析（PDF/HTML/Markdown/表格）
│   │   │   ├── B2.1.2 文档切分策略（固定长度 / 语义切分 / 递归切分）
│   │   │   ├── B2.1.3 向量化（Embedding模型选择）
│   │   │   ├── B2.1.4 向量数据库（FAISS / Milvus / Chroma / Pinecone）
│   │   │   └── B2.1.5 检索与生成
│   │   ├── B2.2 高级检索技术
│   │   │   ├── B2.2.1 混合检索（稠密+稀疏 / BM25+向量）
│   │   │   ├── B2.2.2 重排序（Cross-Encoder / Cohere Rerank）
│   │   │   ├── B2.2.3 上下文优化（压缩 / 去重 / 排序）
│   │   │   └── B2.2.4 查询改写与扩展
│   │   ├── B2.3 RAG关键挑战
│   │   │   ├── B2.3.1 检索精度优化
│   │   │   ├── B2.3.2 上下文长度管理
│   │   │   ├── B2.3.3 幻觉控制
│   │   │   └── B2.3.4 实时性与增量更新
│   │   └── B2.4 高级RAG架构
│   │       ├── B2.4.1 自适应RAG（Adaptive RAG）
│   │       ├── B2.4.2 多跳检索（Multi-hop RAG）
│   │       ├── B2.4.3 GraphRAG（知识图谱增强）
│   │       └── B2.4.4 Self-RAG（自反思RAG）
│   ├── B3. Prompt工程
│   │   ├── B3.1 基础技巧
│   │   │   ├── B3.1.1 角色设定（System Prompt）
│   │   │   ├── B3.1.2 任务描述与约束
│   │   │   ├── B3.1.3 示例引导（Few-shot）
│   │   │   └── B3.1.4 输出格式控制
│   │   ├── B3.2 高级技巧
│   │   │   ├── B3.2.1 思维链（Chain-of-Thought, CoT）
│   │   │   ├── B3.2.2 少样本学习（Few-shot Learning）
│   │   │   ├── B3.2.3 结构化输出（JSON Mode / Function Calling）
│   │   │   ├── B3.2.4 自洽性（Self-Consistency）
│   │   │   └── B3.2.5 思维树（Tree of Thoughts, ToT）
│   │   └── B3.3 Agent级Prompt技术
│   │       ├── B3.3.1 ReAct（Reasoning + Acting）
│   │       ├── B3.3.2 Reflexion（自我反思）
│   │       ├── B3.3.3 Plan-and-Execute
│   │       └── B3.3.4 多Agent协作Prompt
│   ├── B4. Agent系统
│   │   ├── B4.1 Agent架构设计
│   │   │   ├── B4.1.1 感知-规划-执行-反思循环
│   │   │   ├── B4.1.2 工具调用（Function Calling / Tool Use）
│   │   │   └── B4.1.3 记忆系统（短期/长期记忆）
│   │   ├── B4.2 多Agent协作
│   │   │   ├── B4.2.1 Agent角色分工
│   │   │   ├── B4.2.2 通信协议与消息传递
│   │   │   └── B4.2.3 编排框架（LangGraph / AutoGen / CrewAI）
│   │   └── B4.3 人机协同
│   │       ├── B4.3.1 Human-in-the-loop
│   │       └── B4.3.2 人在环上反馈
│   └── B5. 大模型应用全流程
│       ├── B5.1 模型选型与评估
│       ├── B5.2 部署推理
│       ├── B5.3 效果评估与迭代
│       └── B5.4 端到端AI应用方案设计
│
├── C. AI模型应用设计开发能力
│   ├── C1. AI应用架构设计
│   │   ├── C1.1 整体架构模式（单体 vs 微服务 vs 事件驱动）
│   │   ├── C1.2 数据处理流水线设计
│   │   ├── C1.3 模型选型与集成策略
│   │   ├── C1.4 业务逻辑编排
│   │   └── C1.5 API设计（REST / gRPC / WebSocket）
│   ├── C2. 核心模块开发
│   │   ├── C2.1 数据预处理模块
│   │   ├── C2.2 结果后处理模块
│   │   ├── C2.3 模型API调用封装
│   │   ├── C2.4 Prompt管理与版本控制
│   │   └── C2.5 缓存层设计
│   ├── C3. RAG应用开发
│   │   ├── C3.1 文档解析实现
│   │   ├── C3.2 向量检索实现
│   │   ├── C3.3 提示词构建
│   │   ├── C3.4 结果生成与引用标注
│   │   └── C3.5 端到端RAG Pipeline
│   ├── C4. 复杂应用场景开发
│   │   ├── C4.1 长文本处理
│   │   ├── C4.2 多轮对话管理（对话状态追踪 / 上下文窗口管理）
│   │   ├── C4.3 输出格式控制（结构化输出 / JSON Schema）
│   │   ├── C4.4 流式输出
│   │   └── C4.5 多模态交互
│   ├── C5. 性能调优与高可用
│   │   ├── C5.1 推理延迟优化
│   │   ├── C5.2 并发处理能力提升
│   │   ├── C5.3 资源利用率优化
│   │   ├── C5.4 高可用设计（负载均衡 / 故障恢复 / 降级策略）
│   │   └── C5.5 成本优化
│   ├── C6. 复杂业务场景方案设计
│   │   ├── C6.1 多模型协作
│   │   ├── C6.2 工具调用编排
│   │   ├── C6.3 Agent系统设计
│   │   └── C6.4 端到端方案设计方法论
│   └── C7. AI平台建设
│       ├── C7.1 模型管理平台
│       ├── C7.2 Prompt管理平台
│       ├── C7.3 应用编排平台
│       └── C7.4 评估与监控平台
│
└── D. 模型微调与性能优化能力
    ├── D1. 微调全流程
    │   ├── D1.1 数据准备（清洗 / 格式化 / 增强 / 质量筛选）
    │   ├── D1.2 训练配置（学习率 / Batch Size / Epoch / Warmup）
    │   ├── D1.3 分布式训练（DDP / FSDP / DeepSpeed ZeRO）
    │   ├── D1.4 微调执行
    │   └── D1.5 效果评估（自动评估 / 人工评估 / A/B测试）
    ├── D2. 针对性微调方案设计
    │   ├── D2.1 领域适配微调
    │   ├── D2.2 指令微调
    │   ├── D2.3 多任务学习
    │   └── D2.4 数据混合策略
    ├── D3. 推理引擎与优化
    │   ├── D3.1 vLLM
    │   │   ├── D3.1.1 PagedAttention
    │   │   ├── D3.1.2 连续批处理（Continuous Batching）
    │   │   └── D3.1.3 张量并行
    │   ├── D3.2 TGI（Text Generation Inference）
    │   ├── D3.3 TensorRT-LLM
    │   ├── D3.4 llama.cpp / Ollama
    │   └── D3.5 SGLang
    ├── D4. 推理性能优化技术
    │   ├── D4.1 量化（GPTQ / AWQ / GGUF / FP8）
    │   ├── D4.2 剪枝（Pruning）
    │   ├── D4.3 知识蒸馏（Knowledge Distillation）
    │   ├── D4.4 KV Cache管理
    │   ├── D4.5 动态批处理（Dynamic Batching）
    │   ├── D4.6 前缀缓存（Prefix Caching）
    │   ├── D4.7 投机解码（Speculative Decoding）
    │   ├── D4.8 显存优化（Flash Attention / FlashInfer）
    │   └── D4.9 混合精度推理
    ├── D5. 评估体系
    │   ├── D5.1 微调效果评估指标
    │   ├── D5.2 推理性能基准测试
    │   ├── D5.3 模型质量评估（MMLU / HumanEval / MT-Bench等）
    │   └── D5.4 在线监控与持续优化
    └── D6. 企业级微调与推理服务
        ├── D6.1 微调服务框架设计
        ├── D6.2 推理服务框架设计
        ├── D6.3 极端资源约束优化
        └── D6.4 前沿技术追踪（新架构适配 / 极致压缩 / 动态推理）
```

---

## 3. 分领域详细知识点梳理

### 3.1 AI算法原理与模型选型

#### A1. 神经网络基础

- **前馈神经网络（MLP）**：线性变换+非线性激活的堆叠结构，是所有深度学习模型的基础构件。需理解感知机模型、多层感知机的数学表达、通用近似定理。
- **激活函数**：Sigmoid（梯度消失问题）、Tanh、ReLU（稀疏激活）、GELU（GPT/BERT使用）、SwiGLU（LLaMA使用）。需理解各激活函数的数学定义、梯度特性和适用场景。
- **损失函数**：交叉熵损失（分类）、均方误差（回归）、对比损失（嵌入学习）、KL散度（分布匹配）。大模型训练中的损失函数设计。
- **优化器**：SGD（动量法）、Adam（自适应学习率）、AdamW（解耦权重衰减）、Lion。理解一阶/二阶矩估计、学习率预热与衰减策略（Cosine Annealing、线性衰减）。
- **正则化技术**：Dropout（随机失活）、Batch Normalization（批归一化）、Layer Normalization（层归一化，Transformer标准配置）、RMSNorm（LLaMA使用）。理解Pre-Norm与Post-Norm的区别。
- **梯度问题**：梯度消失与梯度爆炸的成因与解决方案。残差连接（ResNet/Transformer）如何缓解梯度消失。梯度裁剪（Gradient Clipping）。

#### A2. CNN（卷积神经网络）

- **卷积操作**：感受野（Receptive Field）概念、卷积核滑动机制、步长与填充、特征图尺寸计算公式。
- **经典架构演进**：LeNet（1998，手写数字）→ AlexNet（2012，ImageNet突破）→ VGG（2014，深层堆叠）→ ResNet（2015，残差连接）→ EfficientNet（2019，复合缩放）。
- **适用场景**：图像分类、目标检测、图像分割等计算机视觉任务。
- **局限性**：无法有效建模长距离依赖关系；对序列数据处理效率低；在NLP领域已被Transformer取代。

#### A3. RNN（循环神经网络）

- **基本RNN**：隐状态传递机制、时间步展开、随时间反向传播（BPTT）。梯度消失/爆炸问题。
- **LSTM**：输入门、遗忘门、输出门、细胞状态。通过门控机制缓解长距离依赖问题。
- **GRU**：重置门、更新门。LSTM的简化版本，参数更少。
- **适用场景**：序列建模（语言模型、机器翻译、时序预测），在Transformer之前是NLP的主流架构。
- **局限性**：无法并行计算（训练效率低）；长序列记忆能力有限；已被Transformer全面取代。

#### A4. GNN（图神经网络）

- **图卷积（GCN）**：基于邻接矩阵的特征聚合，谱图理论与空间方法。
- **消息传递机制**：节点特征聚合→变换→更新。GraphSAGE（归纳学习）、GAT（注意力机制引入图结构）。
- **适用场景**：社交网络分析、推荐系统、分子结构预测、知识图谱补全。
- **局限性**：扩展到大图困难；与Transformer融合的Graph Transformer是前沿方向。

#### A5. Transformer架构（核心）

- **自注意力机制**：
  - Query（Q）、Key（K）、Value（V）的线性变换：Q = XW_Q, K = XW_K, V = XW_V
  - 注意力权重计算：Attention(Q,K,V) = softmax(QK^T / √d_k) V
  - 缩放因子 √d_k 的作用：防止点积过大导致softmax梯度消失
  - 因果掩码（Causal Mask）：解码器中防止关注未来位置
- **多头注意力（MHA）**：
  - 将Q/K/V投影到h个不同的子空间
  - 各头独立计算注意力后拼接
  - 多头的作用：不同头关注不同子空间的信息
  - 头数选择的影响（8/32/64等）
- **位置编码**：
  - 正弦编码（Sinusoidal）：固定公式 sin/cos，可推广到更长序列
  - 可学习编码（Learned PE）：训练学习位置嵌入，固定最大长度限制
  - 旋转位置编码（RoPE）：通过旋转矩阵编码相对位置，外推性好，LLaMA/GPT-NeoX使用
  - ALiBi（Attention with Linear Biases）：线性偏置，无需位置编码，支持外推
- **前馈网络（FFN）**：
  - 两层线性变换+激活函数：FFN(x) = W2·activation(W1·x + b1) + b2
  - SwiGLU变体：FFN(x) = (Swish(xW1) ⊙ xV) W2，LLaMA使用
  - FFN的中间维度通常是模型维度的4倍
- **编码器-解码器架构**：
  - 编码器：双向自注意力 + FFN
  - 解码器：因果自注意力 + 交叉注意力 + FFN
  - 交叉注意力：Decoder的Q关注Encoder的K/V
- **Layer Normalization**：
  - Pre-Norm vs Post-Norm：Pre-Norm（先归一化再残差连接）训练更稳定，现代大模型多用Pre-Norm
  - RMSNorm：LayerNorm的简化版，去除中心化操作，LLaMA使用
- **残差连接**：缓解梯度消失，使深层网络可训练。x_out = x + SubLayer(x)

#### A6. 注意力机制变体

- **稀疏注意力（Sparse Attention）**：只计算部分位置对的注意力，降低O(n²)复杂度。Longformer（滑动窗口+全局token）、BigBird（随机+窗口+全局）。
- **线性注意力（Linear Attention）**：用核函数近似softmax，将复杂度从O(n²)降为O(n)。Performer（随机特征近似）、Linear Transformer。
- **多查询注意力（MQA）**：所有头共享同一组K/V，仅Q不同。大幅减少KV Cache显存，但精度有损。PaLM使用。
- **分组查询注意力（GQA）**：MQA与MHA的折中，将头分组，组内共享K/V。LLaMA-2 70B使用。
- **滑动窗口注意力（SWA）**：每个token只关注局部窗口内的token。Mistral使用，支持长序列。

#### A7. 大模型架构与原理

- **编码器型（BERT系列）**：双向注意力，适合理解任务（分类、NER、问答）。BERT、RoBERTa、DeBERTa。
- **解码器型（GPT系列）**：因果注意力（自回归），适合生成任务。GPT-2/3/4、LLaMA、GLM、Qwen、Mistral、Phi。
- **编码器-解码器型**：T5、BART。适合Seq2Seq任务（翻译、摘要）。
- **缩放定律（Scaling Law）**：模型性能与参数量、数据量、计算量的幂律关系。Chinchilla定律：最优数据/参数比约为20:1。
- **涌现能力（Emergent Abilities）**：模型规模超过某阈值后突然出现的能力（少样本学习、思维链推理、指令遵循）。
- **指令遵循（Instruction Following）**：经过指令微调后，模型能理解并执行自然语言指令。
- **大模型训练阶段**：
  - 预训练（Pre-training）：海量无标注文本上的自监督学习
  - SFT（Supervised Fine-Tuning）：指令-响应对上的监督微调
  - RLHF / DPO：基于人类偏好对齐

#### A8. 前沿算法技术

- **Mixture of Experts（MoE）**：将FFN替换为多个专家网络，路由器动态选择专家。增加参数量而不等比增加计算量。Mixtral 8x7B、DeepSeek-MoE。
- **长上下文建模**：位置编码外推（YaRN、NTK-aware）、注意力稀疏化、分块处理。支持128K+ token上下文。
- **多模态融合**：CLIP（图文对比学习）、LLaVA（视觉编码器+LLM）、Flamingo（交错的图文输入）。
- **强化学习基础**：MDP、策略梯度、PPO（Proximal Policy Optimization）、DPO（Direct Preference Optimization，RLHF的简化方案）。

#### A9. 模型选型方法论

- **业务场景分析维度**：推理速度要求、显存/内存限制、精度要求、成本预算、部署环境（云端/边缘）、开源协议。
- **开源模型对比**：
  - LLaMA系列（Meta）：社区生态最好，微调工具完善
  - GLM系列（智谱）：中英双语能力强，国内合规
  - Qwen系列（阿里）：多模态支持好，中文能力强
  - Mistral系列：效率优化好（GQA/SWA），Apache 2.0
  - Phi系列（微软）：小参数高性能，适合边缘部署
- **模型规模选择**：7B（消费级GPU可运行）、13B（A100单卡）、70B（多卡/量化部署）、405B（集群部署）。
- **选型Trade-off分析**：精度vs速度、参数量vs显存、通用能力vs领域专精、开源性vs性能。

---

### 3.2 大模型应用开发与微调

#### B1. 大模型微调

**B1.1 全量微调（Full Fine-tuning）**
- 更新模型所有参数，效果上限高但计算成本大。
- 需要大量GPU显存（参数量×(2字节梯度+2字节优化器状态+4字节参数)≈8倍参数量）。
- 适合有充足算力且追求最佳效果的场景。

**B1.2 参数高效微调（PEFT）**
- **LoRA（Low-Rank Adaptation）**：
  - 核心思想：冻结原始权重W，引入低秩分解 W + ΔW = W + BA，其中B∈R^(d×r), A∈R^(r×d)，r远小于d。
  - 训练时只更新B和A，参数量从d²降为2dr。
  - 推理时可将BA合并到W中，无额外推理开销。
  - 适用场景：资源受限、多任务适配（每个任务一个LoRA适配器）。
  - 关键超参数：秩r（通常8-64）、alpha缩放因子、目标模块（q_proj/k_proj/v_proj/o_proj等）。
- **QLoRA（Quantized LoRA）**：
  - 在4-bit量化的基座模型上应用LoRA，进一步降低显存需求。
  - NF4量化 + 双重量化 + 分页优化器。
  - 使7B模型微调仅需6GB显存。
- **Adapter Tuning**：在Transformer层中插入小型瓶颈网络（Bottleneck）。
- **Prefix Tuning**：在每层注意力前添加可学习的前缀向量。
- **P-Tuning v2**：类似Prefix Tuning，在每层添加可学习的prompt向量。

**B1.3 SFT（监督微调）**
- **指令数据构造**：Self-Instruct方法、Evol-Instruct方法、人工标注+质量筛选。
- **对话格式模板**：ChatML格式、LLaMA对话模板、Alpaca格式、ShareGPT格式。
- **训练超参数调优**：学习率（通常1e-5到5e-5）、Epoch数（通常2-5）、Batch Size、Warmup比例、权重衰减。

**B1.4 RLHF（基于人类反馈的强化学习）**
- **奖励模型训练**：收集人类偏好数据（A > B），训练奖励模型预测偏好。
- **PPO策略优化**：用奖励模型的信号通过PPO算法优化策略模型。需要Reference Model防止策略漂移。
- **DPO（直接偏好优化）**：绕过奖励模型，直接从偏好数据优化策略。数学上等价于RLHF但更简单稳定。

**B1.5 微调全流程实践**
- **数据准备**：数据清洗（去重/去噪/质量过滤）、格式化（指令-输入-输出）、数据增强。
- **训练配置**：分布式策略选择（DDP/FSDP/DeepSpeed ZeRO Stage 1/2/3）、混合精度训练（BF16/FP16）。
- **效果评估**：自动评估（perplexity/准确率）、人工评估、与基座模型对比。
- **常见问题**：过拟合（loss下降但泛化变差）、灾难性遗忘（微调后丧失通用能力）、训练不稳定（loss spike）。

#### B2. RAG（检索增强生成）

**B2.1 RAG基础流程**
- **文档解析**：PDF解析（PyMuPDF/Unstructured）、HTML解析（BeautifulSoup）、表格解析（Camelot/Table Transformer）、多格式统一处理。
- **文档切分策略**：
  - 固定长度切分（简单但可能截断语义）
  - 语义切分（基于句子/段落边界）
  - 递归切分（LangChain RecursiveCharacterTextSplitter，先按段落再按句子）
  - 重叠窗口（保留上下文连续性）
- **向量化（Embedding）**：
  - 模型选择：OpenAI text-embedding-3、BGE系列、E5系列、GTE系列、Jina Embeddings
  - 维度选择与存储成本权衡
  - 多语言支持
- **向量数据库**：
  - FAISS（Facebook，库级别，适合原型）
  - Milvus（分布式，生产级，支持十亿级向量）
  - Chroma（轻量级，适合中小规模）
  - Pinecone（云托管，免运维）
  - Qdrant（Rust实现，高性能）
  - Weaviate（支持混合检索）
- **检索与生成**：Top-K检索 → 上下文拼接 → Prompt构建 → LLM生成 → 答案后处理。

**B2.2 高级检索技术**
- **混合检索**：稠密向量检索（语义相似）+ 稀疏检索（BM25关键词匹配），融合两者优势。
- **重排序（Reranking）**：Cross-Encoder模型对检索结果重新打分排序。BGE-Reranker、Cohere Rerank。
- **上下文优化**：上下文压缩（去除冗余信息）、去重、按相关性排序、上下文窗口管理。
- **查询改写与扩展**：查询重写（HyDE：假设性文档嵌入）、多查询生成、子问题分解。

**B2.3 RAG关键挑战**
- **检索精度**：Embedding质量、Chunk大小调优、检索策略选择。
- **上下文长度管理**：长上下文导致注意力分散、token成本控制。
- **幻觉控制**：引用标注、置信度校准、拒答机制。
- **实时性与增量更新**：文档更新后的索引重建策略、流式索引。

**B2.4 高级RAG架构**
- **自适应RAG（Adaptive RAG）**：根据查询复杂度动态选择检索策略（直接回答/单次检索/多跳检索）。
- **多跳检索（Multi-hop RAG）**：复杂问题需要多次检索，前一次检索结果指导下一次检索。
- **GraphRAG**：结合知识图谱，从文档中抽取实体关系构建图谱，基于图结构检索。
- **Self-RAG**：模型自我判断是否需要检索、检索结果是否相关、生成答案是否被支持。

#### B3. Prompt工程

**B3.1 基础技巧**
- **角色设定**：System Prompt定义AI角色、能力边界、行为规范。
- **任务描述与约束**：明确的任务指令、输出要求、限制条件。
- **示例引导（Few-shot）**：提供输入-输出示例帮助模型理解任务模式。Zero-shot vs One-shot vs Few-shot。
- **输出格式控制**：指定输出格式（JSON/Markdown/表格）、结构化输出约束。

**B3.2 高级技巧**
- **思维链（CoT）**：引导模型逐步推理（"Let's think step by step"），显著提升数学/逻辑推理能力。
- **少样本学习**：精心设计的示例选择、示例排序对效果的影响。
- **结构化输出**：JSON Mode、Function Calling、Structured Output（Pydantic Schema约束）。
- **自洽性（Self-Consistency）**：多次采样推理路径，取多数投票结果。
- **思维树（ToT）**：将推理过程组织为树结构，支持回溯和多路径探索。

**B3.3 Agent级Prompt技术**
- **ReAct**：交替进行推理（Thought）和行动（Action），观察行动结果（Observation）后继续推理。
- **Reflexion**：Agent执行后自我反思，根据反馈改进后续行动。
- **Plan-and-Execute**：先制定整体计划再逐步执行。
- **多Agent协作Prompt**：不同角色Agent的Prompt设计、Agent间通信协议。

#### B4. Agent系统

**B4.1 Agent架构设计**
- **核心循环**：感知（接收输入）→ 规划（分解任务）→ 执行（调用工具/API）→ 反思（评估结果）→ 循环。
- **工具调用**：Function Calling、Tool Use、API集成。工具描述（OpenAI Function Schema）、工具选择策略。
- **记忆系统**：
  - 短期记忆：对话上下文、工作记忆
  - 长期记忆：向量数据库存储历史交互、经验库
  - 记忆检索与更新策略

**B4.2 多Agent协作**
- **Agent角色分工**：规划者、执行者、评审者、协调者。
- **通信协议**：消息传递、共享黑板模式、发布订阅。
- **编排框架**：
  - LangGraph：基于图的状态机Agent编排
  - AutoGen（微软）：多Agent对话框架
  - CrewAI：角色扮演式多Agent协作
  - MetaGPT：软件公司模拟的多Agent系统

**B4.3 人机协同**
- **Human-in-the-loop**：关键决策节点引入人工审核。
- **人在环上反馈**：人类反馈持续优化Agent行为。

#### B5. 大模型应用全流程
- **模型选型与评估**：根据业务需求选择模型、评估基线效果。
- **部署推理**：模型加载、推理服务部署、负载均衡。
- **效果评估与迭代**：建立评估指标、A/B测试、持续优化。
- **端到端方案设计**：从需求分析到上线的完整流程。

---

### 3.3 AI模型应用设计开发能力

#### C1. AI应用架构设计

- **架构模式**：单体架构（原型快速验证）、微服务架构（生产级，独立扩展）、事件驱动架构（异步处理、解耦）。
- **数据处理流水线**：数据采集 → 清洗 → 预处理 → 向量化 → 索引 → 检索 → 生成 → 后处理。
- **模型选型与集成**：多模型路由（根据任务类型选择模型）、模型级联（粗排→精排）、模型ensemble。
- **业务逻辑编排**：工作流引擎（DAG/状态机）、条件分支、循环、异常处理。
- **API设计**：REST API（同步）、gRPC（高性能RPC）、WebSocket（流式输出）、SSE（Server-Sent Events）。

#### C2. 核心模块开发

- **数据预处理**：文本清洗、格式标准化、敏感信息脱敏、多格式解析。
- **结果后处理**：答案提取、引用标注、格式美化、安全过滤。
- **模型API调用封装**：统一接口封装（OpenAI/Anthropic/本地模型）、重试机制、超时处理、限流。
- **Prompt管理**：版本控制、A/B测试、模板管理、动态变量注入。
- **缓存层**：语义缓存（相似查询复用结果）、精确缓存、多级缓存策略。

#### C3. RAG应用开发

- **文档解析实现**：多格式解析器集成、OCR集成、表格结构识别。
- **向量检索实现**：Embedding模型加载、向量索引构建、相似度搜索。
- **提示词构建**：上下文注入、引用标注模板、系统提示词设计。
- **结果生成与引用标注**：生成答案中标注信息来源、置信度展示。
- **端到端Pipeline**：从文档上传到答案生成的完整流程。

#### C4. 复杂应用场景开发

- **长文本处理**：Map-Reduce摘要、滑动窗口、层次化处理。
- **多轮对话管理**：对话状态追踪（DST）、上下文窗口管理、历史对话压缩/摘要。
- **输出格式控制**：JSON Schema约束、Pydantic模型验证、结构化输出保障。
- **流式输出**：Token-by-token流式生成、前端SSE/WebSocket对接、打字机效果。
- **多模态交互**：图文混合输入、语音输入输出、视频理解。

#### C5. 性能调优与高可用

- **推理延迟优化**：模型量化、批处理、KV Cache、投机解码、模型蒸馏。
- **并发处理**：异步推理、连续批处理、请求队列管理、自动扩缩容。
- **资源利用率**：GPU利用率监控、显存管理、模型并行策略。
- **高可用设计**：负载均衡（多实例部署）、故障恢复（健康检查+自动重启）、降级策略（大模型→小模型→规则）。
- **成本优化**：模型路由（简单任务用小模型）、缓存命中率提升、按需加载。

#### C6. 复杂业务场景方案设计

- **多模型协作**：模型路由策略、级联调用、结果融合。
- **工具调用编排**：工具注册中心、调用链管理、错误处理。
- **Agent系统设计**：任务分解、工具规划、执行监控、结果验证。
- **端到端设计方法论**：需求分析→技术选型→架构设计→原型开发→评估迭代→生产部署。

#### C7. AI平台建设

- **模型管理平台**：模型注册、版本管理、AB测试、灰度发布。
- **Prompt管理平台**：Prompt模板管理、版本控制、效果评估。
- **应用编排平台**：可视化工作流设计、组件复用、一键部署。
- **评估与监控平台**：在线评估、性能监控、告警体系。

---

### 3.4 模型微调与性能优化能力

#### D1. 微调全流程

- **数据准备**：数据清洗（去重/去噪/格式标准化）、数据格式化（Alpaca/ShareGPT格式）、数据增强（回译/改写）、质量筛选（长度/多样性/毒性过滤）。
- **训练配置**：学习率选择（LoRA通常1e-4到5e-4，全量微调1e-5到5e-5）、Batch Size与梯度累积、Epoch数、Warmup比例（通常3%-10%）、权重衰减、梯度裁剪。
- **分布式训练**：
  - DDP（DistributedDataParallel）：数据并行，每张卡完整模型副本。
  - FSDP（FullyShardedDataParallel）：分片数据并行，参数/梯度/优化器状态分片到各卡。
  - DeepSpeed ZeRO：Stage 1（优化器状态分片）、Stage 2（+梯度分片）、Stage 3（+参数分片）。
- **微调执行**：训练监控（Loss/LR/Gradient Norm）、检查点保存、早停策略。
- **效果评估**：自动评估（BLEU/ROUGE/准确率）、人工评估（质量/相关性/安全性）、与基座模型对比、A/B测试。

#### D2. 针对性微调方案设计

- **领域适配微调**：注入领域知识（医疗/法律/金融）、领域术语对齐、领域特定任务优化。
- **指令微调**：通用指令遵循能力提升、多任务指令混合、指令复杂度梯度。
- **多任务学习**：任务采样策略、任务权重平衡、负迁移避免。
- **数据混合策略**：通用数据与领域数据比例、数据课程学习（先易后难）。

#### D3. 推理引擎与优化

- **vLLM**：
  - PagedAttention：将KV Cache组织为固定大小的Page，减少显存碎片，实现高效内存管理。
  - 连续批处理（Continuous Batching）：动态插入/移除请求，最大化GPU利用率。
  - 张量并行：将模型权重分片到多GPU，支持大模型多卡推理。
  - 支持多种量化格式、前缀缓存、流式输出。
- **TGI（Text Generation Inference）**：HuggingFace的推理服务器，支持多模型并发、张量并行、Flash Attention。
- **TensorRT-LLM**：NVIDIA的高性能推理引擎，极致优化NVIDIA GPU推理。
- **llama.cpp / Ollama**：CPU/边缘设备推理，GGUF量化格式，极低资源消耗。
- **SGLang**：高性能推理引擎，支持RadixAttention（前缀树缓存）和结构化生成。

#### D4. 推理性能优化技术

- **量化**：
  - GPTQ：训练后量化，基于二阶信息的逐层量化。
  - AWQ（Activation-aware Weight Quantization）：激活感知的权重量化，保持关键通道精度。
  - GGUF：llama.cpp的量化格式，支持2-8 bit。
  - FP8：NVIDIA H100支持的8位浮点格式。
  - 量化对精度的影响与补偿策略。
- **剪枝（Pruning）**：结构化剪枝（移除整个神经元/层）、非结构化剪枝（稀疏化）、稀疏训练。
- **知识蒸馏（Knowledge Distillation）**：Teacher-Student架构、软标签学习、中间层蒸馏。大模型蒸馏到小模型。
- **KV Cache管理**：PagedAttention（vLLM）、KV Cache量化、KV Cache压缩。
- **动态批处理（Dynamic Batching）**：请求级动态调度、连续批处理、批大小自适应。
- **前缀缓存（Prefix Caching）**：共享系统Prompt的KV Cache复用、SGLang RadixAttention。
- **投机解码（Speculative Decoding）**：小模型快速生成草稿→大模型验证→接受/拒绝。Medusa、EAGLE。
- **显存优化**：Flash Attention（IO-aware的注意力计算）、FlashInfer（高性能注意力内核）、PagedAttention。
- **混合精度推理**：FP16/BF16推理、关键层保留FP32、INT8/INT4混合。

#### D5. 评估体系

- **微调效果评估**：任务特定指标（准确率/F1/BLEU）、通用能力保持度评估、人工偏好评估。
- **推理性能基准**：吞吐量（tokens/s）、延迟（TTFT/TPOT）、并发数、显存占用。
- **模型质量评估**：MMLU（多任务理解）、HumanEval（代码生成）、MT-Bench（多轮对话）、AlpacaEval（指令跟随）、AGIEval。
- **在线监控**：延迟监控、错误率监控、用户满意度反馈、漂移检测。

#### D6. 企业级微调与推理服务

- **微调服务框架**：自动化数据流水线、训练任务调度、模型版本管理、效果回归测试。
- **推理服务框架**：模型热加载、自动扩缩容、多模型路由、SLA保障。
- **极端资源约束优化**：边缘设备部署、CPU推理优化、极致量化压缩。
- **前沿技术追踪**：新架构适配（Mamba/Linear Attention）、极致压缩（1-bit LLM）、动态推理（早退/层跳过）。

---

## 4. 分领域学习资料规划

### 4.1 AI算法原理与模型选型 — 学习资料

#### 📚 书籍

| 序号 | 名称 | 作者/来源 | 类型 | 难度 | 推荐理由 |
|------|------|----------|------|------|----------|
| 1 | **《深度学习》（Deep Learning, "花书"）** | Ian Goodfellow, Yoshua Bengio, Aaron Courville | 书籍 | ⭐⭐⭐ | 深度学习领域圣经级教材，系统覆盖神经网络、CNN、RNN、优化算法等全部基础理论。第一部分（应用数学与机器学习基础）和第二部分（深度网络）是理解所有AI算法的根基。MIT Press官方免费在线版：https://www.deeplearningbook.org/ |
| 2 | **《动手学深度学习》（Dive into Deep Learning, D2L）** | 李沐, Aston Zhang, Zachary C. Lipton, Alexander J. Smola | 书籍+代码 | ⭐⭐ | 理论+代码完美结合，覆盖CNN/RNN/Transformer/注意力机制等全部核心架构。每章都有PyTorch/MXNet代码实现，边学边练。中文在线版：https://zh.d2l.ai/ |
| 3 | **《神经网络与深度学习》（nndl.github.io）** | 邱锡鹏 | 书籍 | ⭐⭐⭐ | 复旦大学邱锡鹏教授著作，中文深度学习教材标杆。系统讲解神经网络基础、CNN、RNN、Transformer等。在线版：https://nndl.github.io/ |
| 4 | **《Natural Language Processing with Transformers》** | Lewis Tunstall, Leandro von Werra, Thomas Wolf (HuggingFace) | 书籍 | ⭐⭐⭐ | O'Reilly出版，HuggingFace团队撰写。聚焦Transformer架构与NLP应用，覆盖BERT/GPT/T5等模型原理与HuggingFace Transformers库实践。 |
| 5 | **《Build a Large Language Model (From Scratch)》** | Sebastian Raschka | 书籍 | ⭐⭐⭐ | 从零实现一个GPT模型，涵盖Tokenization、注意力机制、预训练、微调全流程。最佳的大模型原理理解实践书。 |
| 6 | **《Speech and Language Processing》（SLP3）** | Daniel Jurafsky, James H. Martin | 书籍 | ⭐⭐⭐ | NLP领域经典教材，第3版新增深度学习与LLM章节。Stanford在线草稿版：https://web.stanford.edu/~jurafsky/slp3/ |

#### 🎓 大学课程

| 序号 | 名称 | 来源 | 类型 | 难度 | 推荐理由 |
|------|------|------|------|------|----------|
| 1 | **CS224N: Natural Language Processing with Deep Learning** | Stanford University | 大学课程 | ⭐⭐⭐⭐ | Stanford NLP旗舰课程，Christopher Manning主讲。覆盖词向量、RNN/LSTM、Transformer、大模型预训练。2024版含GPT/LLaMA内容。课程网站：https://web.stanford.edu/class/cs224n/ |
| 2 | **CS231N: CNN for Visual Recognition** | Stanford University | 大学课程 | ⭐⭐⭐ | Stanford CV旗舰课程，深入讲解CNN原理与视觉应用。理解卷积操作、感受野、经典架构。课程网站：http://cs231n.stanford.edu/ |
| 3 | **CS25: Transformers United** | Stanford University | 大学课程 | ⭐⭐⭐⭐ | Stanford专门讲Transformer的课程，邀请Transformer领域前沿研究者做讲座。涵盖注意力机制变体、大模型前沿。课程网站：https://web.stanford.edu/class/cs25/ |
| 4 | **CS224W: Machine Learning with Graphs** | Stanford University | 大学课程 | ⭐⭐⭐ | Stanford图机器学习课程，深入GNN/GCN/GAT。理解图神经网络原理与应用。课程网站：https://web.stanford.edu/class/cs224w/ |
| 5 | **6.S191: Introduction to Deep Learning** | MIT | 大学课程 | ⭐⭐⭐ | MIT深度学习入门课程，覆盖CNN/RNN/Transformer/生成模型。每年更新前沿内容。课程网站：http://introtodeeplearning.com/ |
| 6 | **NYU Deep Learning (Yann LeCun)** | NYU | 大学课程 | ⭐⭐⭐⭐ | Yann LeCun主讲，深度学习理论深度讲解。YouTube有完整视频。 |
| 7 | **Deep Learning Specialization** | Andrew Ng / DeepLearning.AI / Coursera | 在线课程 | ⭐⭐ | Andrew Ng经典深度学习课程，适合打基础。覆盖神经网络、CNN、RNN、序列模型。Coursera链接：https://www.coursera.org/specializations/deep-learning |
| 8 | **The Illustrated Transformer** | Jay Alammar | 博客 | ⭐⭐ | Transformer架构可视化讲解的经典博客，图解Q/K/V计算过程。必读入门材料。博客链接：https://jalammar.github.io/illustrated-transformer/ |

#### 📝 博客与文章

| 序号 | 名称 | 作者/来源 | 类型 | 难度 | 推荐理由 |
|------|------|----------|------|------|----------|
| 1 | **The Annotated Transformer** | Harvard NLP | 博客+代码 | ⭐⭐⭐ | 逐行注释Transformer原始论文的PyTorch实现，理解架构细节的最佳材料。http://nlp.seas.harvard.edu/annotated-transformer/ |
| 2 | **The Illustrated BERT, ELMo, and co.** | Jay Alammar | 博客 | ⭐⭐ | 可视化讲解BERT/GPT等预训练模型的架构差异。https://jalammar.github.io/illustrated-bert/ |
| 3 | **The Illustrated GPT-1** | Jay Alammar | 博客 | ⭐⭐ | GPT架构可视化讲解。https://jalammar.github.io/illustrated-gpt-2/ |
| 4 | **RoFormer: Enhanced Transformer with Rotary Position Embedding** | Jianlin Su | 论文/博客 | ⭐⭐⭐ | RoPE旋转位置编码原始论文，理解现代大模型位置编码的关键。 |
| 5 | **Attention Is All You Need** | Vaswani et al. | 论文 | ⭐⭐⭐ | Transformer原始论文，必读。理解自注意力、多头注意力、位置编码的起源。 |
| 6 | **Scaling Laws for Neural Language Models** | Kaplan et al. (OpenAI) | 论文 | ⭐⭐⭐⭐ | 缩放定律原始论文，理解大模型性能与规模的关系。 |
| 7 | **Emergent Abilities of Large Language Models** | Wei et al. (Google) | 论文 | ⭐⭐⭐⭐ | 涌现能力研究论文，理解大模型能力突现现象。 |
| 8 | **LLaMA: Open and Efficient Foundation Language Models** | Meta AI | 论文 | ⭐⭐⭐ | LLaMA论文，理解现代开源大模型的架构设计（RMSNorm/SwiGLU/RoPE）。 |
| 9 | **Mixtral of Experts** | Mistral AI | 论文 | ⭐⭐⭐⭐ | MoE架构的代表性开源实现论文。 |
| 10 | **Lil'Log Blog (Lilian Weng)** | Lilian Weng (OpenAI) | 博客 | ⭐⭐⭐⭐ | OpenAI研究员Lilian Weng的技术博客，覆盖LLM、Agent、RLHF等前沿主题，深度极高。https://lilianweng.github.io/ |
| 11 | **Sebastian Raschka Blog** | Sebastian Raschka | 博客 | ⭐⭐⭐ | 深度学习与大模型技术深度解析博客，包含LoRA原理、量化技术等详细讲解。https://magazine.sebastianraschka.com/ |
| 12 | **Jay Alammar's Blog** | Jay Alammar | 博客 | ⭐⭐ | 大模型架构可视化讲解系列，入门必读。https://jalammar.github.io/ |

#### 🔧 开源项目/代码库

| 序号 | 名称 | 来源 | 类型 | 难度 | 推荐理由 |
|------|------|------|------|------|----------|
| 1 | **HuggingFace Transformers** | HuggingFace | 开源库 | ⭐⭐⭐ | 最重要的大模型库，支持几乎所有主流模型（BERT/GPT/LLaMA/GLM等）。阅读源码理解模型实现细节。https://github.com/huggingface/transformers |
| 2 | **nanoGPT** | Andrej Karpathy | 开源项目 | ⭐⭐⭐ | Karpathy用约300行PyTorch从零实现GPT，理解大模型训练流程的最佳学习项目。https://github.com/karpathy/nanoGPT |
| 3 | **llm.c** | Andrej Karpathy | 开源项目 | ⭐⭐⭐⭐ | 用C/CUDA从零实现GPT训练，深入理解底层计算。https://github.com/karpathy/llm.c |
| 4 | **PyTorch Transformer Tutorial** | PyTorch官方 | 教程代码 | ⭐⭐ | PyTorch官方Transformer实现教程。https://pytorch.org/tutorials/beginner/transformer_tutorial.html |
| 5 | **The Annotated Transformer (Harvard)** | Harvard NLP | 教学项目 | ⭐⭐⭐ | 逐行注释的Transformer实现。http://nlp.seas.harvard.edu/annotated-transformer/ |

---

### 4.2 大模型应用开发与微调 — 学习资料

#### 📚 书籍

| 序号 | 名称 | 作者/来源 | 类型 | 难度 | 推荐理由 |
|------|------|----------|------|------|----------|
| 1 | **《大语言模型》** | 赵鑫, 李军毅等（中国人民大学） | 书籍 | ⭐⭐⭐ | 中文大模型教材，系统讲解LLM预训练、微调、对齐、提示工程、RAG等。覆盖面广且深度适中。 |
| 2 | **《LangChain实战》** | 李特丽 | 书籍 | ⭐⭐ | LangChain框架实战指南，覆盖RAG、Agent、Chain等应用开发。 |
| 3 | **《Hands-On Large Language Models》** | Jay Alammar, Maarten Grootendorst | 书籍 | ⭐⭐⭐ | O'Reilly出版，Jay Alammar（可视化Transformer作者）新作，聚焦大模型应用开发实践。 |
| 4 | **《Prompt Engineering Guide》** | DAIR.AI | 电子书 | ⭐⭐ | Prompt工程全面指南，从基础到高级技巧。在线版：https://www.promptingguide.ai/ |
| 5 | **《动手做大模型》** | datawhalechina | 开源教程 | ⭐⭐ | Datawhale开源的大模型实践教程，覆盖微调、推理、应用开发全链路。https://github.com/datawhalechina/hugging-llm |

#### 🎓 大学课程

| 序号 | 名称 | 来源 | 类型 | 难度 | 推荐理由 |
|------|------|------|------|------|----------|
| 1 | **CS25: Transformers United** | Stanford University | 大学课程 | ⭐⭐⭐⭐ | 前沿Transformer研究讲座系列，涵盖大模型应用、微调、对齐等。https://web.stanford.edu/class/cs25/ |
| 2 | **Full Stack Deep Learning (FSDL)** | FSDL社区 | 在线课程 | ⭐⭐⭐ | 全栈深度学习课程，2024版覆盖LLM应用开发、RAG、Agent等工程实践。https://fullstackdeeplearning.com/ |
| 3 | **Princeton: COS 597R (Topics in LLMs)** | Princeton University | 大学课程 | ⭐⭐⭐⭐ | Princeton LLM前沿课程，覆盖微调、对齐、推理优化。 |
| 4 | **CMU 11-711 Advanced NLP** | CMU | 大学课程 | ⭐⭐⭐⭐ | CMU高级NLP课程，Graham Neubig主讲，覆盖LLM应用与微调。 |
| 5 | **DeepLearning.AI Short Courses** | Andrew Ng / DeepLearning.AI | 在线课程 | ⭐⭐ | 系列短课程：LangChain for LLM App Development、Finetuning Large Language Models、Building and Evaluating Advanced RAG等。https://www.deeplearning.ai/short-courses/ |

#### 📝 博客与文章

| 序号 | 名称 | 作者/来源 | 类型 | 难度 | 推荐理由 |
|------|------|----------|------|------|----------|
| 1 | **LoRA: Low-Rank Adaptation of Large Language Models** | Hu et al. (Microsoft) | 论文 | ⭐⭐⭐ | LoRA原始论文，理解参数高效微调的核心思想。 |
| 2 | **QLoRA: Efficient Finetuning of Quantized LLMs** | Dettmers et al. (UW) | 论文 | ⭐⭐⭐⭐ | QLoRA论文，4-bit量化+LoRA，使大模型微调平民化。 |
| 3 | **Training language models to follow instructions with human feedback (InstructGPT)** | OpenAI | 论文 | ⭐⭐⭐⭐ | RLHF原始论文，理解SFT→RM→PPO三阶段对齐流程。 |
| 4 | **Direct Preference Optimization (DPO)** | Rafailov et al. (Stanford) | 论文 | ⭐⭐⭐⭐ | DPO原始论文，RLHF的简化替代方案，直接从偏好数据优化。 |
| 5 | **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks** | Lewis et al. (Facebook AI) | 论文 | ⭐⭐⭐ | RAG原始论文，理解检索增强生成的基础概念。 |
| 6 | **Chain-of-Thought Prompting Elicits Reasoning in Large Language Models** | Wei et al. (Google) | 论文 | ⭐⭐⭐ | CoT思维链原始论文，Prompt工程核心技巧。 |
| 7 | **ReAct: Synergizing Reasoning and Acting in Language Models** | Yao et al. (Princeton) | 论文 | ⭐⭐⭐ | ReAct原始论文，Agent核心范式。 |
| 8 | **Tree of Thoughts** | Yao et al. (Princeton) | 论文 | ⭐⭐⭐ | 思维树推理框架，高级Prompt技巧。 |
| 9 | **Self-Consistency Improves Chain of Thought Reasoning** | Wang et al. (Google) | 论文 | ⭐⭐⭐ | 自洽性方法，多次采样+投票提升推理准确率。 |
| 10 | **RAG Survey: Retrieval-Augmented Generation for Large Language Models** | Gao et al. | 综述论文 | ⭐⭐⭐ | RAG技术全面综述，从基础到高级架构。https://arxiv.org/abs/2312.10997 |
| 11 | **A Survey on Large Language Model based Autonomous Agents** | Wang et al. | 综述论文 | ⭐⭐⭐⭐ | LLM Agent技术全面综述。 |
| 12 | **Lil'Log: Prompt Engineering** | Lilian Weng | 博客 | ⭐⭐⭐ | OpenAI研究员的Prompt工程深度总结。https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/ |
| 13 | **Lil'Log: LLM Powered Autonomous Agents** | Lilian Weng | 博客 | ⭐⭐⭐⭐ | Agent系统设计深度总结，覆盖规划/记忆/工具使用。https://lilianweng.github.io/posts/2023-06-23-agent/ |
| 14 | **GraphRAG Paper** | Microsoft Research | 论文 | ⭐⭐⭐⭐ | 微软GraphRAG论文，知识图谱增强RAG的前沿方案。 |
| 15 | **Self-RAG: Learning to Retrieve, Generate, and Critique** | Asai et al. | 论文 | ⭐⭐⭐⭐ | Self-RAG论文，模型自主判断检索需求。 |

#### 🔧 开源项目/代码库

| 序号 | 名称 | 作者/来源 | 类型 | 难度 | 推荐理由 |
|------|------|------|------|------|----------|
| 1 | **LangChain** | LangChain | 开源框架 | ⭐⭐⭐ | 最流行的LLM应用开发框架，支持RAG/Agent/Chain/工具调用。阅读源码学习应用架构设计。https://github.com/langchain-ai/langchain |
| 2 | **LlamaIndex** | LlamaIndex | 开源框架 | ⭐⭐⭐ | 专注RAG的数据框架，提供文档解析/索引/检索/生成全流程。RAG开发首选。https://github.com/run-llama/llama_index |
| 3 | **PEFT (HuggingFace)** | HuggingFace | 开源库 | ⭐⭐⭐ | HuggingFace官方参数高效微调库，支持LoRA/QLoRA/Adapter/Prefix Tuning等。微调实践必备。https://github.com/huggingface/peft |
| 4 | **TRL (Transformer Reinforcement Learning)** | HuggingFace | 开源库 | ⭐⭐⭐⭐ | HuggingFace的RLHF/DPO训练库，支持SFT/奖励模型/PPO/DPO全流程。https://github.com/huggingface/trl |
| 5 | **Axolotl** | OpenAccess AI Collective | 开源工具 | ⭐⭐⭐ | 高级微调工具，支持多种模型和微调方法，配置驱动，生产级。https://github.com/OpenAccess-AI-Collective/axolotl |
| 6 | **Unsloth** | UnslothAI | 开源库 | ⭐⭐⭐ | 超快速LoRA微调库，比标准实现快2-5倍，显存占用减少60%。https://github.com/unslothai/unsloth |
| 7 | **LLaMA-Factory** | hiyouga | 开源工具 | ⭐⭐⭐ | 中文社区最流行的微调工具，WebUI界面，支持多种模型和微调方法。https://github.com/hiyouga/LLaMA-Factory |
| 8 | **AutoGen** | Microsoft | 开源框架 | ⭐⭐⭐⭐ | 微软多Agent对话框架，支持复杂多Agent协作场景。https://github.com/microsoft/autogen |
| 9 | **LangGraph** | LangChain | 开源框架 | ⭐⭐⭐⭐ | 基于图状态机的Agent编排框架，支持循环/条件分支/人机协同。https://github.com/langchain-ai/langgraph |
| 10 | **CrewAI** | CrewAI | 开源框架 | ⭐⭐⭐ | 角色扮演式多Agent协作框架，API简洁易用。https://github.com/crewAIInc/crewAI |
| 11 | **MetaGPT** | DeepWisdom | 开源项目 | ⭐⭐⭐⭐ | 模拟软件公司的多Agent系统，展示多Agent协作的完整范式。https://github.com/geekan/MetaGPT |
| 12 | **Dify** | LangGenius | 开源平台 | ⭐⭐⭐ | 开源LLM应用开发平台，可视化编排RAG/Agent/工作流。https://github.com/langgenius/dify |
| 13 | **Haystack** | deepset | 开源框架 | ⭐⭐⭐ | 生产级NLP/RAG框架，提供文档处理/检索/问答Pipeline。https://github.com/deepset-ai/haystack |
| 14 | **RAGAS** | Exploding Gradients | 开源工具 | ⭐⭐⭐ | RAG系统评估框架，提供检索质量/生成质量/忠实度等评估指标。https://github.com/explodinggradients/ragas |
| 15 | **Chroma** | Chroma | 向量数据库 | ⭐⭐ | 轻量级向量数据库，适合RAG原型开发。https://github.com/chroma-core/chroma |
| 16 | **Milvus** | Zilliz | 向量数据库 | ⭐⭐⭐⭐ | 生产级分布式向量数据库，支持十亿级向量检索。https://github.com/milvus-io/milvus |
| 17 | **FastChat** | LMSYS | 开源项目 | ⭐⭐⭐ | 多模型对话平台，支持模型 serving 和评估。https://github.com/lm-sys/FastChat |
| 18 | **OpenCompass** | 上海AI实验室 | 开源工具 | ⭐⭐⭐ | 大模型评测框架，支持MMLU/HumanEval等主流评测基准。https://github.com/open-compass/opencompass |

---

### 4.3 AI模型应用设计开发能力 — 学习资料

#### 📚 书籍

| 序号 | 名称 | 作者/来源 | 类型 | 难度 | 推荐理由 |
|------|------|----------|------|------|----------|
| 1 | **《Designing Machine Learning Systems》** | Chip Huyen | 书籍 | ⭐⭐⭐ | ML系统设计圣经，覆盖ML系统全生命周期：数据管道、模型部署、监控、迭代。对AI应用架构设计有直接指导意义。 |
| 2 | **《Machine Learning Engineering》** | Andriy Burkov | 书籍 | ⭐⭐⭐ | ML工程实践指南，覆盖从原型到生产的全流程工程化方法。 |
| 3 | **《AI Engineering》** | Chip Huyen | 书籍 | ⭐⭐⭐⭐ | Chip Huyen 2025年新作，聚焦AI工程化，覆盖LLM应用架构、评估、部署、监控。 |
| 4 | **《Building LLM Powered Applications》** | Valentina Alto | 书籍 | ⭐⭐ | LLM应用开发实践指南，覆盖Prompt工程/RAG/Agent开发。 |
| 5 | **《LLM应用开发实践》** | 黄佳 | 书籍 | ⭐⭐ | 中文LLM应用开发实践书籍，覆盖LangChain/RAG/Agent实战。 |

#### 🎓 大学课程与在线课程

| 序号 | 名称 | 来源 | 类型 | 难度 | 推荐理由 |
|------|------|------|------|------|----------|
| 1 | **Full Stack Deep Learning (FSDL)** | FSDL社区 | 在线课程 | ⭐⭐⭐ | 全栈深度学习课程，2024版覆盖LLM应用工程化、部署、监控。https://fullstackdeeplearning.com/ |
| 2 | **DeepLearning.AI: Building Systems with the ChatGPT API** | DeepLearning.AI | 在线课程 | ⭐⭐ | 使用ChatGPT API构建应用系统，覆盖Prompt链/分类/评估/总结。https://www.deeplearning.ai/short-courses/ |
| 3 | **DeepLearning.AI: LangChain for LLM App Development** | DeepLearning.AI | 在线课程 | ⭐⭐ | LangChain框架入门课程，学习LLM应用开发基础。 |
| 4 | **Berkeley CS294/194-196: Large Language Model Agents** | UC Berkeley | 大学课程 | ⭐⭐⭐⭐ | Berkeley LLM Agent课程，前沿Agent系统设计。 |
| 5 | **MLOps Zoomcamp** | DataTalksClub | 在线课程 | ⭐⭐⭐ | MLOps实践课程，覆盖模型部署/监控/管道自动化。https://github.com/DataTalksClub/mlops-zoomcamp |

#### 📝 博客与文章

| 序号 | 名称 | 作者/来源 | 类型 | 难度 | 推荐理由 |
|------|------|----------|------|------|----------|
| 1 | **Chip Huyen Blog** | Chip Huyen | 博客 | ⭐⭐⭐⭐ | ML/AI工程化领域顶级博客，覆盖LLM部署/监控/RAG架构等。https://huyenchip.com/blog/ |
| 2 | **Eugene Yan Blog** | Eugene Yan | 博客 | ⭐⭐⭐ | ML系统设计与工程实践博客，覆盖评估/部署/MLOps。https://eugeneyan.com/ |
| 3 | **Hamel Husain Blog** | Hamel Husain | 博客 | ⭐⭐⭐ | LLM应用评估与工程实践博客。https://hamel.dev/ |
| 4 | **OpenAI Cookbook** | OpenAI | 技术文档 | ⭐⭐⭐ | OpenAI官方最佳实践库，覆盖RAG/Agent/结构化输出/评估等应用模式。https://github.com/openai/openai-cookbook |
| 5 | **Anthropic Prompt Engineering Guide** | Anthropic | 技术文档 | ⭐⭐⭐ | Anthropic官方Prompt工程指南，含Claude最佳实践。https://docs.anthropic.com/claude/docs/prompt-engineering |
| 6 | **Prompt Engineering Guide (DAIR.AI)** | DAIR.AI | 开源指南 | ⭐⭐ | Prompt工程全面指南，从基础到高级。https://www.promptingguide.ai/ |
| 7 | **LangChain Blog** | LangChain | 博客 | ⭐⭐⭐ | LangChain官方博客，RAG/Agent/架构设计最佳实践。https://blog.langchain.dev/ |
| 8 | **LlamaIndex Blog** | LlamaIndex | 博客 | ⭐⭐⭐ | RAG架构设计最佳实践与高级技术。https://www.llamaindex.ai/blog |

#### 🔧 开源项目/代码库

| 序号 | 名称 | 作者/来源 | 类型 | 难度 | 推荐理由 |
|------|------|------|------|------|----------|
| 1 | **LangChain** | LangChain | 开源框架 | ⭐⭐⭐ | LLM应用开发框架标杆，学习应用架构/数据处理/模型集成/Agent设计。https://github.com/langchain-ai/langchain |
| 2 | **Dify** | LangGenius | 开源平台 | ⭐⭐⭐ | 生产级LLM应用平台，学习完整应用架构设计。https://github.com/langgenius/dify |
| 3 | **FastGPT** | labring | 开源平台 | ⭐⭐⭐ | 国产开源知识库问答系统，学习RAG应用全栈开发。https://github.com/labring/FastGPT |
| 4 | **AnythingLLM** | Mintplex Labs | 开源项目 | ⭐⭐⭐ | 全栈RAG应用，学习文档处理/向量检索/多模型集成。https://github.com/Mintplex-Labs/anything-llm |
| 5 | **Open WebUI** | Open WebUI | 开源项目 | ⭐⭐⭐ | 类ChatGPT Web界面，学习多模型管理/对话管理/RAG集成。https://github.com/open-webui/open-webui |
| 6 | **Bisheng** | dataelement | 开源平台 | ⭐⭐⭐⭐ | 国产企业级LLM应用开发平台，学习企业级架构设计。https://github.com/dataelement/bisheng |
| 7 | **Flowise** | FlowiseAI | 开源平台 | ⭐⭐ | 可视化LLM应用编排工具，学习工作流设计。https://github.com/FlowiseAI/Flowise |
| 8 | **n8n** | n8n | 开源平台 | ⭐⭐⭐ | 工作流自动化平台，含AI节点，学习业务逻辑编排。https://github.com/n8n-io/n8n |

---

### 4.4 模型微调与性能优化能力 — 学习资料

#### 📚 书籍

| 序号 | 名称 | 作者/来源 | 类型 | 难度 | 推荐理由 |
|------|------|----------|------|------|----------|
| 1 | **《大语言模型》** | 赵鑫等（中国人民大学） | 书籍 | ⭐⭐⭐ | 系统讲解LLM训练/微调/推理，含分布式训练与推理优化章节。 |
| 2 | **《Hands-On Large Language Models》** | Jay Alammar, Maarten Grootendorst | 书籍 | ⭐⭐⭐ | 含微调与推理优化实践章节。 |
| 3 | **《高效深度学习：模型压缩与设计》** | 唐洋等 | 书籍 | ⭐⭐⭐⭐ | 系统讲解模型压缩技术（量化/剪枝/蒸馏），推理优化理论基础。 |
| 4 | **《Generative Deep Learning with Python》** | Joseph Babcock | 书籍 | ⭐⭐⭐ | 生成模型实践，含模型优化与部署章节。 |

#### 🎓 大学课程与在线课程

| 序号 | 名称 | 来源 | 类型 | 难度 | 推荐理由 |
|------|------|------|------|------|----------|
| 1 | **DeepLearning.AI: Finetuning Large Language Models** | DeepLearning.AI | 在线课程 | ⭐⭐⭐ | Andrew Ng微调课程，手把手教LoRA微调全流程。https://www.deeplearning.ai/short-courses/finetuning-large-language-models/ |
| 2 | **DeepLearning.AI: Quantization in Depth** | DeepLearning.AI | 在线课程 | ⭐⭐⭐ | 量化技术深度课程，覆盖GPTQ/AWQ等量化方法。 |
| 3 | **DeepLearning.AI: How Business Thinkers Can Start Building AI Plugins with Semantic Kernel** | DeepLearning.AI | 在线课程 | ⭐⭐ | AI应用工程化课程。 |
| 4 | **CMU 10-414/714: Deep Learning Systems** | CMU (Tianqi Chen) | 大学课程 | ⭐⭐⭐⭐ | CMU深度学习系统课程，TVM/TensorFlow作者Tianqi Chen主讲。理解DL系统底层优化。https://dlsyscourse.org/ |
| 5 | **MIT 6.5940: TinyML and Efficient Deep Learning Computing** | MIT (Song Han) | 大学课程 | ⭐⭐⭐⭐ | MIT模型压缩与高效推理课程，Song Han（深度压缩开创者）主讲。覆盖量化/剪枝/蒸馏/神经架构搜索。https://hanlab.mit.edu/courses/2023-fall-65940 |
| 6 | **HuggingFace Course on Optimization** | HuggingFace | 在线课程 | ⭐⭐⭐ | HuggingFace官方优化教程，覆盖量化/Flash Attention/推理加速。https://huggingface.co/learn/optimum/ |

#### 📝 博客与文章

| 序号 | 名称 | 作者/来源 | 类型 | 难度 | 推荐理由 |
|------|------|----------|------|------|----------|
| 1 | **vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention** | Kwon et al. (UC Berkeley) | 论文 | ⭐⭐⭐⭐ | vLLM/PagedAttention原始论文，理解KV Cache高效管理的核心创新。 |
| 2 | **FlashAttention: Fast and Memory-Efficient Exact Attention** | Dao et al. (Stanford) | 论文 | ⭐⭐⭐⭐ | Flash Attention原始论文，IO-aware注意力计算，推理加速基石。 |
| 3 | **GPTQ: Accurate Post-Training Quantization** | Frantar et al. | 论文 | ⭐⭐⭐⭐ | GPTQ量化论文，训练后权重量化方法。 |
| 4 | **AWQ: Activation-aware Weight Quantization** | Lin et al. (MIT) | 论文 | ⭐⭐⭐⭐ | AWQ量化论文，激活感知的权重量化。 |
| 5 | **Speculative Decoding** | Leviathan et al. (Google) | 论文 | ⭐⭐⭐⭐ | 投机解码原始论文，小模型生成+大模型验证的推理加速方法。 |
| 6 | **DeepSpeed: System Optimizations Enable Training Deep Learning Models with Over 100 Billion Parameters** | Microsoft | 论文 | ⭐⭐⭐⭐ | DeepSpeed ZeRO论文，理解分布式训练优化。 |
| 7 | **Megatron-LM: Training Multi-Billion Parameter Language Models** | NVIDIA | 论文 | ⭐⭐⭐⭐ | Megatron-LM论文，张量并行+流水线并行的大模型训练。 |
| 8 | **Sebastian Raschka: Practical Tips for LLM Training/Fine-tuning** | Sebastian Raschka | 博客 | ⭐⭐⭐ | 微调实践技巧系列博客，LoRA/QLoRA深度解析。https://magazine.sebastianraschka.com/ |
| 9 | **Sebastian Raschka: Quantization Series** | Sebastian Raschka | 博客 | ⭐⭐⭐⭐ | 量化技术系列深度博客，对比GPTQ/AWQ/GGUF等方案。 |
| 10 | **vLLM Blog** | vLLM Team | 博客 | ⭐⭐⭐⭐ | vLLM官方博客，PagedAttention/连续批处理/前缀缓存等技术深度讲解。https://blog.vllm.ai/ |
| 11 | **HuggingFace Optimum Documentation** | HuggingFace | 技术文档 | ⭐⭐⭐ | HuggingFace推理优化文档，覆盖ONNX Runtime/TensorRT/量化。https://huggingface.co/docs/optimum/ |
| 12 | **NVIDIA TensorRT-LLM Blog** | NVIDIA | 博客 | ⭐⭐⭐⭐ | TensorRT-LLM技术博客，极致GPU推理优化。https://nvidia.github.io/TensorRT-LLM/ |
| 13 | **EleutherAI Blog** | EleutherAI | 博客 | ⭐⭐⭐⭐ | 开源AI研究组织博客，大模型训练/评估/推理前沿。https://blog.eleuther.ai/ |

#### 🔧 开源项目/代码库

| 序号 | 名称 | 作者/来源 | 类型 | 难度 | 推荐理由 |
|------|------|------|------|------|----------|
| 1 | **vLLM** | UC Berkeley | 开源推理引擎 | ⭐⭐⭐⭐ | 最流行的LLM推理引擎，PagedAttention/连续批处理/张量并行。学习推理优化的首选项目。https://github.com/vllm-project/vllm |
| 2 | **DeepSpeed** | Microsoft | 开源训练框架 | ⭐⭐⭐⭐ | 微软分布式训练框架，ZeRO Stage 1/2/3、混合精度训练。大模型训练必备。https://github.com/microsoft/DeepSpeed |
| 3 | **Megatron-LM** | NVIDIA | 开源训练框架 | ⭐⭐⭐⭐⭐ | NVIDIA大模型训练框架，张量并行/流水线并行/数据并行。理解大模型分布式训练的工业级实现。https://github.com/NVIDIA/Megatron-LM |
| 4 | **PEFT (HuggingFace)** | HuggingFace | 开源库 | ⭐⭐⭐ | 参数高效微调库，LoRA/QLoRA/Adapter等实现。https://github.com/huggingface/peft |
| 5 | **TRL (HuggingFace)** | HuggingFace | 开源库 | ⭐⭐⭐⭐ | RLHF/DPO训练库，SFT/奖励模型/PPO/DPO全流程。https://github.com/huggingface/trl |
| 6 | **Axolotl** | OpenAccess AI Collective | 开源工具 | ⭐⭐⭐ | 高级微调工具，配置驱动，支持多种微调方法。https://github.com/OpenAccess-AI-Collective/axolotl |
| 7 | **Unsloth** | UnslothAI | 开源库 | ⭐⭐⭐ | 超快速LoRA微调，手动梯度优化，2-5倍加速。https://github.com/unslothai/unsloth |
| 8 | **LLaMA-Factory** | hiyouga | 开源工具 | ⭐⭐⭐ | WebUI微调工具，支持多种模型和微调方法。https://github.com/hiyouga/LLaMA-Factory |
| 9 | **AutoGPTQ** | PanQiWei | 开源库 | ⭐⭐⭐⭐ | GPTQ量化实现库，训练后权重量化。https://github.com/PanQiWei/AutoGPTQ |
| 10 | **AutoAWQ** | MIT HAN Lab | 开源库 | ⭐⭐⭐⭐ | AWQ量化实现库，激活感知权重量化。https://github.com/casper-hansen/AutoAWQ |
| 11 | **llama.cpp** | Georgi Gerganov | 开源项目 | ⭐⭐⭐⭐ | C/C++实现的LLM推理引擎，GGUF量化格式，CPU/GPU推理。理解推理引擎底层实现。https://github.com/ggerganov/llama.cpp |
| 12 | **Ollama** | Ollama | 开源工具 | ⭐⭐ | 基于llama.cpp的本地LLM运行工具，简化部署。https://github.com/ollama/ollama |
| 13 | **SGLang** | LMSYS | 开源推理引擎 | ⭐⭐⭐⭐⭐ | 高性能推理引擎，RadixAttention（前缀树缓存）和结构化生成。https://github.com/sgl-project/sglang |
| 14 | **TensorRT-LLM** | NVIDIA | 开源推理引擎 | ⭐⭐⭐⭐⭐ | NVIDIA极致推理优化引擎，理解GPU推理优化的最高水平。https://github.com/NVIDIA/TensorRT-LLM |
| 15 | **Text Generation Inference (TGI)** | HuggingFace | 开源推理引擎 | ⭐⭐⭐⭐ | HuggingFace推理服务器，支持多模型并发/张量并行/Flash Attention。https://github.com/huggingface/text-generation-inference |
| 16 | **FlashAttention** | Tri Dao | 开源库 | ⭐⭐⭐⭐⭐ | IO-aware注意力计算库，推理/训练加速核心。阅读源码理解GPU内存层次优化。https://github.com/Dao-AILab/flash-attention |
| 17 | **OpenCompass** | 上海AI实验室 | 开源工具 | ⭐⭐⭐ | 大模型评测框架，MMLU/HumanEval等基准。https://github.com/open-compass/opencompass |
| 18 | **lm-evaluation-harness** | EleutherAI | 开源工具 | ⭐⭐⭐ | EleutherAI的大模型评测框架。https://github.com/EleutherAI/lm-evaluation-harness |
| 19 | **DeepSpeed-MII** | Microsoft | 开源工具 | ⭐⭐⭐⭐ | DeepSpeed推理优化库，低延迟高吞吐推理。https://github.com/microsoft/DeepSpeed-MII |
| 20 | **bitsandbytes** | Tim Dettmers | 开源库 | ⭐⭐⭐⭐ | 8-bit/4-bit量化库，QLoRA的量化后端。https://github.com/TimDettmers/bitsandbytes |

---

## 5. 学习路径建议（推荐学习顺序与阶段划分）

### 阶段一：基础筑基（第1-3个月）

**目标**：建立深度学习与Transformer的理论基础，理解大模型架构原理。

| 周次 | 学习内容 | 核心资料 | 实践任务 |
|------|----------|----------|----------|
| 1-3 | 神经网络基础（MLP/激活函数/反向传播/优化器/正则化） | 《动手学深度学习》D2L前5章 + Andrew Ng DL课程 | 用PyTorch实现MLP分类MNIST |
| 4-5 | CNN原理与经典架构 | D2L CNN章节 + CS231N前5讲 | 理解卷积操作，实现简单CNN |
| 6-7 | RNN/LSTM/GRU原理 | D2L RNN章节 + CS224N前6讲 | 实现字符级RNN语言模型 |
| 8-10 | **Transformer架构（核心重点）** | The Illustrated Transformer + The Annotated Transformer + D2L Transformer章节 | 从零实现Transformer编码器/解码器 |
| 11-12 | 大模型架构（BERT/GPT/LLaMA） | The Illustrated BERT/GPT + LLaMA论文 + HuggingFace Transformers教程 | 用Transformers库加载并运行BERT/GPT模型 |

**阶段产出**：能清晰解释Transformer的Q/K/V计算、多头注意力、位置编码原理；能对比BERT/GPT/LLaMA架构差异；能用HuggingFace Transformers加载和运行主流大模型。

### 阶段二：应用开发入门（第4-6个月）

**目标**：掌握RAG、Prompt工程、Agent等大模型应用开发核心技术。

| 周次 | 学习内容 | 核心资料 | 实践任务 |
|------|----------|----------|----------|
| 13-14 | Prompt工程基础与高级技巧 | Prompt Engineering Guide + CoT/ToT/Self-Consistency论文 | 设计并测试不同Prompt策略的效果对比 |
| 15-16 | RAG基础流程 | LlamaIndex文档 + RAG原始论文 | 用LlamaIndex构建基础RAG问答系统 |
| 17-18 | RAG高级技术（混合检索/重排序/上下文优化） | LlamaIndex Blog + RAG Survey论文 | 升级RAG系统：加入BM25+向量混合检索和Cross-Encoder重排序 |
| 19-20 | Agent基础（ReAct/工具调用） | ReAct论文 + Lilian Weng Agent博客 + LangChain Agent教程 | 用LangChain实现ReAct Agent |
| 21-22 | 多Agent协作 | AutoGen/LangGraph文档 + Multi-Agent Survey | 用LangGraph实现多Agent协作系统 |
| 23-24 | 端到端AI应用开发 | LangChain + Dify + OpenAI Cookbook | 构建一个完整的知识库问答应用（含文档解析/向量检索/Prompt管理/流式输出） |

**阶段产出**：能独立设计并实现RAG问答系统（含混合检索+重排序）；能设计ReAct/Plan-Execute Agent；能解决长文本处理/多轮对话/输出格式控制等常见开发问题。

### 阶段三：微调与推理优化（第7-9个月）

**目标**：掌握大模型微调全流程与推理性能优化技术。

| 周次 | 学习内容 | 核心资料 | 实践任务 |
|------|----------|----------|----------|
| 25-26 | 微调基础（全量微调/PEFT/LoRA原理） | LoRA论文 + PEFT文档 + DeepLearning.AI微调课程 | 用PEFT库对7B模型进行LoRA微调 |
| 27-28 | QLoRA与高效微调实践 | QLoRA论文 + Unsloth/LLaMA-Factory | 用QLoRA在消费级GPU上微调7B模型 |
| 29-30 | RLHF/DPO对齐 | InstructGPT论文 + DPO论文 + TRL文档 | 用TRL库实现DPO微调 |
| 31-32 | 分布式训练（DDP/FSDP/DeepSpeed） | DeepSpeed文档 + Megatron-LM论文 | 用DeepSpeed ZeRO进行多卡微调 |
| 33-34 | 推理引擎（vLLM/TGI） | vLLM论文 + vLLM文档 | 用vLLM部署模型推理服务 |
| 35-36 | 量化与推理优化 | GPTQ/AWQ论文 + Flash Attention论文 + Sebastian Raschka量化博客 | 对模型进行GPTQ/AWQ量化并对比推理性能 |
| 37-38 | 高级推理优化（投机解码/前缀缓存/动态批处理） | Speculative Decoding论文 + SGLang文档 + vLLM Blog | 配置并测试vLLM的连续批处理和前缀缓存 |
| 39-40 | 评估体系搭建 | OpenCompass + RAGAS + lm-evaluation-harness | 搭建微调效果与推理性能评估Pipeline |

**阶段产出**：能独立完成微调全流程（数据准备→训练配置→分布式训练→效果评估）；能用vLLM部署高性能推理服务；能实施量化/投机解码等推理优化；能搭建评估体系。

### 阶段四：架构设计与工程化（第10-12个月）

**目标**：掌握AI应用架构设计、性能调优、高可用与平台化建设。

| 周次 | 学习内容 | 核心资料 | 实践任务 |
|------|----------|----------|----------|
| 41-42 | AI应用架构设计 | 《Designing ML Systems》+ 《AI Engineering》+ Chip Huyen Blog | 设计一个企业级AI应用架构方案 |
| 43-44 | 性能调优与高可用 | FSDL课程 + OpenAI Cookbook | 对已有应用进行延迟优化/并发提升/降级策略设计 |
| 45-46 | 复杂Agent系统设计 | LangGraph + AutoGen + MetaGPT源码 | 设计并实现多Agent协作系统（如代码审查Agent/研究Agent） |
| 47-48 | 企业级平台建设 | Dify/Bisheng源码研读 | 设计模型管理/Prompt管理/应用编排平台方案 |

**阶段产出**：能主导AI应用项目的技术选型与架构决策；能设计高性能、高可用、低成本的AI应用方案；能设计复杂Agent系统和企业级AI平台。

### 阶段五：前沿追踪与面试冲刺（第13-14个月）

**目标**：追踪前沿技术，查漏补缺，综合复习。

| 周次 | 学习内容 | 核心资料 | 实践任务 |
|------|----------|----------|----------|
| 49-50 | 前沿技术追踪 | Lilian Weng Blog + Sebastian Raschka Blog + 最新论文（MoE/长上下文/多模态） | 阅读并总结3-5篇前沿论文 |
| 51-52 | 综合复习 | 本学习计划全部内容 + 岗位岗位能力要求 | 模拟自测：对照4个评估维度逐项自测 |
| 53-54 | 项目复盘与作品集 | 前述阶段所有实践项目 | 整理项目作品集，准备技术深度讲解 |
| 55-56 | 补短板与强化 | 根据自测结果针对性补强 | 针对薄弱环节进行专项学习 |

### 学习路径总结图

```
阶段一（1-3月）          阶段二（4-6月）          阶段三（7-9月）          阶段四（10-12月）       阶段五（13-14月）
基础筑基                应用开发入门             微调与推理优化            架构设计与工程化        前沿追踪与冲刺
┌─────────────┐      ┌─────────────┐       ┌─────────────┐       ┌─────────────┐      ┌─────────────┐
│ 神经网络基础  │      │ Prompt工程  │       │ LoRA/QLoRA  │       │ 应用架构设计 │      │ 前沿论文追踪 │
│ CNN/RNN/GNN │ ──→  │ RAG系统开发  │ ──→  │ RLHF/DPO   │ ──→  │ 性能调优    │ ──→  │ 面试自测    │
│ Transformer │      │ Agent系统   │       │ 分布式训练   │       │ Agent系统   │      │ 作品集整理  │
│ 大模型架构   │      │ 端到端应用   │       │ 推理引擎    │       │ 平台建设    │      │ 补短板强化  │
└─────────────┘      └─────────────┘       │ 量化优化    │       └─────────────┘      └─────────────┘
                                           │ 评估体系    │
                                           └─────────────┘
```

### 关键里程碑检查清单

- [ ] **里程碑1**（第3月末）：能用白板画出Transformer架构图并解释每个组件的计算过程
- [ ] **里程碑2**（第6月末）：能独立完成一个RAG问答系统的端到端开发并上线运行
- [ ] **里程碑3**（第9月末）：能完成LoRA微调全流程并用vLLM部署优化后的推理服务
- [ ] **里程碑4**（第12月末）：能设计企业级AI应用架构方案并通过技术评审
- [ ] **里程碑5**（第14月末）：对照岗位能力要求4个维度，每个维度达到精通级以上水平

---

> **备注**：本学习计划总计推荐 **书籍15+本**、**大学课程10+门**、**博客/论文30+篇**、**开源项目30+个**，覆盖从基础理论到前沿工程的全链路知识。建议按照阶段顺序系统学习，每个阶段完成后对照里程碑检查清单进行自评。所有开源项目均建议实际动手运行和阅读源码，理论结合实践是掌握AI技术应用能力的唯一途径。
