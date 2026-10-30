## 目录

- [[#一、岗位概述]]
- [[#二、知识体系全景图（知识树）]]
- [[#三、分领域详细知识点梳理与学习资料规划]]
  - [[#领域一：模型轻量化技术]]
  - [[#领域二：端侧模型推理与部署优化]]
  - [[#领域三：Transformer与LLM架构基础]]
  - [[#领域四：模型训练基础（预训练/SFT/对齐）]]
  - [[#领域五：数据清洗与合成]]
  - [[#领域六：端侧高效模型架构设计]]
  - [[#领域七：模型部署与工程调试]]
  - [[#领域八：实验分析与问题定位]]
- [[#四、学习路径建议（阶段划分与推荐顺序）]]
- [[#五、综合实战项目建议]]
- [[#六、推荐学习资料总表]]

---

## 一、岗位概述

### 1.1 岗位定位

"AI模型-端侧大模型"岗位聚焦于在**资源受限的端侧设备**（手机、PC、IoT、车载等）上部署和运行大语言模型（LLM）。与云端大模型不同，端侧场景面临严格的**算力限制、内存限制、功耗限制和延迟要求**，因此需要一套完整的技术栈来在保证模型效果的前提下，实现模型的小型化、高效推理和场景适配。

### 1.2 核心能力要求

| 能力维度 | 核心要求 |
|----------|----------|
| 模型轻量化 | 掌握蒸馏、量化、剪枝、SFT微调等模型压缩与适配方法 |
| 推理优化 | 理解Prefill/Decoding/KV Cache/量化等推理加速技术 |
| 架构基础 | 深入理解Transformer/Attention/LLM结构与端侧约束的关系 |
| 训练基础 | 掌握预训练、SFT、DPO/PPO等对齐方法的原理与实践 |
| 数据工程 | 具备端侧场景数据清洗、合成、质量评估能力 |
| 工程部署 | 能完成模型转换、部署验证、性能调优全链路 |
| 实验分析 | 能定位badcase、分析性能瓶颈、推动问题解决 |

### 1.3 岗位技术栈全景

```
端侧大模型技术栈
├── 模型层
│   ├── Transformer架构理解
│   ├── 高效架构设计（小模型/高效Attention/量化友好结构）
│   └── 多模态模块适配
├── 训练层
│   ├── 预训练（Pre-training）
│   ├── 监督微调（SFT）
│   ├── 偏好对齐（DPO/PPO/Reward Model）
│   ├── 知识蒸馏（Distillation）
│   └── 参数高效微调（LoRA/QLoRA/Adapter）
├── 数据层
│   ├── 数据清洗（去重/过滤/质量评估）
│   ├── 指令数据合成
│   ├── 端侧场景数据构造
│   └── 数据质量与覆盖率分析
├── 推理层
│   ├── 推理引擎（Prefill/Decoding/KV Cache）
│   ├── 量化技术（PTQ/QAT/GPTQ/AWQ/INT4/INT8）
│   ├── 推理框架（llama.cpp/MNN/NCNN/TFLite）
│   └── 性能优化（首字延迟/生成速度/内存/功耗）
├── 部署层
│   ├── 模型转换（PyTorch→ONNX→端侧格式）
│   ├── 端侧部署验证
│   ├── 精度一致性校验
│   └── 工程调试与问题定位
└── 评估层
    ├── 模型效果评估
    ├── 性能基准测试
    ├── Badcase分析与优化
    └── 实验报告与协作交付
```

---

## 二、知识体系全景图（知识树）

```
端侧大模型知识体系
│
├── 1. Transformer与LLM架构基础
│   ├── 1.1 Token与Tokenization
│   │   ├── BPE / WordPiece / SentencePiece
│   │   ├── 词表大小对端侧的影响
│   │   └── 多语言Tokenization
│   ├── 1.2 Embedding
│   │   ├── 词嵌入原理
│   │   ├── 位置编码（绝对/相对/RoPE/ALiBi）
│   │   └── Embedding层参数量分析
│   ├── 1.3 Attention机制
│   │   ├── Self-Attention计算流程
│   │   ├── Multi-Head Attention
│   │   ├── 注意力复杂度分析（O(n²d)）
│   │   └── KV Cache原理与内存占用
│   ├── 1.4 Transformer Block
│   │   ├── FFN层结构与参数占比
│   │   ├── LayerNorm / RMSNorm
│   │   ├── 残差连接
│   │   └── 前向传播计算量分析
│   ├── 1.5 LLM整体架构
│   │   ├── Decoder-only架构（GPT系列）
│   │   ├── Encoder-Decoder架构（T5）
│   │   ├── 主流开源LLM架构对比（LLaMA/Qwen/Phi等）
│   │   └── 架构参数量/计算量/显存占用关系
│   └── 1.6 LLM训练与推理的关系
│       ├── 训练阶段vs推理阶段的计算差异
│       ├── KV Cache在推理中的作用
│       └── 结构-显存-速度的三角关系
│
├── 2. 模型训练基础
│   ├── 2.1 预训练（Pre-training）
│   │   ├── 语言建模目标（CLM/MLM）
│   │   ├── 预训练数据规模与质量
│   │   ├── 训练超参数（学习率/Batch Size/Warmup）
│   │   └── Scaling Law
│   ├── 2.2 监督微调（SFT）
│   │   ├── SFT数据格式（instruction/input/output）
│   │   ├── SFT训练目标与损失函数
│   │   ├── 全参数微调 vs 参数高效微调
│   │   └── SFT常见问题（过拟合/灾难性遗忘）
│   ├── 2.3 参数高效微调（PEFT）
│   │   ├── LoRA原理与实现
│   │   ├── QLoRA（量化+LoRA）
│   │   ├── Adapter / Prefix Tuning / Prompt Tuning
│   │   └── LoRA在端侧模型适配中的应用
│   ├── 2.4 偏好对齐
│   │   ├── RLHF流程（SFT→RM→PPO）
│   │   ├── Reward Model训练
│   │   ├── PPO算法原理
│   │   ├── DPO（Direct Preference Optimization）
│   │   ├── 其他对齐方法（KTO/IPO/SimPO等）
│   │   └── 对齐在端侧场景中的特殊考虑
│   └── 2.5 端侧训练的特殊性
│       ├── 端侧训练与通用大模型训练的差异
│       ├── 小模型效果保障策略
│       ├── 场景适配与领域迁移
│       └── 量化友好训练（QAT）
│
├── 3. 模型轻量化技术
│   ├── 3.1 知识蒸馏
│   │   ├── 蒸馏原理（Teacher-Student）
│   │   ├── 输出层蒸馏（Logit蒸馏）
│   │   ├── 中间层蒸馏（Feature蒸馏）
│   │   ├── 蒸馏在端侧小模型中的应用
│   │   └── 在线蒸馏 vs 离线蒸馏
│   ├── 3.2 量化
│   │   ├── 量化基本原理
│   │   ├── 训练后量化（PTQ）
│   │   ├── 量化感知训练（QAT）
│   │   ├── 量化精度格式（FP16/INT8/INT4/INT2）
│   │   ├── 主流量化算法（GPTQ/AWQ/SmoothQuant/GGUF）
│   │   └── 量化对模型效果的影响与补偿
│   ├── 3.3 剪枝
│   │   ├── 结构化剪枝 vs 非结构化剪枝
│   │   ├── 剪枝标准（幅度/梯度/注意力重要性）
│   │   └── 剪枝后微调恢复
│   ├── 3.4 低秩分解
│   │   ├── SVD分解
│   │   ├── 张量分解
│   │   └── 在Embedding/FFN中的应用
│   ├── 3.5 小模型设计
│   │   ├── 小模型架构选择（层数/隐藏维度/头数）
│   │   ├── 参数量与效果的关系
│   │   ├── 端侧模型规格设计（<1B/1-3B/3-7B）
│   │   └── 蒸馏+微调联合优化策略
│   └── 3.6 量化友好训练
│       ├── 量化感知训练（QAT）原理
│       ├── 量化噪声模拟
│       ├── Straight-Through Estimator（STE）
│       └── 量化友好模型结构设计
│
├── 4. 端侧模型推理与部署优化
│   ├── 4.1 推理过程详解
│   │   ├── Prefill阶段（首Token生成）
│   │   ├── Decoding阶段（自回归生成）
│   │   ├── KV Cache机制与内存计算
│   │   └── Prefill vs Decoding的计算特征差异
│   ├── 4.2 推理性能指标
│   │   ├── 首字延迟（TTFT - Time To First Token）
│   │   ├── 生成速度（Tokens/s）
│   │   ├── 内存占用（模型权重+KV Cache+激活值）
│   │   ├── 功耗（Watts）
│   │   └── 各指标的影响因素分析
│   ├── 4.3 推理加速技术
│   │   ├── KV Cache优化（PagedAttention/量化KV Cache）
│   │   ├── Attention优化（Flash Attention/Flash Decoding）
│   │   ├── 投机解码（Speculative Decoding）
│   │   ├── 连续批处理（Continuous Batching）
│   │   └── 前缀缓存（Prefix Caching）
│   ├── 4.4 端侧推理框架
│   │   ├── llama.cpp（CPU/GPU推理）
│   │   ├── MNN（阿里端侧推理引擎）
│   │   ├── NCNN（腾讯端侧推理）
│   │   ├── TFLite（Google移动端推理）
│   │   ├── ONNX Runtime Mobile
│   │   ├── MLC-LLM（通用机器学习编译）
│   │   └── 各框架对比与选择
│   ├── 4.5 芯片与硬件适配
│   │   ├── NPU/GPU/CPU推理差异
│   │   ├── 主流端侧芯片（高通骁龙/苹果A/M/联发科/麒麟）
│   │   ├── 芯片算力与内存带宽分析
│   │   └── 硬件感知的推理优化
│   └── 4.6 端侧推理性能瓶颈分析
│       ├── 计算密集型 vs 内存密集型分析
│       ├── Roofline模型
│       ├── 瓶颈定位方法
│       └── 结合芯片特点的优化策略
│
├── 5. 数据清洗与合成
│   ├── 5.1 数据清洗基础
│   │   ├── 去重（精确去重/模糊去重/MinHash）
│   │   ├── 过滤（规则过滤/质量过滤/安全过滤）
│   │   ├── 数据格式标准化
│   │   └── 数据质量评估指标
│   ├── 5.2 指令数据构造
│   │   ├── 指令数据格式（instruction-input-output）
│   │   ├── 人工标注 vs 模型生成
│   │   ├── Self-Instruct方法
│   │   ├── Evol-Instruct方法
│   │   └── 高质量指令数据集参考
│   ├── 5.3 端侧场景数据构造
│   │   ├── 问答场景数据构造
│   │   ├── 工具调用场景数据
│   │   ├── 设备控制场景数据
│   │   ├── 多轮对话数据构造
│   │   └── 端侧隐私保护与数据脱敏
│   ├── 5.4 数据合成技术
│   │   ├── LLM辅助数据合成
│   │   ├── 数据增强（回译/改写/扰动）
│   │   ├── 蒸馏数据生成
│   │   └── 合成数据质量评估
│   ├── 5.5 数据质量原则
│   │   ├── 高质量（准确性/流畅性/有用性）
│   │   ├── 多样性（指令类型/任务类型/语言风格）
│   │   ├── 覆盖率（场景覆盖/能力覆盖/难度覆盖）
│   │   └── 噪声控制（错误率/偏差控制）
│   └── 5.6 数据工程Pipeline
│       ├── 数据采集→清洗→标注→合成→评估→发布
│       ├── 自动化数据流水线搭建
│       ├── 数据版本管理
│       └── 数据驱动的模型迭代闭环
│
├── 6. 端侧高效模型架构设计
│   ├── 6.1 模型规模设计
│   │   ├── 参数量选择（<1B/1-3B/3-7B）
│   │   ├── 层数与隐藏维度的权衡
│   │   ├── Attention头数设计
│   │   └── 词表大小选择
│   ├── 6.2 高效Attention机制
│   │   ├── Multi-Query Attention（MQA）
│   │   ├── Grouped-Query Attention（GQA）
│   │   ├── Sliding Window Attention
│   │   ├── Linear Attention
│   │   └── KV Cache占用优化
│   ├── 6.3 高效FFN设计
│   │   ├── GLU变体（SwiGLU等）
│   │   ├── FFN中间维度比例
│   │   └── 稀疏FFN
│   ├── 6.4 上下文长度优化
│   │   ├── 长上下文的KV Cache内存问题
│   │   ├── 上下文窗口压缩
│   │   ├── 滑动窗口机制
│   │   └── 端侧合理上下文长度选择
│   ├── 6.5 量化友好架构
│   │   ├── 量化敏感层识别
│   │   ├── 量化友好激活函数
│   │   ├── 量化友好归一化层
│   │   └── 结构设计对量化精度的影响
│   ├── 6.6 多模态模块适配
│   │   ├── 视觉编码器轻量化
│   │   ├── 模态对齐模块优化
│   │   └── 端侧多模态推理约束
│   └── 6.7 架构效果-成本权衡
│       ├── 参数量 vs 效果曲线
│       ├── 计算量（FLOPs） vs 延迟
│       ├── KV Cache内存 vs 上下文长度
│       └── 端侧可部署性综合评估
│
├── 7. 模型部署与工程调试
│   ├── 7.1 模型转换
│   │   ├── PyTorch → ONNX
│   │   ├── ONNX → 端侧格式（MNN/NCNN/TFLite）
│   │   ├── GGUF格式转换
│   │   ├── 量化模型导出
│   │   └── 转换常见问题与解决
│   ├── 7.2 部署验证
│   │   ├── 推理正确性验证
│   │   ├── 精度一致性校验（端侧vs云端）
│   │   ├── 性能基准测试
│   │   └── 内存/功耗测量
│   ├── 7.3 常见部署问题
│   │   ├── 格式转换失败（不支持的算子/结构）
│   │   ├── 精度不一致（量化误差/算子实现差异）
│   │   ├── 推理慢（未利用硬件加速/内存瓶颈）
│   │   ├── 内存溢出（模型过大/KV Cache过大）
│   │   └── 崩溃与稳定性问题
│   ├── 7.4 工程调试技能
│   │   ├── 性能Profile工具使用
│   │   ├── 内存分析工具
│   │   ├── 日志分析与问题定位
│   │   └── A/B测试与对比实验
│   └── 7.5 端侧部署全链路
│       ├── 模型选型→压缩→转换→部署→验证→调优
│       ├── CI/CD for端侧模型
│       └── 模型版本管理与灰度发布
│
└── 8. 实验分析与问题定位
    ├── 8.1 实验设计
    │   ├── 实验变量控制
    │   ├── 评估指标选择
    │   ├── 基线设置与对比实验
    │   └── 消融实验设计
    ├── 8.2 Badcase分析
    │   ├── 回复不准（知识缺失/推理错误）
    │   ├── 回复不稳定（温度/采样策略/上下文敏感）
    │   ├── 安全性问题（有害内容/隐私泄露）
    │   ├── 量化后效果下降（精度损失分析）
    │   └── Badcase归因与修复策略
    ├── 8.3 性能问题定位
    │   ├── 训练不稳定（Loss爆炸/梯度消失）
    │   ├── 推理变慢（KV Cache膨胀/内存碎片）
    │   ├── 效果下降（数据质量/训练策略/量化影响）
    │   └── 系统性排查方法论
    ├── 8.4 实验报告与协作
    │   ├── 实验结果整理与可视化
    │   ├── 技术报告撰写
    │   ├── 与工程团队协作推动
    │   └── 与硬件团队的需求沟通
    └── 8.5 持续迭代
        ├── 数据驱动的模型迭代闭环
        ├── 效果-性能-成本的持续权衡
        └── 版本迭代与回归测试
```

---

## 三、分领域详细知识点梳理与学习资料规划

---

### 领域一：模型轻量化技术

> 对应：专业知识-知识点1（模型轻量化技术基础）、关键技能-技能项1（端侧约束下的模型训练与适配优化能力）

#### 1.1 知识点详细梳理

**1.1.1 知识蒸馏（Knowledge Distillation）**

| 子知识点 | 详细内容 |
|----------|----------|
| 蒸馏基本原理 | Teacher-Student范式：大模型（Teacher）的软标签或中间特征指导小模型（Student）学习 |
| 输出层蒸馏（Logit蒸馏） | KL散度损失：L = α·CE(y, p_student) + (1-α)·T²·KL(p_teacher^T, p_student^T)，温度系数T的作用 |
| 中间层蒸馏（Feature蒸馏） | 对齐Teacher和Student的隐藏层表示，使用MSE或Cosine Embedding Loss |
| 注意力蒸馏 | 传递Teacher的Attention分布，帮助Student学习对齐模式 |
| 在线蒸馏 vs 离线蒸馏 | 离线：Teacher预训练好后指导Student；在线：Teacher和Student同时训练（如Deep Mutual Learning） |
| 蒸馏在LLM中的应用 | 大模型→小模型蒸馏（如LLaMA-70B→7B）、蒸馏数据生成（用大模型生成训练数据） |
| 端侧蒸馏策略 | 针对端侧任务蒸馏、多Teacher蒸馏、蒸馏+量化联合优化 |

**1.1.2 量化（Quantization）**

| 子知识点 | 详细内容 |
|----------|----------|
| 量化基本原理 | 将FP32/FP16权重映射到低精度（INT8/INT4/INT2），减少存储和计算开销 |
| 对称量化 vs 非对称量化 | 对称：量化范围以0为中心；非对称：使用零点偏移 |
| 训练后量化（PTQ） | 不需要重新训练，直接对已训练模型量化。简单但可能有精度损失 |
| 量化感知训练（QAT） | 训练过程中模拟量化噪声，让模型适应低精度。精度更好但需要训练资源 |
| 量化精度格式 | FP16（半精度）、BF16（Brain Float）、INT8（8位整数）、INT4（4位整数）、INT2（二值化） |
| GPTQ | 基于二阶Hessian信息的逐层量化方法，适用于INT4量化 |
| AWQ（Activation-aware Weight Quantization） | 基于激活值重要性感知的权重量化方法，保护重要通道 |
| SmoothQuant | 将激活值的难度迁移到权重上，实现平滑的INT8量化 |
| GGUF格式量化 | llama.cpp使用的量化格式（Q4_0/Q4_1/Q5_0/Q5_1/Q8_0等） |
| 量化对效果的影响 | INT8通常损失<1%，INT4可能损失1-5%，INT2损失较大需要QAT补偿 |
| 量化补偿技术 | 量化感知微调、混合精度量化（敏感层保持高精度） |

**1.1.3 剪枝（Pruning）**

| 子知识点 | 详细内容 |
|----------|----------|
| 结构化剪枝 | 剪除整个通道/层/注意力头，硬件友好，可直接加速 |
| 非结构化剪枝 | 剪除单个权重（置零），压缩率高但需要稀疏矩阵支持 |
| 剪枝标准 | 权重幅度（L1/L2 norm）、梯度幅度、Taylor展开重要性、注意力头重要性 |
| 剪枝流程 | 评估重要性→剪枝→微调恢复→迭代剪枝 |
| LLM剪枝挑战 | 大模型剪枝后恢复困难、结构化剪枝可能破坏预训练知识 |

**1.1.4 参数高效微调（PEFT）**

| 子知识点 | 详细内容 |
|----------|----------|
| LoRA（Low-Rank Adaptation） | 冻结原始权重，注入低秩矩阵A和B（W' = W + BA），大幅减少可训练参数 |
| LoRA超参数 | 秩r（通常8-64）、缩放因子α、目标模块选择（q_proj/v_proj/all） |
| QLoRA | 4-bit量化基座模型 + LoRA微调，极大降低显存需求 |
| Adapter | 在Transformer层中插入小型瓶颈网络 |
| Prefix Tuning | 在输入前添加可学习的前缀向量 |
| Prompt Tuning | 仅优化连续的软提示向量 |
| LoRA在端侧的应用 | 端侧场景适配微调、多任务LoRA切换、LoRA权重合并 |

**1.1.5 小模型设计与蒸馏联合策略**

| 子知识点 | 详细内容 |
|----------|----------|
| 端侧模型规格 | <1B（超轻量，功能受限）、1-3B（平衡型，主流端侧）、3-7B（高性能端侧） |
| 蒸馏+微调联合 | 先蒸馏获取基础能力→SFT适配端侧场景→量化部署 |
| 数据蒸馏 | 用大模型生成高质量指令数据，用于小模型SFT |
| 多阶段蒸馏 | 通用能力蒸馏→领域能力蒸馏→场景任务蒸馏 |

#### 1.2 学习资料规划

##### 书籍

| 序号 | 书名 | 作者/来源 | 难度 | 推荐理由 |
|------|------|-----------|------|----------|
| 1 | **《Deep Learning》** | Ian Goodfellow, Yoshua Bengio, Aaron Courville | 中级 | 深度学习圣经，第6章深度前馈网络、第8章优化，理解模型压缩的理论基础 |
| 2 | **《Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow》第3版** | Aurélien Géron | 入门-中级 | 第11章训练深度神经网络，包含量化、剪枝的实践代码 |
| 3 | **《TinyML: Machine Learning with TensorFlow Lite on Arduino and Ultra-Low-Power Microcontrollers》** | Pete Warden, Daniel Situnayake | 入门-中级 | 端侧机器学习经典，理解资源受限场景下的模型部署 |
| 4 | **《Neural Network Quantization with PyTorch》**（在线书） | PyTorch官方 | 中级 | PyTorch量化官方指南，涵盖PTQ/QAT/动态量化/静态量化 |

##### 大学课程

| 序号 | 课程名 | 来源 | 难度 | 推荐理由 |
|------|--------|------|------|----------|
| 1 | **CS231n: Convolutional Neural Networks for Visual Recognition** | Stanford University | 中级 | 虽然以CV为主，但其关于模型压缩、知识蒸馏的lecture是经典入门 |
| 2 | **MIT 6.5940: TinyML and Efficient Deep Learning Computing** | MIT (Song Han教授) | 中-高级 | **核心推荐**：MIT Song Han教授的端侧高效深度学习课程，系统覆盖量化、剪枝、蒸馏、NAS、端侧部署全链路 |
| 3 | **EE367/CS328: Efficient Deep Learning** | Stanford | 中-高级 | 高效深度学习课程，覆盖模型压缩、高效推理、硬件感知优化 |
| 4 | **CMU 11-785: Deep Learning** | CMU (Bhiksha Raj) | 中级 | 深度学习系统课程，包含模型压缩专题 |

##### 博客与技术文章

| 序号 | 文章名 | 来源 | 难度 | 推荐理由 |
|------|--------|------|------|----------|
| 1 | **A Survey on Model Compression for Natural Language Processing** | arXiv (2022) | 中级 | NLP模型压缩综述，涵盖蒸馏/剪枝/量化/PEFT全景 |
| 2 | **AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration** | MIT Song Han团队 | 高级 | AWQ量化算法原始论文，端侧LLM量化核心方法 |
| 3 | **GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers** | ICLR 2023 | 高级 | GPTQ量化算法论文，INT4量化的里程碑工作 |
| 4 | **SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models** | MIT | 高级 | SmoothQuant论文，解决激活值量化难题 |
| 5 | **QLoRA: Efficient Finetuning of Quantized LLMs** | Tim Dettmers et al. | 高级 | QLoRA论文，端侧/低资源场景微调的核心方法 |
| 6 | **Hugging Face PEFT Documentation** | Hugging Face | 中级 | PEFT库官方文档，LoRA/Adapter/Prefix Tuning的实践指南 |
| 7 | **A Comprehensive Survey on Model Quantization for Deep Neural Networks** | arXiv | 中级 | 量化技术全面综述，理解量化原理与发展脉络 |
| 8 | **llama.cpp 量化格式文档** | GitHub llama.cpp | 中级 | GGUF量化格式详解，端侧LLM部署必读 |

##### 开源项目/代码库

| 序号 | 项目名 | GitHub地址 | 难度 | 推荐理由 |
|------|--------|-----------|------|----------|
| 1 | **AutoGPTQ** | github.com/PanQiWei/AutoGPTQ | 中级 | GPTQ量化自动化工具，支持主流LLM的INT4量化 |
| 2 | **AutoAWQ** | github.com/casper-hansen/AutoAWQ | 中级 | AWQ量化自动化实现 |
| 3 | **PEFT (Hugging Face)** | github.com/huggingface/peft | 中级 | LoRA/QLoRA/Adapter等PEFT方法官方实现 |
| 4 | **bitsandbytes** | github.com/TimDettmers/bitsandbytes | 中级 | 8-bit/4-bit量化库，QLoRA的底层实现 |
| 5 | **LLM-Pruner** | github.com/horseee/LLM-Pruner | 高级 | LLM结构化剪枝工具 |
| 6 | **TinyFusion** | GitHub | 高级 | 轻量模型蒸馏框架参考 |

---

### 领域二：端侧模型推理与部署优化

> 对应：专业知识-知识点2（端侧模型推理与部署优化技术）、关键技能-技能项4（模型部署与工程调试能力）

#### 2.1 知识点详细梳理

**2.1.1 LLM推理过程详解**

| 子知识点 | 详细内容 |
|----------|----------|
| Prefill阶段 | 处理输入prompt，计算所有输入token的KV Cache。计算密集型，可并行处理 |
| Decoding阶段 | 逐token自回归生成，每步利用KV Cache避免重复计算。内存密集型，难以并行 |
| KV Cache机制 | 缓存历史token的Key/Value矩阵，避免重复计算Attention。内存占用 = 2 × n_layers × n_heads × seq_len × head_dim × dtype_size |
| KV Cache内存计算 | 以7B模型为例：32层 × 32头 × 128维 × 2(K&V) × 2字节(FP16) × seq_len。2048上下文约需1GB |
| Prefill vs Decoding差异 | Prefill：计算密集，GPU/NPU利用率高；Decoding：内存密集，受带宽限制 |

**2.1.2 推理性能指标**

| 指标 | 含义 | 影响因素 |
|------|------|----------|
| TTFT（首字延迟） | 从输入到第一个token输出的时间 | Prefill计算速度、模型大小、prompt长度 |
| TPS（生成速度） | 每秒生成token数 | Decoding速度、KV Cache读取速度、内存带宽 |
| 内存占用 | 推理时总内存 | 模型权重 + KV Cache + 激活值 + 框架开销 |
| 功耗 | 推理时芯片功耗 | 计算量、内存访问频率、芯片能效比 |
| 峰值内存 | 推理过程中最大内存 | KV Cache增长、中间激活值 |

**2.1.3 推理加速技术**

| 子知识点 | 详细内容 |
|----------|----------|
| Flash Attention | 通过分块计算减少HBM访问，加速Attention计算。Flash Attention v2进一步优化 |
| Flash Decoding | 针对Decoding阶段的Flash Attention优化，拆分KV Cache并行计算 |
| PagedAttention | 借鉴OS虚拟内存管理，分页管理KV Cache，减少内存碎片 |
| 量化KV Cache | 将KV Cache从FP16量化到INT8/INT4，减少内存占用 |
| 投机解码（Speculative Decoding） | 用小模型快速生成草稿，大模型验证。减少大模型的前向次数 |
| 前缀缓存（Prefix Caching） | 缓存系统提示等公共前缀的KV Cache，多请求复用 |
| 连续批处理（Continuous Batching） | 动态组装batch，不同请求不同长度，提高吞吐量 |
| 混合精度推理 | 计算密集部分用高精度，内存密集部分用低精度 |

**2.1.4 端侧推理框架**

| 框架 | 特点 | 适用场景 |
|------|------|----------|
| llama.cpp | C++实现，支持CPU/GPU，GGUF量化，轻量高效 | 通用端侧LLM推理首选 |
| MLC-LLM | 基于TVM/Unity编译器，跨平台部署 | iOS/Android/Web/嵌入式 |
| MNN | 阿里开源端侧推理引擎，支持LLM | Android端侧 |
| NCNN | 腾讯开源端侧推理框架 | Android/iOS |
| TFLite | Google移动端推理框架 | Android端侧 |
| ONNX Runtime Mobile | 微软跨平台推理引擎 | 跨平台 |
| ExecuTorch | PyTorch官方端侧推理框架 | 跨平台，PyTorch生态 |
| PowerInfer | 上海交大开发，CPU+GPU混合推理 | PC端侧 |

**2.1.5 芯片与硬件适配**

| 子知识点 | 详细内容 |
|----------|----------|
| NPU推理 | 神经网络专用处理器，高能效比，需要专用算子支持 |
| GPU推理 | 利用移动GPU并行计算，需要OpenCL/Metal/Vulkan支持 |
| CPU推理 | 通用性最强，SIMD指令优化（ARM NEON/x86 AVX） |
| 主流端侧芯片 | 高通骁龙（Hexagon NPU）、苹果A/M系列（Neural Engine）、联发科天玑（APU）、麒麟（NPU） |
| 算力与带宽 | 算力（TOPS）决定计算密集型操作速度，带宽（GB/s）决定内存密集型操作速度 |
| Roofline模型 | 分析芯片的算力天花板和带宽天花板，判断操作是计算密集还是内存密集 |

#### 2.2 学习资料规划

##### 书籍

| 序号 | 书名 | 作者/来源 | 难度 | 推荐理由 |
|------|------|-----------|------|----------|
| 1 | **《深度学习系统：核心算法与实现》**（Deep Learning Systems） | Tianqi Chen等 | 中-高级 | 深度学习系统设计原理，理解推理引擎底层机制 |
| 2 | **《AI推理系统》**（在线讲义） | CMU 10-418/10-618 | 中-高级 | CMU AI推理系统课程讲义，覆盖推理优化全链路 |
| 3 | **《Computer Architecture: A Quantitative Approach》** | John L. Hennessy, David A. Patterson | 高级 | 计算机体系结构经典，理解硬件性能瓶颈分析（Roofline模型来源） |

##### 大学课程

| 序号 | 课程名 | 来源 | 难度 | 推荐理由 |
|------|--------|------|------|----------|
| 1 | **MIT 6.5940: TinyML and Efficient Deep Learning Computing** | MIT (Song Han) | 中-高级 | **核心推荐**：Lecture覆盖高效推理、量化、硬件协同设计 |
| 2 | **CMU 10-418: Machine Learning Systems** | CMU | 中-高级 | ML系统课程，涵盖推理引擎、模型服务、性能优化 |
| 3 | **Stanford CS149: Parallel Computing** | Stanford | 高级 | 并行计算课程，理解GPU/NPU并行计算原理 |
| 4 | **MIT 6.S977: Performance Engineering of Software Systems** | MIT | 中-高级 | 软件性能工程，学习性能Profile和瓶颈分析方法 |

##### 博客与技术文章

| 序号 | 文章名 | 来源 | 难度 | 推荐理由 |
|------|--------|------|------|----------|
| 1 | **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness** | Tri Dao et al. (NeurIPS 2022) | 高级 | Flash Attention原始论文，理解Attention计算优化 |
| 2 | **FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning** | Tri Dao | 高级 | Flash Attention v2，端侧推理优化核心 |
| 3 | **vLLM: PagedAttention 论文** | UC Berkeley | 高级 | PagedAttention原理，KV Cache内存管理创新 |
| 4 | **Speculative Decoding 论文** | DeepMind/Google | 高级 | 投机解码原理，端侧推理加速前沿 |
| 5 | **LLM 推理优化技术综述** | 知乎/技术博客 | 中级 | 中文综述，快速了解LLM推理优化全景 |
| 6 | **llama.cpp 技术原理分析** | GitHub/博客 | 中级 | 理解llama.cpp的GGUF量化与CPU推理优化 |
| 7 | **MLC-LLM 技术文档** | mlc.ai | 中级 | MLC-LLM编译优化原理，跨平台端侧部署 |
| 8 | **Roofline Model 原始论文** | UC Berkeley (CACM 2009) | 中级 | Roofline性能分析模型，硬件性能瓶颈分析基础 |
| 9 | **Qualcomm AI Hub 文档** | Qualcomm Developer | 中级 | 高通端侧AI部署文档，理解NPU推理链路 |
| 10 | **Apple MLX 框架文档** | Apple | 中级 | Apple Silicon上的ML推理框架 |

##### 开源项目/代码库

| 序号 | 项目名 | GitHub地址 | 难度 | 推荐理由 |
|------|--------|-----------|------|----------|
| 1 | **llama.cpp** | github.com/ggerganov/llama.cpp | 中级 | **核心项目**：端侧LLM推理标杆，C++实现，GGUF量化，支持CPU/GPU |
| 2 | **MLC-LLM** | github.com/mlc-ai/mlc-llm | 中-高级 | 跨平台端侧LLM部署，编译优化，支持iOS/Android/Web |
| 3 | **vLLM** | github.com/vllm-project/vllm | 中-高级 | PagedAttention实现，虽然主要面向服务端，但推理优化思路通用 |
| 4 | **FlashAttention** | github.com/Dao-AILab/flash-attention | 高级 | Flash Attention官方实现，理解Attention计算优化 |
| 5 | **PowerInfer** | github.com/SJTU-IPADS/PowerInfer | 中-高级 | PC端侧LLM推理，CPU+GPU混合推理 |
| 6 | **ExecuTorch** | github.com/pytorch/executorch | 中级 | PyTorch官方端侧推理框架 |
| 7 | **LMDeploy** | github.com/InternLM/lmdeploy | 中级 | 商汤/InternLM团队推理部署工具，含端侧支持 |
| 8 | **TensorRT-LLM** | github.com/NVIDIA/TensorRT-LLM | 高级 | NVIDIA推理优化框架，理解推理优化最佳实践 |
| 9 | **Apple MLX** | github.com/ml-explore/mlx | 中级 | Apple Silicon上的ML推理框架 |
| 10 | **ONNX Runtime** | github.com/microsoft/onnxruntime | 中级 | 微软跨平台推理引擎，端侧部署参考 |

---

### 领域三：Transformer与LLM架构基础

> 对应：专业知识-知识点3（Transformer基础）

#### 3.1 知识点详细梳理

**3.1.1 Token与Tokenization**

| 子知识点 | 详细内容 |
|----------|----------|
| Token概念 | 文本的最小处理单元，将自然语言映射为离散ID |
| BPE（Byte Pair Encoding） | 从字符级开始，迭代合并最高频字节对。GPT系列使用 |
| WordPiece | 类似BPE但使用似然最大化选择合并。BERT使用 |
| SentencePiece | 直接在原始文本上操作，支持多语言。LLaMA/Qwen使用 |
| 词表大小影响 | 词表大→每个token信息多→序列短→KV Cache小，但Embedding层参数大 |
| 端侧Tokenization考虑 | 多语言支持、中文分词效率、词表大小与Embedding参数的权衡 |

**3.1.2 Embedding**

| 子知识点 | 详细内容 |
|----------|----------|
| 词嵌入 | 将token ID映射为稠密向量，Embedding矩阵大小 = vocab_size × hidden_dim |
| 位置编码 | 为序列中每个位置添加位置信息 |
| 绝对位置编码 | 正弦/余弦编码或可学习位置编码，位置固定 |
| 相对位置编码 | 基于token间相对距离编码，泛化性更好 |
| RoPE（Rotary Position Embedding） | 旋转位置编码，通过旋转矩阵编码相对位置。LLaMA/Qwen使用 |
| ALiBi（Attention with Linear Biases） | 在Attention分数中添加线性偏置，无需位置编码 |
| Embedding参数量 | 词表32000 × 隐藏维度4096 = 1.3亿参数（7B模型中占~2%） |

**3.1.3 Attention机制**

| 子知识点 | 详细内容 |
|----------|----------|
| Self-Attention计算 | Q = X·W_Q, K = X·W_K, V = X·W_V; Attention(Q,K,V) = softmax(QK^T/√d_k)V |
| 缩放因子 | 1/√d_k 防止点积过大导致softmax梯度消失 |
| Multi-Head Attention | 将Q/K/V分成多个头并行计算，再拼接。不同头学习不同子空间 |
| 注意力复杂度 | 时间复杂度 O(n²d)，空间复杂度 O(n²)。n为序列长度 |
| Causal Mask | 解码器中使用三角掩码，防止看到未来token |
| KV Cache | 推理时缓存历史K/V，避免重复计算。内存 = 2 × L × H × n × d × precision |
| MHA vs MQA vs GQA | MHA：每头独立K/V；MQA：所有头共享K/V；GQA：分组共享K/V |

**3.1.4 Transformer Block**

| 子知识点 | 详细内容 |
|----------|----------|
| FFN层 | 两层线性变换+激活函数：FFN(x) = W2·activation(W1·x + b1) + b2 |
| FFN参数占比 | FFN中间维度通常为4倍隐藏维度，参数量约占模型总参数的2/3 |
| 激活函数 | ReLU→GELU→SwiGLU（LLaMA使用），SwiGLU = Swish(xW1) ⊗ xW3 |
| LayerNorm | 层归一化，Pre-Norm vs Post-Norm。现代LLM多用Pre-Norm |
| RMSNorm | LayerNorm的简化版，移除均值计算，更高效。LLaMA使用 |
| 残差连接 | x + Sublayer(x)，解决深层网络梯度消失 |
| 前向传播计算量 | 每token计算量 ≈ 2 × 参数量（矩阵乘法的FLOPs估算） |

**3.1.5 主流LLM架构对比**

| 模型 | 架构特点 | 参数量 | 端侧适用性 |
|------|----------|--------|-----------|
| LLaMA系列 | Decoder-only, RoPE, RMSNorm, SwiGLU, GQA(v2+) | 7B/13B/70B | 7B经量化可端侧部署 |
| Qwen系列 | Decoder-only, RoPE, GQA, 多语言支持 | 0.5B/1.5B/7B/14B | 0.5B/1.5B非常适合端侧 |
| Phi系列 | 小参数量高质量训练，专为端侧设计 | 1.3B/2.7B/3.8B | 专为端侧设计 |
| Gemma系列 | Google开源，Decoder-only | 2B/7B | 2B适合端侧 |
| MobileLLM | Meta专为端侧设计的超轻量架构 | <1B | 端侧专用 |
| MiniCPM | 面壁智能，小参数高性能 | 1.2B/2B | 端侧友好 |

**3.1.6 LLM结构-显存-速度关系**

| 关系 | 说明 |
|------|------|
| 参数量 → 显存 | 模型权重显存 = 参数量 × 每参数字节数（FP16: 2B/参数, INT4: 0.5B/参数） |
| 层数 × 头数 × 头维度 → KV Cache | KV Cache = 2 × L × H × d × seq_len × precision |
| 序列长度 → 计算量 | Attention计算量随序列长度平方增长 |
| 隐藏维度 → FFN计算量 | FFN计算量 ∝ hidden_dim × intermediate_dim |
| 模型深度 vs 宽度 | 深而窄 vs 浅而宽的权衡，影响推理延迟和效果 |

#### 3.2 学习资料规划

##### 书籍

| 序号 | 书名 | 作者/来源 | 难度 | 推荐理由 |
|------|------|-----------|------|----------|
| 1 | **《Natural Language Processing with Transformers》** | Lewis Tunstall, Leandro von Werra, Thomas Wolf (Hugging Face) | 入门-中级 | Hugging Face团队著作，系统讲解Transformer架构与NLP应用 |
| 2 | **《Speech and Language Processing》第3版（草稿）** | Daniel Jurafsky & James H. Martin | 中级 | NLP经典教材，第10-11章深入讲解Transformer和注意力机制 |
| 3 | **《Deep Learning》** | Ian Goodfellow et al. | 中级 | 深度学习基础，第6章前馈网络、第8章优化算法 |
| 4 | **《Build a Large Language Model (From Scratch)》** | Sebastian Raschka | 中级 | 从零构建LLM，深入理解Transformer每个组件 |

##### 大学课程

| 序号 | 课程名 | 来源 | 难度 | 推荐理由 |
|------|--------|------|------|----------|
| 1 | **CS224N: Natural Language Processing with Deep Learning** | Stanford University | 中级 | **核心推荐**：NLP+深度学习经典课程，Lecture 11-12专门讲解Transformer和Attention |
| 2 | **CS25: Transformers United** | Stanford University | 中级 | 专门讲解Transformer的研讨会式课程，覆盖架构原理和前沿应用 |
| 3 | **Introduction to Deep Learning (6.S191)** | MIT | 入门-中级 | MIT深度学习入门课程，包含Transformer和Attention讲解 |
| 4 | **NYU Deep Learning (DS-GA 1008)** | NYU (Yann LeCun) | 中级 | 深度学习理论课程，理解Transformer数学基础 |

##### 博客与技术文章

| 序号 | 文章名 | 来源 | 难度 | 推荐理由 |
|------|--------|------|------|----------|
| 1 | **Attention Is All You Need** | Vaswani et al. (NeurIPS 2017) | 中级 | Transformer原始论文，必读经典 |
| 2 | **The Illustrated Transformer** | Jay Alammar (jalammar.github.io) | 入门 | **核心推荐**：图解Transformer，最直观的Transformer入门文章 |
| 3 | **The Annotated Transformer** | Harvard NLP | 中级 | 带代码注释的Transformer实现，理论+代码结合 |
| 4 | **LLaMA: Open and Efficient Foundation Language Models** | Meta AI | 中级 | LLaMA论文，理解现代LLM架构设计 |
| 5 | **Rotary Position Embedding (RoPE) 论文** | Jianlin Su et al. | 中级 | RoPE位置编码原理，现代LLM标配 |
| 6 | **GQA: Training Generalized Multi-Query Transformer Models** | Google Research | 中级 | GQA论文，端侧KV Cache优化关键架构 |
| 7 | **MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Inference** | Meta AI | 中级 | Meta端侧LLM架构设计论文，专为端侧优化 |
| 8 | **Phi-3 Technical Report** | Microsoft | 中级 | 小模型高质量训练的架构与策略 |
| 9 | **MiniCPM 技术报告** | 面壁智能 | 中级 | 中文端侧小模型技术报告 |
| 10 | **Let's build the GPT Tokenizer** | Andrej Karpathy (YouTube) | 入门-中级 | Karpathy讲解BPE Tokenizer原理与实现 |
| 11 | **The Illustrated GPT-2 (Visualizing Transformer Language Models)** | Jay Alammar | 入门-中级 | 图解GPT-2，理解Decoder-only架构 |

##### 开源项目/代码库

| 序号 | 项目名 | GitHub地址 | 难度 | 推荐理由 |
|------|--------|-----------|------|----------|
| 1 | **nanoGPT** | github.com/karpathy/nanoGPT | 中级 | **核心推荐**：Karpathy从零实现GPT，简洁清晰，理解Transformer每个组件 |
| 2 | **Hugging Face Transformers** | github.com/huggingface/transformers | 中级 | 主流LLM库，支持几乎所有开源模型，理解架构实现 |
| 3 | **LLaMA (Meta)** | github.com/meta-llama/llama | 中级 | Meta LLaMA官方实现，理解现代LLM架构 |
| 4 | **llama2.c** | github.com/karpathy/llama2.c | 中级 | Karpathy用纯C实现的LLaMA 2推理，理解LLM推理底层 |
| 5 | **Qwen (阿里)** | github.com/QwenLM/Qwen2 | 中级 | Qwen模型实现，端侧友好小模型参考 |
| 6 | **Phi-3 (微软)** | github.com/microsoft/Phi-3 | 中级 | 微软端侧小模型实现 |
| 7 | **MiniCPM** | github.com/OpenBMB/MiniCPM | 中级 | 面壁智能端侧小模型，中文场景参考 |

---

### 领域四：模型训练基础（预训练/SFT/对齐）

> 对应：专业知识-知识点4（模型训练基础）、关键技能-技能项1（端侧约束下的模型训练与适配优化能力）

#### 4.1 知识点详细梳理

**4.1.1 预训练（Pre-training）**

| 子知识点 | 详细内容 |
|----------|----------|
| 语言建模目标 | CLM（Causal Language Modeling）：预测下一个token；MLM（Masked Language Modeling）：预测被mask的token |
| 预训练数据 | 网页文本、书籍、代码、学术论文等。数据量通常为万亿token级 |
| 训练超参数 | 学习率（通常1e-4到3e-4）、Batch Size、Warmup步数、权重衰减、梯度裁剪 |
| Scaling Law | Kaplan et al. (2020) 和 Chinchilla (2022)：模型参数量、数据量、计算量的最优配比关系 |
| 预训练与端侧关系 | 端侧小模型通常从预训练大模型蒸馏/继承知识，而非从零预训练 |

**4.1.2 监督微调（SFT）**

| 子知识点 | 详细内容 |
|----------|----------|
| SFT数据格式 | {"instruction": "...", "input": "...", "output": "..."}，多轮对话格式 |
| SFT训练目标 | 对output部分计算交叉熵损失，instruction部分不计算损失（loss masking） |
| 全参数微调 | 更新所有参数，效果最好但需要大量显存 |
| LoRA微调 | 仅训练低秩矩阵，显存需求降低10-100倍 |
| 灾难性遗忘 | 微调后模型遗忘预训练知识，缓解策略：小学习率、混合预训练数据 |
| SFT与端侧 | 端侧SFT聚焦场景适配：问答、工具调用、设备控制等 |
| SFT数据配比 | 通用数据与场景数据的配比影响模型泛化性与专业性 |

**4.1.3 偏好对齐（Alignment）**

| 子知识点 | 详细内容 |
|----------|----------|
| 对齐目标 | 让模型输出符合人类偏好（有用性、无害性、诚实性） |
| RLHF流程 | SFT → 训练Reward Model → PPO强化学习优化策略 |
| Reward Model | 使用偏好数据（chosen vs rejected）训练，输出标量奖励值 |
| PPO算法 | Proximal Policy Optimization，限制策略更新幅度，防止崩溃 |
| DPO（Direct Preference Optimization） | 无需训练RM，直接用偏好数据优化策略。简化RLHF流程 |
| DPO损失函数 | L_DPO = -log σ(β·(log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x))) |
| 其他对齐方法 | KTO（Kahneman-Tversky Optimization）、IPO（Identity Preference Optimization）、SimPO（Simple Preference Optimization）、ORPO |
| 对齐与端侧 | 端侧场景对安全性要求高，需在有限数据下实现有效对齐 |
| 对齐税（Alignment Tax） | 对齐后模型能力可能下降，需要平衡对齐效果与基础能力 |

**4.1.4 端侧训练的特殊性**

| 子知识点 | 详细内容 |
|----------|----------|
| 端侧 vs 云端训练差异 | 端侧关注小模型效果、数据质量和场景适配；云端关注大模型规模和通用能力 |
| 小模型效果保障 | 蒸馏继承大模型知识、高质量场景数据、精细微调策略 |
| 场景适配 | 针对具体端侧任务（设备控制、本地问答）定制微调 |
| 量化友好训练 | 训练过程中模拟量化噪声，使模型在量化后保持效果 |
| 端侧训练资源约束 | 训练在云端完成但面向端侧部署，需考虑推理约束进行训练优化 |

#### 4.2 学习资料规划

##### 书籍

| 序号 | 书名 | 作者/来源 | 难度 | 推荐理由 |
|------|------|-----------|------|----------|
| 1 | **《Natural Language Processing with Transformers》** | Hugging Face团队 | 入门-中级 | 第7章SFT微调实践，第3章预训练原理 |
| 2 | **《Build a Large Language Model (From Scratch)》** | Sebastian Raschka | 中级 | 从零实现预训练→SFT→对齐全流程，代码级理解 |
| 3 | **《Deep Reinforcement Learning》** | Aske Plaat et al. | 高级 | 理解PPO算法原理，RLHF的理论基础 |
| 4 | **《强化学习导论》（Reinforcement Learning: An Introduction）** | Richard S. Sutton & Andrew G. Barto | 中-高级 | RL经典教材，理解RLHF的强化学习基础 |

##### 大学课程

| 序号 | 课程名 | 来源 | 难度 | 推荐理由 |
|------|--------|------|------|----------|
| 1 | **CS224N: NLP with Deep Learning** | Stanford | 中级 | Lecture涵盖预训练、微调、LLM训练 |
| 2 | **CS285: Deep Reinforcement Learning** | UC Berkeley | 高级 | 深度强化学习课程，理解PPO和RLHF的RL基础 |
| 3 | **Deep RL Bootcamp** | UC Berkeley | 中-高级 | 深度强化学习训练营，PPO实践 |
| 4 | **Hugging Face NLP Course** | Hugging Face | 入门-中级 | 免费在线课程，SFT/PEFT/对齐实践 |
| 5 | **Stanford CS324: Large Language Models** | Stanford | 中级 | 专门讲解LLM，涵盖训练、对齐、评估 |

##### 博客与技术文章

| 序号 | 文章名 | 来源 | 难度 | 推荐理由 |
|------|--------|------|------|----------|
| 1 | **Training language models to follow instructions with human feedback (InstructGPT)** | OpenAI | 高级 | RLHF里程碑论文，理解SFT→RM→PPO完整流程 |
| 2 | **Direct Preference Optimization: Your Language Model is Secretly a Reward Model** | Rafailov et al. | 高级 | DPO原始论文，简化RLHF的核心方法 |
| 3 | **LoRA: Low-Rank Adaptation of Large Language Models** | Edward Hu et al. | 中级 | LoRA原始论文，端侧微调核心技术 |
| 4 | **Scaling Laws for Neural Language Models** | Kaplan et al. | 中级 | Scaling Law论文，理解模型规模与效果关系 |
| 5 | **Training Compute-Optimal Large Language Models (Chinchilla)** | DeepMind | 中级 | Chinchilla Scaling Law，最优计算配比 |
| 6 | **LIMA: Less Is More for Alignment** | Meta AI | 中级 | 少量高质量数据即可对齐，端侧数据策略参考 |
| 7 | **Zephyr: Direct Distillation of LM Alignment** | Hugging Face | 中级 | 使用DPO进行对齐蒸馏的实践 |
| 8 | **ORPO: Monolithic Preference Optimization without Reference Model** | | 中级 | 无需参考模型的对齐方法，简化训练流程 |
| 9 | **SimPO: Simple Preference Optimization with a Reference-Free Reward** | | 中级 | 无参考模型偏好优化，端侧对齐新方法 |
| 10 | **RLHF / DPO / PPO 综合解读** | 知乎/Hugging Face Blog | 中级 | 中文/英文综合解读，快速理解对齐方法对比 |
| 11 | **Karpathy: State of GPT (YouTube/Blog)** | Andrej Karpathy | 中级 | Karpathy对LLM训练全流程的系统性讲解 |
| 12 | **Karpathy: Deep Dive into LLMs like ChatGPT** | Andrej Karpathy (YouTube) | 中级 | 从预训练到RLHF的完整讲解 |

##### 开源项目/代码库

| 序号 | 项目名 | GitHub地址 | 难度 | 推荐理由 |
|------|--------|-----------|------|----------|
| 1 | **Hugging Face TRL (Transformer Reinforcement Learning)** | github.com/huggingface/trl | 中级 | **核心推荐**：SFT/DPO/PPO/RLHF的官方训练库 |
| 2 | **LLaMA-Factory** | github.com/hiyouga/LLaMA-Factory | 入门-中级 | **核心推荐**：一站式LLM微调框架，支持SFT/DPO/LoRA/量化，中文友好 |
| 3 | **Axolotl** | github.com/OpenAccess-AI-Collective/axolotl | 中级 | 高级LLM微调工具，支持多种训练方法 |
| 4 | **DeepSpeed-Chat** | github.com/microsoft/DeepSpeed | 高级 | 微软RLHF训练框架，包含SFT/RM/PPO全流程 |
| 5 | **trlX (CarperAI)** | github.com/CarperAI/trlx | 中-高级 | RLHF训练框架，PPO/DPO实现 |
| 6 | **Unsloth** | github.com/unslothai/unsloth | 入门-中级 | 超快速LLM微调库，显存优化，适合端侧实验 |
| 7 | **LLM-Foundry (MosaicML)** | github.com/mosaicml/llm-foundry | 中级 | LLM训练全流程工具，预训练→SFT→对齐 |
| 8 | **MiniCPM 微调工具链** | github.com/OpenBMB/MiniCPM | 中级 | 端侧小模型微调实践参考 |

---

### 领域五：数据清洗与合成

> 对应：专业知识-知识点5（数据清洗与合成）、关键技能-技能项3（数据处理与数据集构建能力）

#### 5.1 知识点详细梳理

**5.1.1 数据清洗基础**

| 子知识点 | 详细内容 |
|----------|----------|
| 精确去重 | 完全匹配去重，使用Hash表 |
| 模糊去重 | MinHash + LSH（Locality Sensitive Hashing），去除相似但不完全相同的数据 |
| 规则过滤 | 长度过滤、语言检测、特殊字符比例、重复n-gram比例 |
| 质量过滤 | 使用质量分类器（如fasttext）过滤低质量文本，Perplexity过滤 |
| 安全过滤 | 有害内容检测、PII（个人身份信息）检测与脱敏 |
| 数据格式标准化 | 统一编码、统一JSON格式、去除HTML标签 |
| 数据质量评估 | 准确率、多样性指标、覆盖率指标、噪声率 |

**5.1.2 指令数据构造**

| 子知识点 | 详细内容 |
|----------|----------|
| 指令数据格式 | {"instruction": "任务描述", "input": "可选输入", "output": "期望输出"} |
| 人工标注 | 人工编写高质量指令-回答对，成本高但质量好 |
| Self-Instruct | 用LLM自身生成指令数据，通过种子指令引导 |
| Evol-Instruct | 逐步演化指令复杂度（WizardLM方法），增加指令多样性 |
| 蒸馏数据生成 | 用大模型（GPT-4/Claude）生成高质量指令数据用于小模型训练 |
| 高质量数据集参考 | Alpaca(52K), Vicuna, ShareGPT, OpenOrca, Dolly, OASST1 |

**5.1.3 端侧场景数据构造**

| 子知识点 | 详细内容 |
|----------|----------|
| 问答场景 | 本地知识问答、FAQ问答、文档摘要问答 |
| 工具调用场景 | 系统API调用、天气查询、日历管理等Function Calling数据 |
| 设备控制场景 | 智能家居控制、手机设置操作、车载系统交互 |
| 多轮对话 | 上下文理解、指代消解、多轮推理数据 |
| 端侧隐私保护 | 数据脱敏、本地化处理、不涉及云端数据传输 |
| 场景数据配比 | 通用能力数据与场景专用数据的配比策略 |

**5.1.4 数据合成技术**

| 子知识点 | 详细内容 |
|----------|----------|
| LLM辅助合成 | 用大模型生成训练数据（如GPT-4生成指令-回答对） |
| 回译增强 | 翻译到其他语言再翻译回来，增加表达多样性 |
| 文本改写 | 同义改写、风格变换、复杂度调整 |
| 数据扰动 | 同义词替换、随机删除、句序调整 |
| 蒸馏数据 | 大模型输出作为小模型训练目标 |
| 合成数据质量评估 | 自动评估（用LLM评估LLM生成数据）、人工抽检 |

**5.1.5 数据质量原则**

| 原则 | 说明 |
|------|------|
| 高质量 | 准确性（事实正确）、流畅性（语言通顺）、有用性（回答切题） |
| 多样性 | 指令类型多样（问答/推理/创作/翻译）、语言风格多样、难度梯度 |
| 覆盖率 | 场景覆盖（所有目标场景）、能力覆盖（理解/推理/生成/工具调用）、难度覆盖 |
| 噪声控制 | 控制错误率、偏差控制（性别/种族/文化）、对抗样本过滤 |

#### 5.2 学习资料规划

##### 书籍

| 序号 | 书名 | 作者/来源 | 难度 | 推荐理由 |
|------|------|-----------|------|----------|
| 1 | **《数据准备 for 机器学习》**（Data Preparation for Machine Learning） | Jason Brownlee | 入门-中级 | 数据清洗与预处理实践指南 |
| 2 | **《Feature Engineering and Selection》** | Max Kuhn, Kjell Johnson | 中级 | 特征工程与数据选择，理解数据质量对模型的影响 |
| 3 | **《数据密集型应用系统设计》**（Designing Data-Intensive Applications） | Martin Kleppmann | 中级 | 理解大规模数据处理系统设计 |

##### 大学课程

| 序号 | 课程名 | 来源 | 难度 | 推荐理由 |
|------|--------|------|------|----------|
| 1 | **Stanford CS324: Large Language Models** | Stanford | 中级 | LLM课程中数据构造章节，理解LLM训练数据Pipeline |
| 2 | **CMU 11-611: Advanced NLP** | CMU | 中级 | 高级NLP课程，包含数据标注与构造 |
| 3 | **Hugging Face NLP Course** | Hugging Face | 入门-中级 | 数据集处理实践，使用datasets库 |

##### 博客与技术文章

| 序号 | 文章名 | 来源 | 难度 | 推荐理由 |
|------|--------|------|------|----------|
| 1 | **Self-Instruct: Aligning Language Models with Self-Generated Instructions** | UW | 中级 | Self-Instruct论文，LLM指令数据自动生成方法 |
| 2 | **WizardLM: Empowering Large Language Models to Follow Complex Instructions** | Xu et al. | 中级 | Evol-Instruct方法，指令演化增强多样性 |
| 3 | **Phi-3 Technical Report: 数据质量部分** | Microsoft | 中级 | 微软小模型高质量数据构造策略 |
| 4 | **Textbooks Are All You Need (Phi-1)** | Microsoft | 中级 | "教科书级"数据质量理念 |
| 5 | **DataComp: In Search of the Next Generation of Multimodal Datasets** | DataComp | 中级 | 数据集设计竞赛，数据质量评估方法论 |
| 6 | **RedPajama 数据处理文档** | Together AI | 中级 | 开源LLM训练数据处理Pipeline |
| 7 | **CCNet: Extracting High Quality Monolingual Datasets from Web Crawl Data** | Facebook AI | 中级 | 网页数据清洗Pipeline |
| 8 | **Deduplicating Training Data Makes Language Models Better** | Lee et al. | 中级 | 数据去重对LLM训练的影响研究 |
| 9 | **Alpaca 数据集说明文档** | Stanford | 入门 | Alpaca 52K指令数据集构造方法 |
| 10 | **OpenOrca / OpenHermes 数据集** | Hugging Face | 入门 | 开源高质量指令数据集参考 |

##### 开源项目/代码库

| 序号 | 项目名 | GitHub地址 | 难度 | 推荐理由 |
|------|--------|-----------|------|----------|
| 1 | **Hugging Face Datasets** | github.com/huggingface/datasets | 入门-中级 | 数据集加载与处理标准库 |
| 2 | **Self-Instruct** | github.com/yizhongw/self-instruct | 中级 | Self-Instruct方法官方实现 |
| 3 | **WizardLM** | github.com/nlpxucan/WizardLM | 中级 | Evol-Instruct方法实现 |
| 4 | **RedPajama-Data** | github.com/togethercomputer/RedPajama-Data | 中级 | 开源LLM训练数据处理Pipeline |
| 5 | **datatrove** | github.com/huggingface/datatrove | 中级 | Hugging Face数据处理库，去重/过滤/质量评估 |
| 6 | **Cleanlab** | github.com/cleanlab/cleanlab | 中级 | 数据质量自动检测与清洗工具 |
| 7 | **distilabel** | github.com/argilla-io/distilabel | 中级 | 数据合成与标注Pipeline工具 |
| 8 | **OpenOrca** | huggingface.co/datasets | 入门 | 开源大规模指令数据集 |

---

### 领域六：端侧高效模型架构设计

> 对应：关键技能-技能项2（面向端侧的高效模型架构设计能力）

#### 6.1 知识点详细梳理

**6.1.1 模型规模设计**

| 子知识点 | 详细内容 |
|----------|----------|
| 参数量选择 | <1B：超轻量，适合简单任务；1-3B：平衡型，主流端侧选择；3-7B：高性能端侧，需量化 |
| 层数 vs 隐藏维度 | 深而窄（更多层，更小隐藏维度）：更好的推理效果但延迟高；浅而宽：延迟低但效果可能受限 |
| 注意力头数 | 头数影响模型表达能力与KV Cache大小。GQA/MQA可减少KV Cache |
| 词表大小 | 大词表→更短序列→更小KV Cache，但Embedding参数更大 |
| 端侧模型规格参考 | Phi-3(3.8B, 32层, 3072隐藏维度)、Qwen-1.5B(28层, 1536隐藏维度)、MiniCPM(2.4B) |

**6.1.2 高效Attention机制**

| 子知识点 | 详细内容 |
|----------|----------|
| MHA（Multi-Head Attention） | 标准注意力，每头独立K/V。KV Cache最大 |
| MQA（Multi-Query Attention） | 所有头共享一组K/V。KV Cache最小，但效果可能下降 |
| GQA（Grouped-Query Attention） | 分组共享K/V，在MHA和MQA之间取得平衡。LLaMA-2/3使用 |
| KV Cache优化 | 量化KV Cache（FP16→INT8/INT4）、KV Cache压缩、滑动窗口 |
| Sliding Window Attention | 只关注局部窗口内的token，减少Attention计算量 |
| Linear Attention | 将Attention复杂度从O(n²)降到O(n)，适合长序列 |

**6.1.3 量化友好架构**

| 子知识点 | 详细内容 |
|----------|----------|
| 量化敏感层 | 某些层（如LayerNorm后的激活值、Embedding层）对量化敏感 |
| 量化友好激活函数 | ReLU/SiLU比GELU更量化友好（无指数运算） |
| 量化友好归一化 | RMSNorm比LayerNorm更量化友好（无均值计算） |
| 结构对量化精度影响 | 残差连接的尺度、注意力softmax的温度、FFN中间维度比例 |
| 混合精度量化策略 | 敏感层保持高精度，非敏感层使用低精度 |

**6.1.4 架构效果-成本权衡**

| 维度 | 权衡关系 |
|------|----------|
| 参数量 vs 效果 | 遵循Scaling Law，但小模型通过蒸馏可超越同参数量从零训练模型 |
| 计算量(FLOPs) vs 延迟 | FLOPs不完全等于延迟，受内存带宽、并行度影响 |
| KV Cache vs 上下文长度 | KV Cache随上下文线性增长，端侧需限制上下文长度 |
| 端侧可部署性综合评估 | 参数量+计算量+内存+功耗+精度的综合评分 |

#### 6.2 学习资料规划

##### 书籍

| 序号 | 书名 | 作者/来源 | 难度 | 推荐理由 |
|------|------|-----------|------|----------|
| 1 | **《Designing Neural Network Architectures》** | 各种综述 | 高级 | 神经网络架构设计方法论 |
| 2 | **《Efficient Deep Learning》**（在线） | MIT 6.5940 讲义 | 中-高级 | 高效深度学习架构设计 |

##### 大学课程

| 序号 | 课程名 | 来源 | 难度 | 推荐理由 |
|------|--------|------|------|----------|
| 1 | **MIT 6.5940: TinyML and Efficient Deep Learning Computing** | MIT (Song Han) | 中-高级 | **核心推荐**：高效模型架构设计、NAS、硬件感知优化 |
| 2 | **Stanford CS324: Large Language Models** | Stanford | 中级 | LLM架构设计与Scaling Law |
| 3 | **CMU 10-708: Probabilistic Graphical Models** | CMU | 高级 | 理解概率模型与架构设计关系 |

##### 博客与技术文章

| 序号 | 文章名 | 来源 | 难度 | 推荐理由 |
|------|--------|------|------|----------|
| 1 | **MobileLLM: Optimizing Sub-billion Parameter Language Models** | Meta AI | 中级 | **核心推荐**：Meta端侧LLM架构设计论文 |
| 2 | **GQA: Training Generalized Multi-Query Transformer Models** | Google | 中级 | GQA论文，端侧KV Cache优化核心 |
| 3 | **Longformer: The Long-Document Transformer** | Allen AI | 中级 | 滑动窗口注意力机制 |
| 4 | **Linear Attention Is (Nearly) All You Need** | | 中级 | 线性注意力机制 |
| 5 | **Phi-3 Technical Report** | Microsoft | 中级 | 小模型架构设计实践 |
| 6 | **Scaling Laws for Neural Language Models** | OpenAI | 中级 | 理解模型规模设计原理 |
| 7 | **A Simple and Effective Pruning Approach for Large Language Models (Wanda)** | | 高级 | LLM结构化剪枝方法 |
| 8 | **Efficient Streaming Language Models with Attention Sinks (StreamingLLM)** | MIT | 高级 | 流式LLM注意力优化 |

##### 开源项目/代码库

| 序号 | 项目名 | GitHub地址 | 难度 | 推荐理由 |
|------|--------|-----------|------|----------|
| 1 | **MobileLLM** | github.com/facebookresearch/MobileLLM | 中-高级 | Meta端侧LLM架构官方实现 |
| 2 | **Phi-3** | github.com/microsoft/Phi-3 | 中级 | 微软端侧小模型架构 |
| 3 | **Qwen2** | github.com/QwenLM/Qwen2 | 中级 | Qwen架构（GQA+RoPE+SwiGLU）参考 |
| 4 | **MiniCPM** | github.com/OpenBMB/MiniCPM | 中级 | 端侧小模型架构设计参考 |
| 5 | **StreamingLLM** | github.com/mit-han-lab/streaming-llm | 高级 | 流式LLM注意力优化实现 |

---

### 领域七：模型部署与工程调试

> 对应：关键技能-技能项4（模型部署与工程调试能力）

#### 7.1 知识点详细梳理

**7.1.1 模型转换**

| 子知识点 | 详细内容 |
|----------|----------|
| PyTorch → ONNX | 使用torch.onnx.export导出，需注意动态轴设置（序列长度） |
| ONNX → 端侧格式 | ONNX→MNN（使用MNNConvert）、ONNX→NCNN（使用onnx2ncnn）、ONNX→TFLite |
| GGUF格式转换 | 使用llama.cpp的convert脚本将HF模型转为GGUF格式 |
| 量化模型导出 | PTQ量化后导出、QAT训练后导出量化模型 |
| 常见转换问题 | 不支持的算子、动态shape处理、精度损失、自定义算子实现 |

**7.1.2 部署验证**

| 子知识点 | 详细内容 |
|----------|----------|
| 推理正确性验证 | 比较端侧推理输出与PyTorch/原始模型输出的一致性 |
| 精度一致性校验 | 逐层对比输出，定位精度偏差来源（量化误差/算子实现差异） |
| 性能基准测试 | 测量TTFT、TPS、内存峰值、功耗 |
| 内存/功耗测量 | 使用系统工具（Android Profiler/Xcode Instruments/perf） |

**7.1.3 常见部署问题**

| 问题类型 | 原因 | 解决思路 |
|----------|------|----------|
| 格式转换失败 | 不支持的算子/结构 | 自定义算子实现、修改模型结构 |
| 精度不一致 | 量化误差、算子实现差异 | 混合精度、逐层定位、算子替换 |
| 推理慢 | 未利用硬件加速、内存瓶颈 | 启用NPU/GPU、量化减少内存访问 |
| 内存溢出 | 模型过大、KV Cache过大 | 量化、减小上下文长度、GQA/MQA |
| 崩溃 | 内存不足、算子不兼容 | 分块推理、更新框架版本 |

**7.1.4 工程调试技能**

| 子知识点 | 详细内容 |
|----------|----------|
| 性能Profile | 使用Profiler工具（PyTorch Profiler/Android Profiler/Xcode Instruments） |
| 内存分析 | 内存泄漏检测、峰值内存分析、内存碎片评估 |
| 日志分析 | 推理日志解析、错误码识别、调用链追踪 |
| A/B测试 | 端侧A/B测试框架、模型版本对比 |

#### 7.2 学习资料规划

##### 书籍

| 序号 | 书名 | 作者/来源 | 难度 | 推荐理由 |
|------|------|-----------|------|----------|
| 1 | **《TinyML》** | Pete Warden et al. | 入门-中级 | 端侧模型部署实践 |
| 2 | **《AI模型部署与推理优化》**（相关技术书籍） | 多种 | 中级 | 模型部署工程实践 |

##### 大学课程

| 序号 | 课程名 | 来源 | 难度 | 推荐理由 |
|------|--------|------|------|----------|
| 1 | **MIT 6.5940: TinyML and Efficient Deep Learning Computing** | MIT (Song Han) | 中-高级 | **核心推荐**：包含端侧部署Lab实践 |
| 2 | **Harvard CS249r: Large Language Models** | Harvard | 中级 | LLM应用与部署实践课程 |

##### 博客与技术文章

| 序号 | 文章名 | 来源 | 难度 | 推荐理由 |
|------|--------|------|------|----------|
| 1 | **llama.cpp 部署指南** | GitHub Wiki | 入门-中级 | llama.cpp模型转换与部署全流程 |
| 2 | **MLC-LLM 部署教程** | mlc.ai | 中级 | MLC-LLM跨平台部署教程（iOS/Android/Web） |
| 3 | **MNN LLM 部署文档** | Alibaba MNN | 中级 | MNN端侧LLM部署实践 |
| 4 | **ONNX Runtime Mobile 部署指南** | Microsoft | 中级 | ONNX Runtime端侧部署 |
| 5 | **ExecuTorch 教程** | PyTorch | 中级 | PyTorch官方端侧推理框架教程 |
| 6 | **Qualcomm AI Hub LLM 部署** | Qualcomm | 中级 | 高通NPU端侧LLM部署 |
| 7 | **Apple MLX LLM 部署** | Apple | 中级 | Apple Silicon端侧LLM部署 |
| 8 | **端侧大模型部署实践** | 知乎/技术博客 | 入门-中级 | 中文端侧部署实践分享 |

##### 开源项目/代码库

| 序号 | 项目名 | GitHub地址 | 难度 | 推荐理由 |
|------|--------|-----------|------|----------|
| 1 | **llama.cpp** | github.com/ggerganov/llama.cpp | 中级 | **核心项目**：端侧LLM部署首选工具 |
| 2 | **MLC-LLM** | github.com/mlc-ai/mlc-llm | 中-高级 | 跨平台端侧LLM部署 |
| 3 | **MNN** | github.com/alibaba/MNN | 中级 | 阿里端侧推理引擎 |
| 4 | **NCNN** | github.com/Tencent/ncnn | 中级 | 腾讯端侧推理框架 |
| 5 | **ExecuTorch** | github.com/pytorch/executorch | 中级 | PyTorch端侧推理框架 |
| 6 | **PowerInfer** | github.com/SJTU-IPADS/PowerInfer | 中-高级 | PC端侧LLM推理 |
| 7 | **LMDeploy** | github.com/InternLM/lmdeploy | 中级 | 推理部署工具，含端侧支持 |

---

### 领域八：实验分析与问题定位

> 对应：关键技能-技能项5（实验分析、问题定位与协作交付能力）

#### 8.1 知识点详细梳理

**8.1.1 实验设计**

| 子知识点 | 详细内容 |
|----------|----------|
| 实验变量控制 | 单变量原则：每次只改变一个因素，控制其他不变 |
| 评估指标选择 | 效果指标（Accuracy/BLEU/ROUGE/人工评估）、性能指标（TTFT/TPS/内存/功耗） |
| 基线设置 | 确定baseline模型和指标，所有改进与baseline对比 |
| 消融实验 | 逐一移除组件，评估各组件贡献 |

**8.1.2 Badcase分析**

| Badcase类型 | 原因分析 | 修复策略 |
|-------------|----------|----------|
| 回复不准 | 知识缺失、推理错误、训练数据不足 | 补充领域数据SFT、蒸馏增强 |
| 回复不稳定 | 温度过高、采样策略不当、上下文敏感 | 调整采样参数、增加一致性训练数据 |
| 安全性问题 | 有害内容生成、隐私泄露 | 安全对齐（DPO/RLHF）、安全过滤数据 |
| 量化后效果下降 | 精度损失、量化敏感层 | QAT量化感知训练、混合精度量化 |
| 指令跟随差 | SFT数据不足/质量差 | 增加高质量指令数据、改进SFT策略 |

**8.1.3 性能问题定位**

| 问题 | 排查方法 |
|------|----------|
| 训练不稳定 | 检查学习率、梯度范数、Loss曲线、数据质量 |
| 推理变慢 | 分析KV Cache大小、内存带宽瓶颈、算子效率 |
| 效果下降 | 对比训练数据、检查训练超参、验证量化精度 |
| 内存溢出 | 分析模型大小、KV Cache、激活值内存 |

**8.1.4 实验报告与协作**

| 子知识点 | 详细内容 |
|----------|----------|
| 实验结果整理 | 表格化对比指标、可视化Loss/Accuracy曲线 |
| 技术报告撰写 | 背景→方法→实验→结果→分析→结论的标准结构 |
| 跨团队协作 | 与工程团队沟通部署需求、与硬件团队沟通芯片约束 |
| 问题推动 | Issue跟踪、优先级排序、修复验证闭环 |

#### 8.2 学习资料规划

##### 书籍

| 序号 | 书名 | 作者/来源 | 难度 | 推荐理由 |
|------|------|-----------|------|----------|
| 1 | **《Designing Machine Learning Systems》** | Chip Huyen | 中级 | ML系统设计，包含实验管理与迭代流程 |
| 2 | **《Machine Learning Engineering》** | Andriy Burkov | 中级 | ML工程实践，实验设计与交付 |
| 3 | **《Trustworthy Machine Learning》** | Kush R. Varshney | 中级 | 可信ML，包含模型评估与问题分析 |

##### 大学课程

| 序号 | 课程名 | 来源 | 难度 | 推荐理由 |
|------|--------|------|------|----------|
| 1 | **Stanford CS329S: Machine Learning Systems Design** | Stanford (Chip Huyen) | 中级 | **核心推荐**：ML系统设计与实验管理 |
| 2 | **MIT 6.S191: Introduction to Deep Learning** | MIT | 入门-中级 | 深度学习实验设计与评估 |
| 3 | **Full Stack Deep Learning** | 全栈深度学习课程 | 中级 | ML工程全流程，含实验管理与部署 |

##### 博客与技术文章

| 序号 | 文章名 | 来源 | 难度 | 推荐理由 |
|------|--------|------|------|----------|
| 1 | **LM Evaluation Harness** | EleutherAI | 中级 | 开源LLM评估框架，标准化评估实践 |
| 2 | **HELM: Holistic Evaluation of Language Models** | Stanford CRFM | 中级 | 斯坦福LLM全面评估框架 |
| 3 | **Open LLM Leaderboard** | Hugging Face | 入门 | 开源LLM排行榜，了解主流评估基准 |
| 4 | **LLM 评估方法综述** | 知乎/技术博客 | 中级 | LLM评估方法与指标全面综述 |
| 5 | **A Guide to LLM Evaluation** | Hugging Face Blog | 中级 | LLM评估实践指南 |

##### 开源项目/代码库

| 序号 | 项目名 | GitHub地址 | 难度 | 推荐理由 |
|------|--------|-----------|------|----------|
| 1 | **lm-evaluation-harness** | github.com/EleutherAI/lm-evaluation-harness | 中级 | **核心推荐**：LLM评估标准框架 |
| 2 | **HELM** | github.com/stanford-crfm/helm | 中级 | 斯坦福LLM全面评估框架 |
| 3 | **OpenCompass** | github.com/open-compass/opencompass | 中级 | 上海AI实验室LLM评估框架，中文友好 |
| 4 | **Weights & Biases** | wandb.ai | 入门-中级 | 实验追踪与可视化平台 |
| 5 | **MLflow** | github.com/mlflow/mlflow | 入门-中级 | ML实验管理与模型跟踪 |
| 6 | **LMFlow** | github.com/OptimalScale/LMFlow | 中级 | LLM微调与评估一体化工具 |

---

## 四、学习路径建议（阶段划分与推荐顺序）

### 阶段一：基础筑基（8-12周）

**目标**：建立Transformer/LLM架构基础和训练基础理论认知

| 周次 | 学习内容 | 核心资料 | 产出 |
|------|----------|----------|------|
| 1-2 | Transformer架构原理 | The Illustrated Transformer + Attention Is All You Need + CS224N Lecture 11-12 | 能手推Self-Attention计算流程 |
| 2-3 | Tokenization与Embedding | Karpathy "Let's build the GPT Tokenizer" + nanoGPT代码阅读 | 理解BPE/SentencePiece原理 |
| 3-4 | Transformer Block细节 | The Annotated Transformer + nanoGPT实现 | 能用PyTorch实现完整Transformer |
| 5-6 | LLM架构与主流模型 | LLaMA论文 + Qwen/Phi技术报告 + llama2.c阅读 | 理解RoPE/RMSNorm/SwiGLU/GQA |
| 7-8 | 预训练与Scaling Law | Scaling Laws论文 + Chinchilla论文 + Karpathy视频 | 理解预训练目标与规模设计 |
| 9-10 | SFT原理与实践 | Hugging Face NLP Course + LLaMA-Factory实践 | 完成一次SFT微调实验 |
| 11-12 | 偏好对齐（RLHF/DPO） | InstructGPT论文 + DPO论文 + TRL文档 | 理解RLHF/DPO流程与区别 |

### 阶段二：核心技术深入（8-10周）

**目标**：掌握模型轻量化、推理优化、数据工程三大核心技术

| 周次 | 学习内容 | 核心资料 | 产出 |
|------|----------|----------|------|
| 13-14 | 知识蒸馏 | MIT 6.5940 Lecture（蒸馏部分）+ 蒸馏综述 | 理解Logit/Feature蒸馏原理 |
| 15-16 | 量化技术（PTQ/QAT） | MIT 6.5940 Lecture（量化部分）+ GPTQ/AWQ/SmoothQuant论文 | 用AutoGPTQ/AutoAWQ完成模型量化 |
| 17-18 | LoRA/QLoRA | LoRA论文 + QLoRA论文 + PEFT文档 + Unsloth实践 | 完成LoRA微调实验 |
| 19-20 | LLM推理过程与KV Cache | llama.cpp源码阅读 + Flash Attention论文 | 理解Prefill/Decoding/KV Cache |
| 21-22 | 推理加速技术 | Flash Attention v2 + PagedAttention + Speculative Decoding论文 | 理解各加速技术原理与适用场景 |

### 阶段三：端侧专项（6-8周）

**目标**：掌握端侧推理框架、模型部署、架构设计

| 周次 | 学习内容 | 核心资料 | 产出 |
|------|----------|----------|------|
| 23-24 | 端侧推理框架 | llama.cpp部署实践 + MLC-LLM教程 | 在手机/PC上部署一个量化LLM |
| 25-26 | 模型转换与部署验证 | ONNX转换 + MNN/NCNN部署文档 | 完成PyTorch→端侧格式转换链路 |
| 27-28 | 端侧高效架构设计 | MobileLLM论文 + GQA论文 + Phi-3报告 | 设计一个端侧模型规格方案 |
| 29-30 | 芯片适配与性能调优 | Roofline模型 + Qualcomm AI Hub + Apple MLX | 分析端侧推理性能瓶颈 |

### 阶段四：数据与实战（6-8周）

**目标**：掌握数据工程，完成端到端实战项目

| 周次 | 学习内容 | 核心资料 | 产出 |
|------|----------|----------|------|
| 31-32 | 数据清洗与去重 | datatrove + RedPajama-Data + Cleanlab | 搭建数据清洗Pipeline |
| 33-34 | 指令数据构造 | Self-Instruct + WizardLM + 蒸馏数据生成 | 构造端侧场景指令数据集 |
| 35-36 | 端侧场景全链路实战 | 综合运用所有知识 | 完成端侧LLM全链路项目 |
| 37-38 | 评估与Badcase分析 | lm-evaluation-harness + OpenCompass + HELM | 完成模型评估与问题定位 |

### 阶段五：进阶与前沿（持续）

**目标**：跟踪前沿，深化理解，准备面试

| 内容 | 核心资料 |
|------|----------|
| 端侧LLM最新进展 | 跟踪MobileLLM/Phi/Qwen/MiniCPM等最新发布 |
| 量化前沿 | 关注INT2量化、混合精度、量化+蒸馏联合优化 |
| 推理优化前沿 | 关注Speculative Decoding改进、Flash Decoding演进 |
| 端侧多模态 | 关注端侧VLM（视觉语言模型）部署 |
| 综合复习 | 回顾所有知识点，形成自己的理解体系 |

---

## 五、综合实战项目建议

### 项目1：端侧LLM部署全链路（必做）

**目标**：将一个开源LLM量化部署到端侧设备

```
步骤：
1. 选择模型：Qwen-1.5B 或 Phi-3-mini
2. 量化：使用AutoGPTQ/AutoAWQ进行INT4量化
3. 转换：转换为GGUF格式
4. 部署：使用llama.cpp在PC/手机上部署
5. 评估：测量TTFT、TPS、内存占用、功耗
6. 对比：FP16 vs INT8 vs INT4的效果与性能对比
7. 报告：撰写完整的部署与评估报告
```

### 项目2：端侧场景模型适配（必做）

**目标**：针对端侧场景微调优化小模型

```
步骤：
1. 场景选择：本地问答 / 工具调用 / 设备控制
2. 数据构造：用大模型生成场景指令数据（500-2000条）
3. 数据清洗：去重、过滤、质量检查
4. SFT微调：使用LoRA对Qwen-1.5B/Phi-3进行微调
5. 量化部署：量化微调后模型并部署
6. Badcase分析：收集badcase，分析原因并迭代优化
7. 评估报告：效果对比、性能对比、问题分析
```

### 项目3：模型架构分析（推荐）

**目标**：分析不同模型架构在端侧的性能表现

```
步骤：
1. 选择3-5个不同规格的开源模型（<1B, 1-3B, 3-7B）
2. 分析架构差异（层数/头数/隐藏维度/Attention类型）
3. 对比KV Cache占用、推理速度、内存
4. 分析量化后效果变化
5. 给出端侧模型选型建议
```

---

## 六、推荐学习资料总表

### 核心必读资料（Top 15）

| 序号 | 资料名 | 类型 | 领域 | 优先级 |
|------|--------|------|------|--------|
| 1 | **MIT 6.5940: TinyML and Efficient Deep Learning Computing** | 课程 | 量化/剪枝/蒸馏/端侧部署 | ⭐⭐⭐⭐⭐ |
| 2 | **The Illustrated Transformer** (Jay Alammar) | 博客 | Transformer基础 | ⭐⭐⭐⭐⭐ |
| 3 | **nanoGPT** (Karpathy) | 开源项目 | Transformer实现 | ⭐⭐⭐⭐⭐ |
| 4 | **llama.cpp** | 开源项目 | 端侧LLM推理部署 | ⭐⭐⭐⭐⭐ |
| 5 | **Attention Is All You Need** | 论文 | Transformer基础 | ⭐⭐⭐⭐⭐ |
| 6 | **CS224N: NLP with Deep Learning** (Stanford) | 课程 | NLP/Transformer | ⭐⭐⭐⭐⭐ |
| 7 | **LLaMA-Factory** | 开源项目 | SFT/LoRA微调实践 | ⭐⭐⭐⭐⭐ |
| 8 | **Hugging Face TRL** | 开源项目 | SFT/DPO/RLHF训练 | ⭐⭐⭐⭐⭐ |
| 9 | **LoRA论文** | 论文 | 参数高效微调 | ⭐⭐⭐⭐ |
| 10 | **DPO论文** | 论文 | 偏好对齐 | ⭐⭐⭐⭐ |
| 11 | **GPTQ/AWQ论文** | 论文 | 量化技术 | ⭐⭐⭐⭐ |
| 12 | **MobileLLM论文** | 论文 | 端侧架构设计 | ⭐⭐⭐⭐ |
| 13 | **Self-Instruct** | 论文/项目 | 指令数据生成 | ⭐⭐⭐⭐ |
| 14 | **lm-evaluation-harness** | 开源项目 | LLM评估 | ⭐⭐⭐⭐ |
| 15 | **Karpathy: Let's build the GPT Tokenizer / State of GPT** | 视频 | LLM全流程理解 | ⭐⭐⭐⭐ |

### 按类型分类

#### 书籍（10本）

1. 《Deep Learning》- Ian Goodfellow et al.
2. 《Natural Language Processing with Transformers》- Hugging Face
3. 《Build a Large Language Model (From Scratch)》- Sebastian Raschka
4. 《Hands-On Machine Learning》第3版 - Aurélien Géron
5. 《TinyML》- Pete Warden et al.
6. 《Speech and Language Processing》第3版 - Jurafsky & Martin
7. 《Designing Machine Learning Systems》- Chip Huyen
8. 《Machine Learning Engineering》- Andriy Burkov
9. 《强化学习导论》- Sutton & Barto
10. 《Computer Architecture: A Quantitative Approach》- Hennessy & Patterson

#### 大学课程（8门）

1. MIT 6.5940: TinyML and Efficient Deep Learning Computing
2. Stanford CS224N: NLP with Deep Learning
3. Stanford CS25: Transformers United
4. Stanford CS324: Large Language Models
5. Stanford CS329S: Machine Learning Systems Design
6. CMU 10-418: Machine Learning Systems
7. UC Berkeley CS285: Deep Reinforcement Learning
8. Hugging Face NLP Course（在线免费）

#### 关键论文（15篇）

1. Attention Is All You Need (Transformer)
2. LLaMA: Open and Efficient Foundation Language Models
3. LoRA: Low-Rank Adaptation of Large Language Models
4. QLoRA: Efficient Finetuning of Quantized LLMs
5. Training language models to follow instructions (InstructGPT/RLHF)
6. Direct Preference Optimization (DPO)
7. GPTQ: Accurate Post-Training Quantization
8. AWQ: Activation-aware Weight Quantization
9. SmoothQuant: Accurate and Efficient PTQ
10. FlashAttention: Fast and Memory-Efficient Exact Attention
11. GQA: Training Generalized Multi-Query Transformer Models
12. MobileLLM: Optimizing Sub-billion Parameter LLMs
13. Self-Instruct: Aligning LMs with Self-Generated Instructions
14. Scaling Laws for Neural Language Models
15. Speculative Decoding

#### 开源项目（15个）

1. nanoGPT (Karpathy) - 从零实现GPT
2. llama.cpp - 端侧LLM推理
3. MLC-LLM - 跨平台端侧部署
4. LLaMA-Factory - 一站式LLM微调
5. Hugging Face TRL - SFT/DPO/RLHF训练
6. Hugging Face PEFT - LoRA/Adapter微调
7. Hugging Face Transformers - 主流LLM库
8. AutoGPTQ - GPTQ量化
9. AutoAWQ - AWQ量化
10. bitsandbytes - 8/4-bit量化
11. Self-Instruct - 指令数据生成
12. datatrove - 数据处理Pipeline
13. lm-evaluation-harness - LLM评估
14. OpenCompass - LLM评估（中文友好）
15. Unsloth - 快速LLM微调

---

> **文档说明**：本学习计划针对"AI模型-端侧大模型"岗位编制，覆盖5个专业知识领域和5个关键技能领域，共计8个知识领域、40+子领域、200+知识点。推荐资料涵盖10本书籍、8门大学课程、15篇核心论文、15个开源项目，以及大量技术博客和文档。按建议的5阶段学习路径（约30-40周），可系统掌握该岗位所需全部知识。

