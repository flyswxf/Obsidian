## 目录

- [1. 岗位概述](#1-岗位概述)
- [2. 知识体系全景图](#2-知识体系全景图)
- [3. 分领域详细知识点梳理](#3-分领域详细知识点梳理)
  - [3.1 大模型训练系统](#31-大模型训练系统)
  - [3.2 大模型推理系统](#32-大模型推理系统)
  - [3.3 高性能计算与通信](#33-高性能计算与通信)
  - [3.4 训推加速技术](#34-训推加速技术)
  - [3.5 训推系统架构设计与开发能力](#35-训推系统架构设计与开发能力)
  - [3.6 性能分析与优化能力](#36-性能分析与优化能力)
  - [3.7 软硬协同优化能力](#37-软硬协同优化能力)
- [4. 学习资料规划](#4-学习资料规划)
  - [4.1 基础前置知识资料](#41-基础前置知识资料)
  - [4.2 大模型训练系统资料](#42-大模型训练系统资料)
  - [4.3 大模型推理系统资料](#43-大模型推理系统资料)
  - [4.4 高性能计算与通信资料](#44-高性能计算与通信资料)
  - [4.5 训推加速技术资料](#45-训推加速技术资料)
  - [4.6 系统架构与工程实践资料](#46-系统架构与工程实践资料)
  - [4.7 性能分析与优化资料](#47-性能分析与优化资料)
  - [4.8 软硬协同优化资料](#48-软硬协同优化资料)
- [5. 学习路径建议](#5-学习路径建议)
- [6. 推荐学习时间表](#6-推荐学习时间表)

---

## 1. 岗位概述

### 1.1 岗位定位

AI Infra-AI训推系统岗位是AI基础设施方向的核心技术岗位，聚焦于大模型训练与推理系统的设计、开发、优化和运维。该岗位要求候选人具备从底层硬件架构到上层系统软件的全栈理解能力，能够：

- **训练侧**：设计和优化大规模分布式训练系统，解决数百卡乃至千卡级别的训练稳定性、吞吐量、显存管理等复杂工程问题
- **推理侧**：开发高性能推理引擎，优化模型服务化部署，降低延迟、提升吞吐
- **加速侧**：掌握算子级、图级、系统级的多层次加速技术，实现软硬协同优化
- **通信侧**：深入理解高性能计算通信机制，优化集群级别的计算-通信协同

### 1.2 掌握程度目标

| 等级 | 目标描述 |
|------|----------|
| **基础级（基础）** | 了解基本概念与术语，对主流框架有一般性了解 |
| **进阶级（核心目标）** | 理解核心原理与优化方法，熟悉全流程，能解决常见问题 |
| **精通级（进阶目标）** | 深入理解瓶颈与优化，有实际开发/调优经验，能系统性解决复杂问题 |

### 1.3 核心技术栈概览

```
┌─────────────────────────────────────────────────────────────┐
│                    AI训推系统技术栈                           │
├─────────────────────────────────────────────────────────────┤
│  应用层    │ 模型服务API / 负载均衡 / 多实例调度               │
├────────────┼──────────────────────────────────────────────────┤
│  系统层    │ 训练框架 / 推理引擎 / 调度系统 / 容错恢复         │
├────────────┼──────────────────────────────────────────────────┤
│  加速层    │ 算子优化 / 计算图优化 / 量化 / 蒸馏 / 编译器后端   │
├────────────┼──────────────────────────────────────────────────┤
│  通信层    │ 集合通信 / 通信拓扑 / 计算通信重叠 / RDMA/NVLink  │
├────────────┼──────────────────────────────────────────────────┤
│  硬件层    │ GPU/NPU架构 / 显存层次 / 计算单元 / 互联拓扑       │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. 知识体系全景图

```
AI Infra-AI训推系统
│
├── 1. 大模型训练系统
│   ├── 1.1 分布式训练基础
│   │   ├── 1.1.1 数据并行
│   │   ├── 1.1.2 模型并行
│   │   ├── 1.1.3 流水线并行
│   │   └── 1.1.4 张量并行
│   ├── 1.2 分布式训练策略
│   │   ├── 1.2.1 ZeRO
│   │   ├── 1.2.2 Megatron-LM
│   │   ├── 1.2.3 FSDP
│   │   └── 1.2.4 3D并行
│   ├── 1.3 训练全流程
│   │   ├── 1.3.1 预训练
│   │   ├── 1.3.2 监督微调 (SFT)
│   │   ├── 1.3.3 RLHF/RL
│   │   └── 1.3.4 对齐训练 (DPO/PPO等)
│   ├── 1.4 训练机制
│   │   ├── 1.4.1 混合精度训练
│   │   ├── 1.4.2 梯度累积
│   │   ├── 1.4.3 检查点保存与恢复
│   │   └── 1.4.4 梯度同步与AllReduce
│   ├── 1.5 训练稳定性
│   │   ├── 1.5.1 Loss Spike
│   │   ├── 1.5.2 梯度爆炸/消失
│   │   ├── 1.5.3 数值溢出 (NaN/Inf)
│   │   └── 1.5.4 稳定性应对策略
│   ├── 1.6 大规模训练运维
│   │   ├── 1.6.1 容错恢复
│   │   ├── 1.6.2 弹性调度
│   │   ├── 1.6.3 集群资源管理
│   │   └── 1.6.4 训练监控与告警
│   └── 1.7 训练框架底层实现
│       ├── 1.7.1 PyTorch Distributed
│       ├── 1.7.2 MindSpore分布式机制
│       ├── 1.7.3 DeepSpeed引擎
│       └── 1.7.4 Megatron-LM引擎
│
├── 2. 大模型推理系统
│   ├── 2.1 推理基础流程
│   │   ├── 2.1.1 模型加载
│   │   ├── 2.1.2 输入预处理
│   │   ├── 2.1.3 推理计算
│   │   └── 2.1.4 输出后处理
│   ├── 2.2 推理优化技术
│   │   ├── 2.2.1 KV Cache管理
│   │   ├── 2.2.2 PagedAttention
│   │   ├── 2.2.3 Continuous Batching
│   │   ├── 2.2.4 Prefix Caching
│   │   └── 2.2.5 Speculative Decoding
│   ├── 2.3 推理加速方法
│   │   ├── 2.3.1 量化推理 (INT8/INT4)
│   │   ├── 2.3.2 Tensor并行推理
│   │   ├── 2.3.3 算子融合推理
│   │   └── 2.3.4 稀疏化推理
│   ├── 2.4 推理服务化架构
│   │   ├── 2.4.1 模型服务API
│   │   ├── 2.4.2 负载均衡
│   │   ├── 2.4.3 多实例调度
│   │   └── 2.4.4 自动扩缩容
│   ├── 2.5 推理性能指标
│   │   ├── 2.5.1 首字延迟 (TTFT)
│   │   ├── 2.5.2 生成速度 (TPS)
│   │   ├── 2.5.3 吞吐量
│   │   └── 2.5.4 显存占用
│   └── 2.6 推理引擎
│       ├── 2.6.1 vLLM
│       ├── 2.6.2 TensorRT-LLM
│       ├── 2.6.3 TGI (Text Generation Inference)
│       ├── 2.6.4 MindSpore Lite
│       └── 2.6.5 llama.cpp
│
├── 3. 高性能计算与通信
│   ├── 3.1 GPU/NPU架构
│   │   ├── 3.1.1 SM/AICore结构
│   │   ├── 3.1.2 显存层次
│   │   ├── 3.1.3 计算单元
│   │   └── 3.1.4 内存带宽与计算吞吐
│   ├── 3.2 集合通信
│   │   ├── 3.2.1 AllReduce
│   │   ├── 3.2.2 AllGather
│   │   ├── 3.2.3 ReduceScatter
│   │   ├── 3.2.4 Broadcast
│   │   └── 3.2.5 Send/Recv
│   ├── 3.3 通信库与协议
│   │   ├── 3.3.1 NCCL
│   │   ├── 3.3.2 HCCL (Collective Communication Library)
│   │   ├── 3.3.3 MPI
│   │   └── 3.3.4 Gloo
│   ├── 3.4 通信拓扑与算法
│   │   ├── 3.4.1 环形通信
│   │   ├── 3.4.2 树形通信
│   │   ├── 3.4.3 拓扑感知调度
│   │   └── 3.4.4 层次化通信
│   ├── 3.5 高速互联技术
│   │   ├── 3.5.1 NVLink
│   │   ├── 3.5.2 RDMA (InfiniBand/RoCE)
│   │   ├── 3.5.3 PCIe
│   │   └── 3.5.4 HCCS (Cache Coherent System)
│   ├── 3.6 计算通信重叠
│   │   ├── 3.6.1 通信-计算Overlap策略
│   │   ├── 3.6.2 异步通信
│   │   └── 3.6.3 通信调度优化
│   └── 3.7 并行计算模型
│       ├── 3.7.1 SIMD/SIMT
│       ├── 3.7.2 并行度与 occupancy
│       └── 3.7.3 Roofline模型
│
├── 4. 训推加速技术
│   ├── 4.1 量化技术
│   │   ├── 4.1.1 INT8量化
│   │   ├── 4.1.2 INT4量化
│   │   ├── 4.1.3 FP8量化
│   │   ├── 4.1.4 训练后量化 (PTQ)
│   │   ├── 4.1.5 量化感知训练 (QAT)
│   │   └── 4.1.6 混合精度量化
│   ├── 4.2 知识蒸馏
│   │   ├── 4.2.1 教师-学生模型
│   │   ├── 4.2.2 软标签蒸馏
│   │   ├── 4.2.3 特征蒸馏
│   │   └── 4.2.4 在线蒸馏
│   ├── 4.3 算子优化
│   │   ├── 4.3.1 算子融合
│   │   ├── 4.3.2 算子自动调优 (AutoTuning)
│   │   ├── 4.3.3 自定义CUDA/AscendC算子开发
│   │   └── 4.3.4 算子性能分析
│   ├── 4.4 计算图优化
│   │   ├── 4.4.1 常量折叠
│   │   ├── 4.4.2 死代码消除
│   │   ├── 4.4.3 算子重排
│   │   ├── 4.4.4 子图替换
│   │   └── 4.4.5 内存复用
│   ├── 4.5 Flash Attention
│   │   ├── 4.5.1 Flash Attention v1/v2/v3
│   │   ├── 4.5.2 IO感知注意力
│   │   └── 4.5.3 分块计算
│   ├── 4.6 显存优化
│   │   ├── 4.6.1 梯度检查点
│   │   ├── 4.6.2 激活值重计算
│   │   ├── 4.6.3 显存池化
│   │   └── 4.6.4 显存碎片管理
│   ├── 4.7 自动混合精度 (AMP)
│   │   ├── 4.7.1 FP16/BF16训练
│   │   ├── 4.7.2 损失缩放
│   │   └── 4.7.3 精度-性能权衡
│   └── 4.8 编译器后端优化
│       ├── 4.8.1 XLA (Accelerated Linear Algebra)
│       ├── 4.8.2 TVM
│       ├── 4.8.3 Triton
│       └── 4.8.4 图编译器
│
├── 5. 训推系统架构设计与开发能力
│   ├── 5.1 系统架构设计
│   │   ├── 5.1.1 分布式系统架构模式
│   │   ├── 5.1.2 训推一体化架构
│   │   └── 5.1.3 微服务架构
│   ├── 5.2 模块设计与开发
│   │   ├── 5.2.1 调度模块设计
│   │   ├── 5.2.2 通信模块设计
│   │   ├── 5.2.3 显存管理模块设计
│   │   └── 5.2.4 容错模块设计
│   └── 5.3 工程实践
│       ├── 5.3.1 大规模代码工程组织
│       ├── 5.3.2 CI/CD for AI系统
│       └── 5.3.3 性能基准测试体系
│
├── 6. 性能分析与优化能力
│   ├── 6.1 性能分析工具
│   │   ├── 6.1.1 Nsight Systems/Compute
│   │   ├── 6.1.2 PyTorch Profiler
│   │   ├── 6.1.3 MindSpore Profiler
│   │   └── 6.1.4 自定义性能采集
│   ├── 6.2 性能瓶颈分析
│   │   ├── 6.2.1 计算瓶颈识别
│   │   ├── 6.2.2 通信瓶颈识别
│   │   ├── 6.2.3 显存瓶颈识别
│   │   └── 6.2.4 I/O瓶颈识别
│   ├── 6.3 优化方法论
│   │   ├── 6.3.1 Roofline分析
│   │   ├── 6.3.2 Amdahl定律
│   │   ├── 6.3.3 强/弱扩展性分析
│   │   └── 6.3.4 瓶颈转移分析
│   └── 6.4 性能建模
│       ├── 6.4.1 计算量建模
│       ├── 6.4.2 通信量建模
│       └── 6.4.3 显存占用建模
│
└── 7. 软硬协同优化能力
    ├── 7.1 硬件特性理解
    │   ├── 7.1.1 GPU架构特性
    │   ├── 7.1.2 NPU/Ascend架构特性
    │   └── 7.1.3 异构计算特性
    ├── 7.2 算子级协同优化
    │   ├── 7.2.1 CUDA Kernel优化
    │   ├── 7.2.2 AscendC算子开发
    │   └── 7.2.3 算子映射到硬件
    ├── 7.3 系统级协同优化
    │   ├── 7.3.1 数据布局优化
    │   ├── 7.3.2 内存访问模式优化
    │   ├── 7.3.3 流水线调度
    │   └── 7.3.4 资源分配优化
    └── 7.4 跨平台适配
        ├── 7.4.1 多硬件后端适配
        ├── 7.4.2 性能可移植性
        └── 7.4.3 硬件特性抽象
```

---

## 3. 分领域详细知识点梳理

### 3.1 大模型训练系统

#### 3.1.1 分布式训练基础

**数据并行**
- 原理：将数据批次分割到多个设备，每个设备持有完整模型副本
- 梯度同步：通过AllReduce实现梯度聚合
- 同步SGD vs 异步SGD
- Data Parallel vs Distributed Data Parallel (DDP)
- 关键问题：通信开销随设备数线性增长

**模型并行**
- 原理：将模型参数分割到多个设备
- 适用场景：模型过大无法放入单卡显存
- 层内并行（Intra-layer）vs 层间并行（Inter-layer）
- 通信模式：前向传播中的Send/Recv

**流水线并行**
- 原理：将模型按层切分到不同设备，形成流水线
- GPipe方案：micro-batch切分 + 气泡填充
- 1F1B (One Forward One Backward) 调度
- Interleaved 1F1B调度
- 流水线气泡分析：bubble fraction = (P-1) / (P+M-1)
- 关键问题：气泡开销、显存占用不均衡

**张量并行**
- 原理：将单个算子（如矩阵乘法）的权重矩阵切分到多个设备
- Column Parallel（列切分）：W = [W_1, W_2, ..., W_n]
- Row Parallel（行切分）：W = [W_1; W_2; ...; W_n]
- MLP层的张量并行：Column Parallel → Row Parallel
- Attention层的张量并行：多头注意力天然适合切分
- 通信模式：前向AllReduce / AllGather，反向AllReduce
- Megatron-LM的张量并行方案

#### 3.1.2 分布式训练策略

**ZeRO (Zero Redundancy Optimizer)**
- ZeRO-1：优化器状态切分
- ZeRO-2：优化器状态 + 梯度切分
- ZeRO-3：优化器状态 + 梯度 + 模型参数切分
- 显存节省分析：ZeRO-3可将显存占用降至1/N
- 通信开销分析：ZeRO-3引入额外的AllGather通信
- ZeRO-Infinity：利用CPU/NVMe存储扩展显存
- ZeRO-Offload：将优化器状态卸载到CPU

**Megatron-LM**
- 3D并行：数据并行 × 张量并行 × 流水线并行
- 张量并行：Megatron的Column/Row Parallel Linear
- 流水线并行：Interleaved 1F1B
- 通信优化：通信-计算重叠
- 适用场景：千亿参数模型训练

**FSDP (Fully Sharded Data Parallel)**
- PyTorch原生的ZeRO-3实现
- 参数分片：前向前AllGather，反向后ReduceScatter
- 混合精度支持
- CPU Offload支持
- 与DDP的兼容性

**3D并行**
- 维度组合：DP × TP × PP
- 典型配置：DP=8, TP=8, PP=4 (256 GPU)
- 通信域划分：TP通信域（节点内NVLink），DP通信域（跨节点），PP通信域（跨节点）
- 通信量分析：TP通信（每层），PP通信（每stage），DP通信（每step）
- 资源分配策略：TP优先放节点内，PP跨节点，DP填充剩余

#### 3.1.3 训练全流程

**预训练**
- 数据预处理：Tokenization, BPE/SentencePiece
- 训练目标：自回归语言建模（Causal LM）/ 掩码语言建模（MLM）
- 学习率调度：Warmup + Cosine Decay
- 批次大小策略：渐进增大
- 数据配比与课程学习
- 训练监控：Loss曲线、梯度范数、学习率

**监督微调 (SFT)**
- 指令数据格式：Instruction-Input-Output
- 全参数微调 vs 参数高效微调（PEFT）
- LoRA (Low-Rank Adaptation)：W = W₀ + BA
- QLoRA：量化基础上的LoRA
- 训练数据质量与多样性

**RLHF/RL (人类反馈强化学习)**
- RLHF三阶段：SFT → 奖励模型训练 → PPO强化学习
- 奖励模型：Pairwise ranking loss
- PPO训练：策略模型 + 参考模型 + 奖励模型 + 价值模型
- 关键挑战：训练稳定性、KL散度控制、奖励黑客
- 替代方案：DPO (Direct Preference Optimization)
- GRPO (Group Relative Policy Optimization)

**对齐训练**
- DPO：直接偏好优化，无需奖励模型
- IPO (Identity Preference Optimization)
- KTO (Kahneman-Tversky Optimization)
- Constitutional AI方法
- Safety训练

#### 3.1.4 训练机制

**混合精度训练**
- FP16/BF16训练原理
- 主权重（FP32）+ 前向/反向（FP16/BF16）
- 损失缩放：动态损失缩放 vs 静态损失缩放
- BF16优势：无需损失缩放，更大动态范围
- 精度问题：哪些层保持FP32（如LayerNorm）

**梯度累积**
- 原理：多次前向反向累积梯度后统一更新
- 目的：模拟更大batch size
- 实现方式：梯度累积步数 = 目标batch / 实际batch
- 注意事项：梯度同步时机、BatchNorm统计

**检查点保存与恢复**
- 模型状态保存：参数、优化器状态、学习率调度器
- 分布式检查点：分片保存、并行I/O
- 异步检查点：训练与保存重叠
- 检查点压缩与增量保存
- 恢复策略：从最近检查点恢复 vs 指定step恢复

**梯度同步与AllReduce**
- 梯度同步时机：反向传播过程中逐层同步 vs 反向结束后统一同步
- Bucket化梯度同步：将梯度分桶，桶满即同步
- 梯度同步与反向计算重叠
- 梯度累积时的同步控制

#### 3.1.5 训练稳定性

**Loss Spike**
- 现象：训练过程中loss突然大幅上升
- 原因：学习率过大、数据异常、梯度爆炸、数值不稳定
- 检测方法：loss监控、梯度范数监控
- 应对策略：回滚到稳定检查点、降低学习率、跳过异常batch

**梯度爆炸/消失**
- 梯度爆炸：梯度裁剪（Gradient Clipping）— 按范数裁剪
- 梯度消失：残差连接、LayerNorm位置选择
- 初始化策略：Xavier/He初始化
- 激活函数选择对梯度的影响

**数值溢出 (NaN/Inf)**
- FP16溢出：最大值65504，容易溢出
- BF16优势：动态范围与FP32相同
- 检测方法：梯度/激活值范围监控
- 应对策略：降低学习率、切换BF16、增加损失缩放

**稳定性应对策略**
- 学习率warmup：线性warmup避免初期不稳定
- 梯度裁剪：限制梯度范数
- 权重衰减：L2正则化
- EMA (Exponential Moving Average)：平滑权重更新
- 训练监控体系：实时监控关键指标

#### 3.1.6 大规模训练运维

**容错恢复**
- 节点故障检测：心跳检测、健康检查
- 自动恢复：从检查点恢复训练
- 恢复优化：快速检查点加载、并行恢复
- 故障隔离：故障节点排除，剩余节点继续训练

**弹性调度**
- 动态资源分配：根据可用资源调整并行度
- 训练中断与续训
- 优先级调度：多任务资源竞争
- Spot实例利用

**集群资源管理**
- Kubernetes/Slurm调度
- GPU/NPU资源分配
- 网络资源管理
- 存储资源管理

**训练监控与告警**
- 实时监控：Loss、梯度、吞吐、显存
- 告警机制：异常检测、阈值告警
- 日志聚合与分析
- 可视化看板（TensorBoard/WandB）

#### 3.1.7 训练框架底层实现

**PyTorch Distributed**
- torch.distributed架构
- Process Group与Backend (NCCL/Gloo)
- DDP实现原理：梯度Bucket化、自动同步
- RPC框架：远程调用支持模型并行
- DTensor / DeviceMesh：PyTorch原生分布式抽象

**MindSpore分布式机制**
- 自动并行：数据并行 + 模型并行自动搜索
- 半自动并行：用户配置切分策略
- 算子级切分
- 通信算子自动插入
- Ascend平台优化

**DeepSpeed引擎**
- ZeRO系列实现
- 3D并行编排
- 混合精度引擎
- 检查点管理
- 数据加载优化

**Megatron-LM引擎**
- 模型定义：MegatronParallelModule
- 张量并行：ColumnParallelLinear, RowParallelLinear
- 流水线并行：PipelineEngine
- 通信优化：通信-计算重叠

---

### 3.2 大模型推理系统

#### 3.2.1 推理基础流程

**模型加载**
- 模型权重加载到设备显存
- 模型量化加载（量化权重直接加载）
- 多设备分布加载（张量并行/流水线并行推理）
- 模型预热（Warmup）

**输入预处理**
- Tokenization：BPE/SentencePiece
- 特殊Token处理
- 上下文长度管理
- Prompt模板

**推理计算**
- 前向传播
- 自回归生成：逐Token生成
- KV Cache维护
- 采样策略：Greedy、Top-k、Top-p (nucleus)、Temperature

**输出后处理**
- Detokenization
- 特殊Token过滤
- 停止条件检测
- 流式输出处理

#### 3.2.2 推理优化技术

**KV Cache管理**
- 原理：缓存已计算的Key/Value，避免重复计算
- 显存占用分析：KV Cache大小 = 2 × n_layers × n_heads × head_dim × seq_len × batch_size × dtype_size
- 显存碎片问题：连续分配 vs 动态分配
- KV Cache驱逐策略

**PagedAttention**
- 原理：借鉴OS虚拟内存管理，将KV Cache按固定大小的Block管理
- Block Table：逻辑Block到物理Block的映射
- 解决显存碎片问题
- 支持变长序列高效管理
- 实现框架：vLLM
- 性能提升：减少显存浪费，提升batch size

**Continuous Batching**
- 原理：动态将新请求加入当前batch，已完成请求退出
- 对比静态Batching：避免短序列等待长序列
- Iteration-level调度：每次生成Token时重新组batch
- 性能提升：显著提高GPU利用率

**Prefix Caching**
- 原理：缓存共享前缀的KV Cache
- 适用场景：多轮对话、Few-shot提示
- 自动前缀检测与复用
- 显存管理：LRU淘汰策略

**Speculative Decoding**
- 原理：用小模型快速生成候选Token，大模型验证
- Draft Model + Target Model
- 接受/拒绝采样：保证输出分布不变
- 投机解码变体：Medusa头、EAGLE、Lookahead Decoding
- 性能提升：减少大模型前向次数

#### 3.2.3 推理加速方法

**量化推理 (INT8/INT4)**
- 权重量化：INT8/INT4权重量化
- KV Cache量化：INT8 KV Cache
- 激活值量化挑战
- 量化感知推理
- 性能-精度权衡

**Tensor并行推理**
- 推理时的张量并行
- 通信开销：每层AllReduce
- 适合大模型推理（>70B）
- 通信优化：流水线通信

**算子融合推理**
- 融合模式：LayerNorm + Linear, Attention融合
- 减少Kernel Launch开销
- 减少中间结果显存访问
- Flash Attention在推理中的应用

**稀疏化推理**
- 结构化剪枝推理
- MoE (Mixture of Experts)推理
- 稀疏注意力推理

#### 3.2.4 推理服务化架构

**模型服务API**
- REST API vs gRPC
- 流式输出接口（SSE/WebSocket）
- 批处理API
- 异步推理API

**负载均衡**
- 请求路由策略：Round-robin、Least-loaded
- 跨实例负载均衡
- 基于队列长度的调度
- 延迟感知路由

**多实例调度**
- 多模型实例管理
- GPU资源分配
- 模型热加载/热切换
- 优先级队列

**自动扩缩容**
- 基于QPS/延迟的自动扩缩
- 基于GPU利用率的扩缩
- 冷启动优化
- 预测性扩缩

#### 3.2.5 推理性能指标

**首字延迟 (TTFT - Time To First Token)**
- 定义：从请求到达到生成第一个Token的延迟
- 影响因素：Prompt长度、Prefill计算量、队列等待时间
- 优化方向：Prefill优化、调度优化

**生成速度 (TPS - Tokens Per Second)**
- 定义：生成阶段每秒输出Token数
- 影响因素：Batch Size、KV Cache访问效率、计算效率
- 优化方向：Decode优化、Batch优化

**吞吐量**
- 定义：单位时间处理的请求数/Token数
- 与延迟的权衡
- 最大吞吐 vs SLA约束吞吐

**显存占用**
- 模型权重显存
- KV Cache显存
- 激活值显存
- 显存利用率优化

#### 3.2.6 推理引擎

**vLLM**
- PagedAttention实现
- Continuous Batching
- 张量并行推理
- Prefix Caching
- 支持多种模型架构

**TensorRT-LLM**
- NVIDIA官方推理引擎
- 算子融合优化
- INT8/FP8量化推理
- In-flight Batching
- Plugin机制

**TGI (Text Generation Inference)**
- HuggingFace推理引擎
- Continuous Batching
- 张量并行
- 量化支持

**MindSpore Lite**
- Ascend推理引擎
- 量化推理
- 算子优化
- 端侧推理支持

**llama.cpp**
- CPU/GPU混合推理
- GGML/GGUF格式
- 极低资源推理
- 量化推理

---

### 3.3 高性能计算与通信

#### 3.3.1 GPU/NPU架构

**SM/AICore结构**
- GPU SM (Streaming Multiprocessor)：CUDA Core、Tensor Core、共享内存、寄存器文件
- NPU AICore：AI Core结构、Cube/Vector单元
- Warp概念：32线程为一组（GPU）
- 硬件资源限制：最大线程数、共享内存大小、寄存器数量

**显存层次**
- 寄存器（Registers）：最快，容量最小（~256KB/SM）
- 共享内存（Shared Memory）：~228KB/SM，需要显式管理
- L1/L2 Cache：硬件自动管理
- HBM (High Bandwidth Memory)：大容量，~80GB，带宽~3TB/s
- 内存层次对性能的影响

**计算单元**
- CUDA Core：通用计算单元
- Tensor Core：矩阵乘加加速单元（FP16/INT8/INT4）
- Tensor Core矩阵尺寸：16×16×16 (Volta), 16×8×16 (Ampere FP32)
- 计算吞吐量：TFLOPS分析
- NPU Cube单元：矩阵乘加速

**内存带宽与计算吞吐**
- 计算强度（Arithmetic Intensity）：FLOPs/Byte
- Roofline模型：性能上限 = min(峰值计算, 峰值带宽 × 计算强度)
- 计算密集型 vs 带宽密集型
- GEMM（矩阵乘法）是计算密集型，Attention是带宽密集型

#### 3.3.2 集合通信

**AllReduce**
- 语义：所有设备上的张量求和后广播到所有设备
- Ring AllReduce算法：两阶段（ReduceScatter + AllGather）
- 通信量：2(N-1)/N × 数据量（N为设备数）
- 适用场景：数据并行梯度同步

**AllGather**
- 语义：收集所有设备上的张量，拼接后广播到所有设备
- 通信量：(N-1)/N × 数据量
- 适用场景：ZeRO-3前向参数收集

**ReduceScatter**
- 语义：求和后分散到各设备（每个设备获得部分结果）
- 通信量：(N-1)/N × 数据量
- 适用场景：ZeRO-3反向梯度分散

**Broadcast**
- 语义：一个设备的数据广播到所有设备
- 通信量：数据量
- 适用场景：参数初始化、模型加载

**Send/Recv**
- 语义：点对点通信
- 适用场景：流水线并行相邻Stage通信
- 同步 vs 异步Send/Recv

#### 3.3.3 通信库与协议

**NCCL (NVIDIA Collective Communications Library)**
- NVIDIA GPU集合通信库
- 支持AllReduce、AllGather、ReduceScatter、Broadcast等
- Ring算法与Tree算法
- 自动拓扑发现与优化
- NVLink/InfiniBand自动选择
- NCCL Net Plugin：自定义网络插件

**HCCL (Collective Communication Library)**
- Ascend NPU集合通信库
- 与NCCL类似的接口设计
- 针对Ascend硬件优化
- HCCS互联支持

**MPI (Message Passing Interface)**
- 标准消息传递接口
- MPI_Allreduce、MPI_Bcast等
- CPU通信常用
- 与NCCL/HCCL的对比

**Gloo**
- PyTorch默认CPU通信后端
- 支持TCP/共享内存
- 用于CPU分布式训练

#### 3.3.4 通信拓扑与算法

**环形通信**
- Ring AllReduce：每个节点与左右邻居通信
- 通信步数：2(N-1)步
- 每步通信量：数据量/N
- 总通信量：2(N-1)/N × 数据量
- 优点：带宽利用率高，无瓶颈节点

**树形通信**
- Recursive Doubling算法
- 通信步数：log(N)步
- 适合小消息通信
- 根节点带宽瓶颈

**拓扑感知调度**
- 节点内通信优先（NVLink/HCCS带宽高）
- 跨节点通信最小化
- 层次化AllReduce：节点内Ring → 节点间Ring
- NCCL的拓扑感知ring构建

**层次化通信**
- 两层AllReduce：节点内ReduceScatter + 节点间AllReduce + 节点内AllGather
- 带宽利用优化：充分利用节点内高带宽
- 通信域分层：local group + global group

#### 3.3.5 高速互联技术

**NVLink**
- NVIDIA GPU间高速互联
- 带宽：NVLink 3.0 ~600GB/s, NVLink 4.0 ~900GB/s
- 对比PCIe 4.0 ~64GB/s
- 拓扑：NVSwitch全互联 vs Direct NVLink
- 对集合通信性能的影响

**RDMA (Remote Direct Memory Access)**
- 绕过CPU，直接内存访问
- InfiniBand：~200-400 Gb/s, 微秒级延迟
- RoCE (RDMA over Converged Ethernet)
- GPUDirect RDMA：GPU显存直接RDMA
- 对训练通信性能的关键作用

**PCIe**
- PCIe 4.0: ~64GB/s (16x)
- PCIe 5.0: ~128GB/s (16x)
- GPU-CPU通信瓶颈
- PCIe Switch拓扑

**HCCS (Cache Coherent System)**
- Ascend NPU互联技术
- 片间高速互联
- Cache一致性
- 对HCCL通信性能的支撑

#### 3.3.6 计算通信重叠

**通信-计算Overlap策略**
- 反向传播中梯度同步与计算重叠
- Bucket化梯度同步：梯度准备好即开始同步
- 流水线并行中的通信-计算重叠
- 张量并行中的通信隐藏

**异步通信**
- 异步AllReduce：发起后不阻塞
- CUDA Stream分离：计算Stream + 通信Stream
- 事件同步：确保数据依赖
- PyTorch中的异步通信API

**通信调度优化**
- 通信操作重排：减少通信等待
- 通信融合：合并多个小通信为一个大通信
- 通信量化：减少通信数据量
- 梯度压缩：Top-K稀疏化、量化

#### 3.3.7 并行计算模型

**SIMD/SIMT**
- SIMD (Single Instruction Multiple Data)：CPU向量化
- SIMT (Single Instruction Multiple Thread)：GPU执行模型
- Warp执行：32线程同步执行
- 分支分歧对性能的影响

**并行度与Occupancy**
- Occupancy = 活跃Warp数 / 最大Warp数
- 影响因素：寄存器使用、共享内存使用
- 高Occupancy不等于高性能（需考虑指令级并行）
- Occupancy计算器

**Roofline模型**
- 性能上限 = min(峰值计算吞吐, 峰值带宽 × 计算强度)
- 计算强度 = FLOPs / Bytes
- Roofline图：横轴计算强度，纵轴性能
- 瓶颈识别：计算密集型 vs 带宽密集型
- 算子优化方向判断

---

### 3.4 训推加速技术

#### 3.4.1 量化技术

**INT8量化**
- 对称量化 vs 非对称量化
- 量化公式：q = round(r/scale + zero_point)
- Per-tensor vs Per-channel量化
- 校准方法：Min-Max、Percentile、MSE、ACIQ
- INT8推理性能：2-4倍加速

**INT4量化**
- 更激进的量化：4bit表示权重
- GPTQ：基于二阶信息的量化
- AWQ (Activation-aware Weight Quantization)
- 4bit推理的精度挑战
- 分组量化

**FP8量化**
- E4M3 vs E5M2格式
- FP8训练：前向E4M3，反向E5M2
- FP8推理优势：硬件原生支持（Hopper架构）
- 精度-性能权衡

**训练后量化 (PTQ)**
- 无需重新训练
- 校准数据集选择
- 逐层量化 vs 全局量化
- 精度损失评估

**量化感知训练 (QAT)**
- 训练中模拟量化效果
- Straight-Through Estimator (STE)
- 量化噪声注入
- 精度恢复

**混合精度量化**
- 敏感层保持高精度
- 非敏感层使用低精度
- 自动混合精度策略
- 精度-性能自动搜索

#### 3.4.2 知识蒸馏

**教师-学生模型**
- 大模型（教师）指导小模型（学生）
- 软标签 vs 硬标签
- 温度参数T对软标签的影响
- 蒸馏损失设计

**软标签蒸馏**
- KL散度损失
- L_soft = T² × KL(p_teacher || p_student)
- 软标签包含暗知识

**特征蒸馏**
- 中间层特征对齐
- FitNets方法
- 注意力图蒸馏
- 隐藏层表示蒸馏

**在线蒸馏**
- 多模型同时训练互相蒸馏
- 自蒸馏
- 深度相互学习

#### 3.4.3 算子优化

**算子融合**
- 融合策略：相邻算子合并
- 常见融合：Bias+Act、LayerNorm+Linear、Attention融合
- 融合收益：减少Kernel Launch开销、减少中间结果读写
- 融合限制：数据依赖、显存限制
- 手动融合 vs 自动融合

**算子自动调优**
- AutoTuning原理：搜索最优Kernel配置
- 搜索空间：Block大小、Tile大小、展开因子等
- TVM AutoScheduler
- Triton autotune
- 编译时调优 vs 运行时调优

**自定义CUDA/AscendC算子开发**
- CUDA Kernel开发流程
- AscendC算子开发流程
- 性能优化技巧：共享内存使用、Bank Conflict消除、向量化访存、指令流水
- 算子性能分析
- 算子正确性验证

**算子性能分析**
- Nsight Compute：单算子分析
- Roofline分析
- 内存带宽利用率
- 计算利用率
- Warp Stall分析

#### 3.4.4 计算图优化

**常量折叠**
- 编译时计算常量表达式
- 减少运行时计算
- 示例：常量矩阵乘法在编译时完成

**死代码消除**
- 移除输出不被使用的算子
- 减少不必要的计算
- 图遍历标记可达节点

**算子重排**
- 等价变换：改变算子顺序减少计算量
- 示例：Transpose + MatMul → MatMul + Transpose
- 代数简化规则

**子图替换**
- 模式匹配：识别可替换的子图
- 替换为更高效的等价子图
- 硬件特定子图优化

**内存复用**
- 分析张量生命周期
- 重用已释放的显存
- 减少显存峰值
- 静态内存规划 vs 动态内存池

#### 3.4.5 Flash Attention

**Flash Attention v1/v2/v3**
- v1核心思想：分块计算Attention，减少HBM访问
- IO复杂度：O(N²d²/M) vs 标准Attention的O(N²d)，M为SRAM大小
- 在线Softmax：分块计算Softmax的数学等价
- v2改进：减少非矩阵乘法计算、优化并行度
- v3改进：利用FP16 Tensor Core、异步数据搬运
- 性能提升：2-4倍加速，显存从O(N²)降至O(N)

**IO感知注意力**
- 标准Attention的瓶颈：HBM带宽
- 分块计算：Q×K^T分块，Softmax分块
- 重计算策略：反向传播时重计算Attention（省显存）
- SRAM利用：中间结果保持在SRAM

**分块计算**
- Q, K, V分块加载到SRAM
- 分块计算Attention Score
- 分块Softmax与累加
- 分块加权求和

#### 3.4.6 显存优化

**梯度检查点**
- 原理：前向传播只保存部分激活值，反向传播时重计算
- 显存-计算权衡：用计算换显存
- Checkpoint策略：每n层保存一次
- 梯度检查点与混合精度结合

**激活值重计算**
- 选择性重计算：只重计算显存大、计算快的层
- Attention重计算 vs MLP重计算
- 重计算与Flash Attention结合

**显存池化**
- 预分配大块显存
- 自定义内存分配器
- 减少显存碎片
- PyTorch的CachingAllocator

**显存碎片管理**
- 内部碎片：分配块大于请求
- 外部碎片：空闲块不连续
- 显存碎片整理
- PagedAttention的Block管理

#### 3.4.7 自动混合精度 (AMP)

**FP16/BF16训练**
- FP16：5位指数 + 10位尾数，范围±65504
- BF16：8位指数 + 7位尾数，范围同FP32
- FP16需要损失缩放，BF16不需要
- 精度敏感性分析

**损失缩放**
- 动态损失缩放：自动调整scale factor
- 静态损失缩放：固定scale factor
- 溢出检测与处理
- 损失缩放对梯度的影响

**精度-性能权衡**
- 哪些操作用FP16/BF16：矩阵乘法、卷积
- 哪些操作保持FP32：Softmax、LayerNorm、累加
- 精度损失评估
- 混合精度训练最佳实践

#### 3.4.8 编译器后端优化

**XLA (Accelerated Linear Algebra)**
- TensorFlow/JAX的编译器后端
- 算子融合、内存布局优化
- JIT编译
- HLO IR

**TVM**
- 深度学习编译器框架
- Relay IR + AutoTVM/AutoScheduler
- 多硬件后端支持
- 算子级和图级优化

**Triton**
- OpenAI开发的GPU编程语言
- 类Python语法编写GPU Kernel
- 自动分块和向量化
- Flash Attention的Triton实现
- 算子开发效率高

**图编译器**
- 计算图IR设计
- 图级优化Pass
- 硬件特定代码生成
- PyTorch的TorchDynamo + TorchInductor
- MindSpore的图编译

---

### 3.5 训推系统架构设计与开发能力

#### 3.5.1 系统架构设计

**分布式系统架构模式**
- 主从架构：Parameter Server
- 去中心化架构：Ring AllReduce
- 混合架构：3D并行编排
- 架构选择考量：通信模式、容错需求、扩展性

**训推一体化架构**
- 训练-推理权重转换
- 动态图→静态图转换
- 训推一致性问题
- 在线学习架构

**微服务架构**
- 推理服务拆分
- API Gateway设计
- 服务发现与注册
- 配置中心

#### 3.5.2 模块设计与开发

**调度模块设计**
- 任务调度器
- 资源分配器
- 优先级队列管理
- 调度算法实现

**通信模块设计**
- 通信原语封装
- 通信域管理
- 通信拓扑构建
- 通信错误处理

**显存管理模块设计**
- 显存分配器
- 显存池设计
- 显存监控
- OOM处理

**容错模块设计**
- 故障检测
- 状态保存与恢复
- 自动重试机制
- 故障隔离

#### 3.5.3 工程实践

**大规模代码工程组织**
- 模块化设计
- 接口设计
- 代码规范
- 测试覆盖

**CI/CD for AI系统**
- 自动化测试
- 性能回归测试
- 模型精度回归
- 部署自动化

**性能基准测试体系**
- Micro-benchmark
- Macro-benchmark
- 性能基线管理
- 性能回归检测

---

### 3.6 性能分析与优化能力

#### 3.6.1 性能分析工具

**Nsight Systems/Compute**
- Nsight Systems：系统级性能分析
- Nsight Compute：Kernel级性能分析
- Timeline分析：CPU-GPU时间线
- Kernel性能指标：计算利用率、内存带宽利用率

**PyTorch Profiler**
- CPU/GPU活动追踪
- 内存分析
- 通信分析
- TensorBoard可视化

**MindSpore Profiler**
- Ascend性能分析
- 算子级性能分析
- 通信性能分析
- 内存分析

**自定义性能采集**
- 自定义计时器
- 性能事件记录
- 分布式Trace
- 性能数据聚合与分析

#### 3.6.2 性能瓶颈分析

**计算瓶颈识别**
- GPU利用率低 → 计算瓶颈
- Kernel执行时间分析
- 算子热点识别
- 计算强度分析

**通信瓶颈识别**
- 通信时间占比分析
- 通信带宽利用率
- 通信延迟分析
- 通信-计算重叠度

**显存瓶颈识别**
- 显存利用率
- 显存碎片
- 显存带宽利用率
- OOM分析

**I/O瓶颈识别**
- 数据加载瓶颈
- 检查点I/O瓶颈
- 网络I/O瓶颈
- 存储带宽分析

#### 3.6.3 优化方法论

**Roofline分析**
- 绘制Roofline图
- 标记各算子在图上的位置
- 识别瓶颈类型
- 确定优化方向

**Amdahl定律**
- 加速比 = 1 / ((1-P) + P/N)
- P为可并行比例，N为并行度
- 识别串行瓶颈
- 优化优先级确定

**强/弱扩展性分析**
- 强扩展性：固定问题规模，增加设备数
- 弱扩展性：固定每设备问题规模，增加设备数
- 理想扩展性曲线
- 扩展性瓶颈识别

**瓶颈转移分析**
- 优化一个瓶颈后，瓶颈转移
- 迭代优化策略
- 全局优化 vs 局部优化

#### 3.6.4 性能建模

**计算量建模**
- FLOPs计算：矩阵乘法 2×M×N×K
- Attention计算量：2×seq_len²×d_model
- MLP计算量：2×seq_len×d_model×d_ff
- 总FLOPs = 6×N×T（N参数量，T Token数）

**通信量建模**
- 数据并行通信量：2(N-1)/N × 参数量 × dtype_size
- 张量并行通信量：2 × n_layers × seq_len × batch × d_model × dtype_size
- 流水线并行通信量：n_stages × seq_len × batch × d_model × dtype_size
- 3D并行总通信量分析

**显存占用建模**
- 模型权重：N × dtype_size
- 优化器状态：N × 12 (Adam, FP32) / N × 4 (ZeRO)
- 梯度：N × dtype_size
- 激活值：n_layers × seq_len × batch × d_model × factor
- KV Cache（推理）：2 × n_layers × seq_len × batch × d_model × dtype_size

---

### 3.7 软硬协同优化能力

#### 3.7.1 硬件特性理解

**GPU架构特性**
- Ampere/Hopper/Blackwell架构演进
- Tensor Core特性与限制
- 共享内存Bank Conflict
- Warp Divergence
- 异步内存拷贝（cp.async）
- TMA (Tensor Memory Accelerator)

**NPU/Ascend架构特性**
- 达芬奇架构
- Cube/Vector/AI Core结构
- Ascend C编程模型
- L1/L0/AI Cache层次
- 数据搬运与计算流水

**异构计算特性**
- CPU-GPU/NPU协同
- 数据传输开销
- 异构任务调度
- 统一内存架构

#### 3.7.2 算子级协同优化

**CUDA Kernel优化**
- 共享内存Tiling
- 寄存器优化
- 内存合并访问
- 指令流水
- Warp级原语（__shfl, __ballot）
- Tensor Core编程（wmma API）

**AscendC算子开发**
- AscendC编程模型
- 数据Tiling策略
- Double Buffer
- L1 Cache利用
- Cube/Vector协同

**算子映射到硬件**
- 计算图到硬件指令映射
- 算子选择策略
- 硬件特性感知的算子选择
- 数据布局与硬件效率

#### 3.7.3 系统级协同优化

**数据布局优化**
- NHWC vs NCHW
- 内存对齐
- Bank Conflict避免
- Tensor Core对齐要求

**内存访问模式优化**
- 合并内存访问
- 缓存友好的数据访问
- 数据预取
- 异步数据搬运

**流水线调度**
- 多级流水线：数据搬运 → 计算 → 数据写回
- 双缓冲技术
- 流水线气泡最小化

**资源分配优化**
- 寄存器分配
- 共享内存分配
- 线程块配置
- Occupancy优化

#### 3.7.4 跨平台适配

**多硬件后端适配**
- 抽象层设计
- 硬件特定优化路径
- 运行时后端选择
- 功能降级策略

**性能可移植性**
- 自动调优框架
- 性能模型预测
- 跨平台性能一致性

**硬件特性抽象**
- 统一算子接口
- 硬件能力描述
- 能力查询机制

---

## 4. 学习资料规划

### 4.1 基础前置知识资料

> 以下资料为学习AI训推系统前需要掌握的基础知识。如果你已具备相关背景，可快速浏览或跳过。

#### 4.1.1 深度学习基础

| 资料名称 | 作者/来源 | 类型 | 难度 | 推荐理由 |
|----------|----------|------|------|----------|
| **Deep Learning** | Ian Goodfellow, Yoshua Bengio, Aaron Courville | 书籍 | 中级 | 深度学习圣经，系统覆盖数学基础、经典模型、训练方法 |
| **动手学深度学习 (Dive into Deep Learning, D2L)** | 李沐, Aston Zhang 等 | 书籍+在线课程 | 入门-中级 | 代码驱动，理论实践结合，PyTorch/MXNet双框架，免费在线版 |
| **CS231n: Convolutional Neural Networks for Visual Recognition** | Stanford University (Fei-Fei Li, Andrej Karpathy) | 大学课程 | 中级 | 经典CV课程，深度学习基础概念讲得透彻，课件/视频公开 |
| **CS224n: Natural Language Processing with Deep Learning** | Stanford University (Christopher Manning) | 大学课程 | 中级 | NLP深度学习经典课程，涵盖Transformer、预训练模型等 |
| **Neural Networks and Deep Learning** | Michael Nielsen | 在线书籍 | 入门 | 直觉性讲解神经网络，适合零基础入门 |
| **PyTorch官方教程** | PyTorch Team | 官方文档 | 入门-中级 | PyTorch框架最佳学习资源，从基础到分布式训练全覆盖 |

**在线资源链接：**
- D2L: https://zh.d2l.ai/
- CS231n: https://cs231n.github.io/
- CS224n: https://web.stanford.edu/class/cs224n/
- PyTorch教程: https://pytorch.org/tutorials/

#### 4.1.2 分布式系统基础

| 资料名称 | 作者/来源 | 类型 | 难度 | 推荐理由 |
|----------|----------|------|------|----------|
| **Designing Data-Intensive Applications** | Martin Kleppmann | 书籍 | 中级 | 分布式系统设计圣经，覆盖数据复制、分区、一致性等核心概念 |
| **Distributed Systems** | Maarten van Steen & Andrew S. Tanenbaum | 书籍 | 中级 | 分布式系统经典教材，系统讲解分布式原理 |
| **MIT 6.824: Distributed Systems** | MIT (Robert Morris) | 大学课程 | 高级 | MIT经典分布式系统课程，含Raft共识算法实验，视频公开 |
| **CS149: Parallel Computing** | Stanford University | 大学课程 | 中级-高级 | 并行计算原理与实战，涵盖多核、GPU、集群并行 |

**在线资源链接：**
- MIT 6.824: https://pdos.csail.mit.edu/6.824/
- CS149: https://cs149.stanford.edu/

#### 4.1.3 CUDA/GPU编程基础

| 资料名称 | 作者/来源 | 类型 | 难度 | 推荐理由 |
|----------|----------|------|------|----------|
| **Programming Massively Parallel Processors (4th Ed.)** | David B. Kirk & Wen-mei W. Hwu | 书籍 | 中级 | GPU并行编程权威教材，CUDA编程必读，涵盖架构+编程 |
| **CUDA C++ Programming Guide** | NVIDIA | 官方文档 | 中级-高级 | NVIDIA官方CUDA编程指南，最权威的参考文档 |
| **CUDA C++ Best Practices Guide** | NVIDIA | 官方文档 | 中级-高级 | CUDA性能优化最佳实践，直接面向性能优化 |
| **CS179: GPU Programming** | Caltech | 大学课程 | 中级 | GPU编程入门课程，涵盖CUDA基础、并行算法 |
| **CUDA by Example** | Jason Sanders, Edward Kandrot | 书籍 | 入门-中级 | 通过实例学习CUDA编程，适合初学者上手 |
| **NVIDIA CUDA Crash Course** | CoffeeBeforeArch (YouTube) | 视频博客 | 入门-中级 | 系列视频教程，从CUDA基础到高级优化，实战性强 |
| **GPU Mode YouTube Channel** | GPU Mode Community | 视频博客 | 中级-高级 | 深度GPU编程社区，涵盖CUDA kernel优化、Triton等前沿话题 |

**在线资源链接：**
- CUDA编程指南: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- CUDA最佳实践: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
- CS179: https://sites.google.com/lbl.gov/cs179-fall-2021/
- GPU Mode: https://www.youtube.com/@GPUMODE

#### 4.1.4 Python/C++系统编程基础

| 资料名称 | 作者/来源 | 类型 | 难度 | 推荐理由 |
|----------|----------|------|------|----------|
| **Effective Modern C++** | Scott Meyers | 书籍 | 中级 | C++11/14现代编程最佳实践，训推框架底层用C++ |
| **Python Cookbook (3rd Ed.)** | David Beazley, Brian K. Jones | 书籍 | 中级 | Python高级编程技巧，框架层Python开发必备 |
| **C++ Concurrency in Action (2nd Ed.)** | Anthony Williams | 书籍 | 高级 | C++多线程编程权威指南，推理引擎开发必备 |
| **The C++ Programming Language (4th Ed.)** | Bjarne Stroustrup | 书籍 | 高级 | C++之父编写，C++11全面参考 |

---

### 4.2 大模型训练系统资料

#### 4.2.1 分布式训练策略

| 资料名称 | 作者/来源 | 类型 | 难度 | 推荐理由 |
|----------|----------|------|------|----------|
| **Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism** | Mohammad Shoeybi et al. (NVIDIA) | 论文 | 高级 | Megatron-LM张量并行原始论文，理解张量并行的必读论文 |
| **Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM** | Deepak Narayanan et al. (NVIDIA) | 论文 | 高级 | 3D并行（DP×TP×PP）的原始论文，大规模训练系统设计核心参考 |
| **ZeRO: Memory Optimizations Toward Training Trillion Parameter Models** | Samyam Rajbhandari et al. (Microsoft) | 论文 | 高级 | ZeRO系列原始论文，理解显存优化核心原理的必读论文 |
| **ZeRO-Offload: Democratizing Billion-Scale Model Training** | Jie Ren et al. (Microsoft) | 论文 | 高级 | ZeRO-Offload论文，CPU卸载技术原理 |
| **PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel** | Zhao Yan et al. (Meta) | 论文/博客 | 高级 | FSDP设计与实践经验，PyTorch原生ZeRO-3实现 |
| **DeepSpeed Documentation** | Microsoft DeepSpeed Team | 官方文档 | 中级-高级 | DeepSpeed框架官方文档，ZeRO实现细节与使用指南 |
| **DeepSpeed GitHub Repository** | Microsoft | 开源项目 | 中级-高级 | 深入阅读DeepSpeed源码，理解ZeRO实现机制的最佳途径 |
| **Megatron-LM GitHub Repository** | NVIDIA | 开源项目 | 高级 | Megatron-LM源码，理解3D并行编排的实现 |
| **Efficient Training on Large Clusters** | HuggingFace Documentation | 博客/文档 | 中级 | HuggingFace关于大规模训练的实践指南，涵盖FSDP、DeepSpeed等 |
| **Large-Scale AI Models Training: A Survey** | 多位作者 | 综述论文 | 中级-高级 | 大规模训练系统综述，系统梳理训练优化技术 |

**在线资源链接：**
- DeepSpeed: https://github.com/microsoft/DeepSpeed
- Megatron-LM: https://github.com/NVIDIA/Megatron-LM
- HuggingFace Training: https://huggingface.co/docs/transformers/perf_train_gpu_many
- FSDP Blog: https://engineering.fb.com/2021/07/15/open-source/fsdp-the-first-experience-in-training-a-trillion-scale-model/

#### 4.2.2 训练全流程

| 资料名称 | 作者/来源 | 类型 | 难度 | 推荐理由 |
|----------|----------|------|------|----------|
| **CS324: Large Language Models** | Stanford (Tatsu Hashimoto, Percy Liang) | 大学课程 | 中级-高级 | Stanford LLM课程，涵盖预训练、微调、对齐全流程 |
| **Training Large Language Models (LLMs) — A Practical Guide** | Sebastian Raschka | 博客 | 中级 | 系统梳理LLM训练全流程的实践指南，从预训练到RLHF |
| **LLM Training Guide** | HuggingFace | 官方文档 | 中级 | HuggingFace训练指南，SFT、PEFT、RLHF实践 |
| **RLHF / RLAIF / DPO — Everything You Need to Know** | HuggingFace Blog | 博客 | 中级 | RLHF和DPO的全面介绍，含代码示例 |
| **Direct Preference Optimization: Your Language Model is Secretly a Reward Model** | Rafailov et al. | 论文 | 高级 | DPO原始论文，理解对齐训练新范式 |
| **Fine-tuning Large Language Models** | DeepLearning.AI (Andrew Ng) | 在线课程 | 入门-中级 | Andrew Ng的微调课程，实操性强 |
| **LLaMA-Factory GitHub** | hiyouga | 开源项目 | 中级 | 一站式LLM微调框架，支持SFT/RLHF/DPO，代码可读性好 |
| **ColossalAI GitHub** | HPC-AI Tech | 开源项目 | 中级-高级 | 大模型训练系统，涵盖多种并行策略实现 |

**在线资源链接：**
- CS324: https://stanford-cs324.github.io/llm/
- HuggingFace TRL: https://github.com/huggingface/trl
- LLaMA-Factory: https://github.com/hiyouga/LLaMA-Factory
- ColossalAI: https://github.com/hpcaitech/ColossalAI

#### 4.2.3 训练机制与稳定性

| 资料名称 | 作者/来源 | 类型 | 难度 | 推荐理由 |
|----------|----------|------|------|----------|
| **Mixed Precision Training** | Paul Micikevicius et al. (NVIDIA) | 论文 | 中级-高级 | 混合精度训练原始论文，理解FP16训练原理 |
| **FP8 Formats for Deep Learning** | NVIDIA | 技术白皮书 | 高级 | FP8格式技术白皮书，理解最新精度格式 |
| **On the Stability of Large Language Model Training** | Xinyuan Li et al. | 论文 | 高级 | LLM训练稳定性分析，loss spike原因与对策 |
| **Scaling Laws for Neural Language Models** | Jared Kaplan et al. (OpenAI) | 论文 | 高级 | Scaling Laws论文，理解模型规模与训练参数关系 |
| **Training Compute-Optimal Large Language Models (Chinchilla)** | Jordan Hoffmann et al. (DeepMind) | 论文 | 高级 | Chinchilla论文，计算最优训练策略 |
| **PyTorch Distributed Data Parallel (DDP) Tutorial** | PyTorch | 官方教程 | 中级 | DDP使用教程，理解梯度同步机制 |
| **PyTorch FSDP Tutorial** | PyTorch | 官方教程 | 中级-高级 | FSDP使用教程，理解参数分片机制 |
| **NVIDIA NeMo GitHub** | NVIDIA | 开源项目 | 中级-高级 | NVIDIA的大模型训练框架，含训练稳定性工具 |

**在线资源链接：**
- NeMo: https://github.com/NVIDIA/NeMo
- PyTorch DDP: https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
- PyTorch FSDP: https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html

#### 4.2.4 大规模训练运维

| 资料名称 | 作者/来源 | 类型 | 难度 | 推荐理由 |
|----------|----------|------|------|----------|
| **Bamboo: Pretraining Large Language Models on Failure-prone Clusters** | Jacob R. et al. | 论文 | 高级 | 大规模训练容错方案，理解故障恢复机制 |
| **Environment Selective Recomputation** | Microsoft DeepSpeed Blog | 博客 | 高级 | DeepSpeed的容错与弹性训练实践 |
| **Kubernetes for AI/ML Workloads** | CNCF | 博客/文档 | 中级 | K8s调度AI训练任务的实践 |
| **Ray: A Distributed Framework for Scaling AI Applications** | UC Berkeley RISELab | 开源项目 | 中级-高级 | 分布式计算框架，支持弹性训练和推理 |
| **Torchrun / PyTorch Elastic** | PyTorch | 官方文档 | 中级 | PyTorch弹性训练工具，支持容错和动态伸缩 |
| **Microsoft DeepSpeed Blog** | Microsoft | 博客 | 中级-高级 | DeepSpeed团队博客，大量训练系统优化实践文章 |

**在线资源链接：**
- Ray: https://github.com/ray-project/ray
- PyTorch Elastic: https://pytorch.org/docs/stable/elastic.html
- DeepSpeed Blog: https://www.deepspeed.ai/blog/

#### 4.2.5 训练框架源码阅读

| 资料名称 | 作者/来源 | 类型 | 难度 | 推荐理由 |
|----------|----------|------|------|----------|
| **PyTorch Distributed源码** | PyTorch | 开源项目 | 高级 | 阅读torch.distributed源码，理解DDP/FSDP/RPC底层实现 |
| **DeepSpeed源码** | Microsoft | 开源项目 | 高级 | 阅读DeepSpeed源码，理解ZeRO/3D并行实现 |
| **Megatron-LM源码** | NVIDIA | 开源项目 | 高级 | 阅读Megatron-LM源码，理解张量并行/流水线并行实现 |
| **MindSpore分布式训练文档** | 官方 | 官方文档 | 中级-高级 | MindSpore自动并行/半自动并行机制文档 |
| **PyTorch DTensor Documentation** | PyTorch | 官方文档 | 高级 | PyTorch原生分布式张量抽象，理解未来分布式训练方向 |

**在线资源链接：**
- PyTorch Distributed: https://github.com/pytorch/pytorch/tree/main/torch/distributed
- MindSpore: https://www.mindspore.cn/docs/
- DTensor: https://pytorch.org/docs/stable/distributed.tensor.html

---

### 4.3 大模型推理系统资料

#### 4.3.1 推理优化技术

| 资料名称 | 作者/来源 | 类型 | 难度 | 推荐理由 |
|----------|----------|------|------|----------|
| **Efficient Memory Management for Large Language Model Serving with PagedAttention** | Woosuk Kwon et al. (UC Berkeley) | 论文 | 高级 | PagedAttention/vLLM原始论文，推理系统优化的里程碑工作 |
| **vLLM GitHub Repository** | UC Berkeley | 开源项目 | 中级-高级 | vLLM源码，PagedAttention和Continuous Batching的实现 |
| **Orca: A Distributed Serving System for Transformer-Based Generative Models** | Yu Gao et al. (OSDI 2022) | 论文 | 高级 | Continuous Batching（Iteration-level scheduling）原始论文 |
| **Fast Inference from Transformers via Speculative Decoding** | Yaniv Leviathan et al. (Google) | 论文 | 高级 | Speculative Decoding原始论文，理解投机解码原理 |
| **Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads** | Tianle Cai et al. | 论文 | 高级 | Medusa投机解码方法，并行生成多个候选Token |
| **EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty** | Yuhui Li et al. | 论文 | 高级 | EAGLE投机解码方法，当前SOTA投机解码方案 |
| **AttentionStore: Cost-effective Attention Reuse across Multi-turn Conversations** | 论文 | 论文 | 高级 | Prefix Caching技术原理 |
| **LLM Inference series** | Lei Wang (知乎博客) | 博客系列 | 中级-高级 | 系统讲解LLM推理优化技术，从KV Cache到PagedAttention |
| **Serving LLMs: From vLLM to SGLang** | LMSYS Blog | 博客 | 中级-高级 | LMSYS团队推理系统实践，涵盖vLLM/SGLang技术对比 |

**在线资源链接：**
- vLLM: https://github.com/vllm-project/vllm
- vLLM文档: https://docs.vllm.ai/
- SGLang: https://github.com/sgl-project/sglang
- LLM推理系列博客: https://zhuanlan.zhihu.com/p/655325872

#### 4.3.2 推理引擎

| 资料名称 | 作者/来源 | 类型 | 难度 | 推荐理由 |
|----------|----------|------|------|----------|
| **vLLM Documentation & Source** | vLLM Project | 开源项目+文档 | 中级-高级 | vLLM完整源码与文档，理解PagedAttention/Continuous Batching实现 |
| **TensorRT-LLM Documentation** | NVIDIA | 官方文档 | 高级 | NVIDIA官方推理引擎文档，涵盖算子融合、量化推理、In-flight Batching |
| **TensorRT-LLM GitHub** | NVIDIA | 开源项目 | 高级 | TensorRT-LLM源码，理解NVIDIA推理优化实现 |
| **TGI (Text Generation Inference) GitHub** | HuggingFace | 开源项目 | 中级 | HuggingFace推理引擎源码，理解Rust+Python推理服务架构 |
| **llama.cpp GitHub** | Georgi Gerganov | 开源项目 | 中级 | 极致优化的CPU/GPU推理引擎，理解GGML量化推理 |
| **MindSpore Lite Documentation** | 官方 | 官方文档 | 中级 | Ascend推理引擎文档，理解NPU推理优化 |
| **DeepSpeed-FastGen** | Microsoft | 开源项目 | 中级-高级 | DeepSpeed推理引擎，结合DeepSpeed推理优化与vLLM技术 |
| **FlexFlow Serve** | Stanford/FlexFlow | 开源项目 | 高级 | 分布式推理引擎，理解多模型推理调度 |

**在线资源链接：**
- TensorRT-LLM: https://github.com/NVIDIA/TensorRT-LLM
- TGI: https://github.com/huggingface/text-generation-inference
- llama.cpp: https://github.com/ggerganov/llama.cpp
- MindSpore Lite: https://www.mindspore.cn/lite
- DeepSpeed-FastGen: https://github.com/microsoft/DeepSpeed/tree/master/blogs/deepspeed-fastgen

#### 4.3.3 推理服务化

| 资料名称 | 作者/来源 | 类型 | 难度 | 推荐理由 |
|----------|----------|------|------|----------|
| **MLPerf Inference Benchmark** | MLCommons | 基准测试 | 中级 | 推理性能标准基准，理解推理性能评估标准 |
| **Ray Serve Documentation** | Ray Project | 官方文档 | 中级 | Ray的模型服务框架，理解弹性推理服务 |
| **Triton Inference Server** | NVIDIA | 开源项目 | 中级-高级 | NVIDIA推理服务器，多模型管理、动态batching |
| **KServe (formerly KFServing)** | CNCF | 开源项目 | 中级 | Kubernetes上的Serverless推理框架，理解云原生推理服务 |
| **Building ML Powered Applications** | Emmanuel Ameisen | 书籍 | 中级 | ML应用工程化实践，含模型服务化部署 |

**在线资源链接：**
- MLPerf: https://mlcommons.org/benchmarks/inference-datacenter/
- Ray Serve: https://docs.ray.io/en/latest/serve/index.html
- Triton: https://github.com/triton-inference-server/server
- KServe: https://github.com/kserve/kserve

#### 4.3.4 推理性能优化实践

| 资料名称 | 作者/来源 | 类型 | 难度 | 推荐理由 |
|----------|----------|------|------|----------|
| **How to Scale LLM Inference** | LMSYS Blog | 博客 | 高级 | LLM推理扩展性分析，涵盖吞吐/延迟优化 |
| **LLM Inference Performance Analysis** | HuggingFace Blog | 博客 | 中级-高级 | HuggingFace推理性能分析，含profiling实践 |
| **vLLM Performance Tuning Guide** | vLLM Project | 文档 | 中级 | vLLM性能调优指南 |
| **Continuous Batching vs Static Batching** | AnyScale Blog | 博客 | 中级 | Continuous Batching性能分析对比 |
| **Making LLM Inference 3x Faster** | DeepSpeed Blog | 博客 | 中级-高级 | DeepSpeed推理优化实践 |
| **NVIDIA TensorRT-LLM Performance Guide** | NVIDIA | 技术文档 | 高级 | TensorRT-LLM性能调优指南 |

**在线资源链接：**
- HuggingFace推理优化: https://huggingface.co/blog/optimum-tpu
- AnyScale Blog: https://www.anyscale.com/blog
- DeepSpeed推理: https://www.deepspeed.ai/inference/

---

### 4.4 高性能计算与通信资料

#### 4.4.1 GPU/NPU架构

| 资料名称 | 作者/来源 | 类型 | 难度 | 推荐理由 |
|----------|----------|------|------|----------|
| **NVIDIA Hopper Architecture In-Depth Whitepaper** | NVIDIA | 技术白皮书 | 高级 | Hopper架构白皮书，理解Tensor Core、TMA等新特性 |
| **NVIDIA Ampere Architecture Whitepaper** | NVIDIA | 技术白皮书 | 高级 | Ampere架构白皮书，理解SM结构、稀疏Tensor Core |
| **Programming Massively Parallel Processors (4th Ed.)** | Kirk & Hwu | 书籍 | 中级 | GPU架构与编程权威教材，覆盖从CUDA基础到高级优化 |
| **Computer Organization and Design RISC-V Edition** | David Patterson & John Hennessy | 书籍 | 中级 | 计算机体系结构经典教材，理解存储层次、并行计算基础 |
| **Ascend AI Processor Architecture and Programming** | 官方 | 技术文档 | 高级 | Ascend NPU架构文档，理解达芬奇架构 |
| **CUDA C++ Programming Guide — Hardware Implementation** | NVIDIA | 官方文档 | 中级-高级 | CUDA编程指南中的硬件实现章节，理解SM/Warp/SIMT |
| **NVIDIA GPU Architecture Blog Series** | Chipressa / Lei Mao | 博客系列 | 高级 | GPU架构深度解析博客系列 |
| **CMU 15-418: Parallel Computer Architecture and Programming** | CMU | 大学课程 | 中级-高级 | CMU并行计算课程，涵盖GPU架构与并行编程 |

**在线资源链接：**
- Hopper白皮书: https://resources.nvidia.com/en-us-hopper-architecture
- Ampere白皮书: https://www.nvidia.com/en-us/data-center/ampere-architecture/
- CUDA硬件实现: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#hardware-implementation
- CMU 15-418: https://www.cs.cmu.edu/418/
- Lei Mao Blog: https://leimao.github.io/

#### 4.4.2 集合通信

| 资料名称 | 作者/来源 | 类型 | 难度 | 推荐理由 |
|----------|----------|------|------|----------|
| **NCCL Documentation** | NVIDIA | 官方文档 | 中级-高级 | NCCL官方文档，理解集合通信库接口与实现 |
| **NCCL GitHub (with internal details)** | NVIDIA | 开源项目 | 高级 | NCCL源码（部分开源），理解Ring/Tree算法实现 |
| **HCCL Documentation** | 官方 | 官方文档 | 中级-高级 | HCCL文档，理解NPU集合通信 |
| **MPI Forum Documentation** | MPI Forum | 标准文档 | 中级 | MPI标准文档，理解消息传递接口 |
| **Bringing HPC Techniques to Deep Learning** | Andrew Gibiansky | 博客 | 中级 | 经典博客，手把手推导Ring AllReduce算法 |
| **Collective Communication in Distributed Deep Learning** | Amar Phanishayee | 博客 | 中级 | 集合通信原理与性能分析 |
| **Performance Analysis of Collective Communication on GPU Clusters** | 多位作者 | 论文 | 高级 | GPU集群集合通信性能分析论文 |

**在线资源链接：**
- NCCL: https://docs.nvidia.com/deeplearning/nccl/
- NCCL GitHub: https://github.com/NVIDIA/nccl
- Ring AllReduce博客: https://andrew.gibiansky.com/blog/machine-learning/baidu-allreduce/
- HCCL: https://www.hiascend.com/document/detail/zh/CANNCommunityEdition

#### 4.4.3 通信拓扑与高速互联

| 资料名称 | 作者/来源 | 类型 | 难度 | 推荐理由 |
|----------|----------|------|------|----------|
| **NVIDIA NVLink and NVSwitch Whitepaper** | NVIDIA | 技术白皮书 | 高级 | NVLink/NVSwitch技术白皮书，理解GPU高速互联 |
| **InfiniBand Architecture Specification** | IBTA | 标准文档 | 高级 | InfiniBand架构规范，理解RDMA技术 |
| **RDMA over Converged Ethernet (RoCE) Specification** | IBTA | 标准文档 | 高级 | RoCE规范，理解以太网上的RDMA |
| **GPUDirect RDMA Documentation** | NVIDIA | 官方文档 | 高级 | GPUDirect RDMA文档，理解GPU显存直接RDMA |
| **Topology-Aware Scheduling for Distributed Training** | 多位作者 | 论文 | 高级 | 拓扑感知调度论文，理解通信优化策略 |
| **Hierarchical AllReduce for Multi-GPU Training** | 多位作者 | 论文 | 高级 | 层次化AllReduce算法论文 |
| **HPC Networking: InfiniBand vs RoCE vs Ethernet** | NVIDIA/Mellanox | 技术博客 | 中级 | 高速互联技术对比分析 |

**在线资源链接：**
- NVLink: https://www.nvidia.com/en-us/data-center/nvlink/
- GPUDirect: https://docs.nvidia.com/cuda/gpudirect-rdma/
- InfiniBand: https://www.infinibandta.org/

#### 4.4.4 计算通信重叠

| 资料名称 | 作者/来源 | 类型 | 难度 | 推荐理由 |
|----------|----------|------|------|----------|
| **Overlap Communication and Computation in DDP** | PyTorch Blog | 博客 | 中级-高级 | PyTorch DDP通信-计算重叠机制详解 |
| **Gradient Bucketing in PyTorch DDP** | PyTorch Blog | 博客 | 中级 | DDP梯度分桶同步机制 |
| **Communication-Efficient SGD via Gradient Quantization** | Frank Seide et al. | 论文 | 高级 | 梯度量化压缩通信论文 |
| **PowerSGD: Cross-Replica Gradient Compression for Distributed Training** | Thakur et al. | 论文 | 高级 | 梯度低秩压缩通信论文 |
| **Deep Gradient Compression** | Yujun Lin et al. | 论文 | 高级 | 深度梯度压缩，大幅减少通信量 |
| **PyTorch Distributed Overview** | PyTorch | 官方文档 | 中级 | PyTorch分布式训练全景文档 |

**在线资源链接：**
- PyTorch DDP Blog: https://pytorch.org/blog/overview-of-utilities-for-distributed-training/
- PyTorch Distributed: https://pytorch.org/tutorials/beginner/dist_overview.html

#### 4.4.5 Roofline模型与性能分析

| 资料名称 | 作者/来源 | 类型 | 难度 | 推荐理由 |
|----------|----------|------|------|----------|
| **Roofline: An Insightful Visual Performance Model** | Samuel Williams et al. (LBNL) | 论文 | 中级-高级 | Roofline模型原始论文，理解性能分析方法论 |
| **NVIDIA Nsight Compute Roofline Analysis** | NVIDIA | 官方文档 | 中级-高级 | Nsight Compute中的Roofline分析功能 |
| **CS217: Parallel Computer Architecture** | Stanford | 大学课程 | 高级 | Stanford并行计算机架构课程，深入性能分析 |
| **MIT 6.172: Performance Engineering of Software Systems** | MIT | 大学课程 | 中级-高级 | MIT性能工程课程，系统讲解性能优化方法论 |

**在线资源链接：**
- Roofline论文: https://crd.lbl.gov/departments/computer-science/PAR/research/roofline/
- Nsight Compute: https://docs.nvidia.com/nsight-compute/
- CS217: https://cs217.stanford.edu/
- MIT 6.172: https://ocw.mit.edu/courses/6-172-performance-engineering-of-software-systems-fall-2018/

---

### 4.5 训推加速技术资料

#### 4.5.1 Flash Attention

| 资料名称 | 作者/来源 | 类型 | 难度 | 推荐理由 |
|----------|----------|------|------|----------|
| **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness** | Tri Dao et al. (Stanford) | 论文 | 高级 | Flash Attention v1原始论文，IO感知注意力机制 |
| **FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning** | Tri Dao | 论文 | 高级 | Flash Attention v2论文，改进并行度和工作分配 |
| **FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-Precision** | Tri Dao et al. | 论文 | 高级 | Flash Attention v3论文，利用Hopper异步和FP8 |
| **FlashAttention GitHub** | Tri Dao | 开源项目 | 高级 | Flash Attention源码，理解分块计算和在线Softmax实现 |
| **FlashAttention: Understanding the Algorithm** | Alex Yang (博客) | 博客 | 中级 | Flash Attention算法详细图解，帮助理解核心思想 |
| **Making FlashAttention Fast** | Tri Dao (Blog) | 博客 | 高级 | Flash Attention优化细节，作者亲自解读 |

**在线资源链接：**
- FlashAttention: https://github.com/Dao-AILab/flash-attention
- Flash Attention论文: https://arxiv.org/abs/2205.14135
- Flash Attention图解: https://alexatpantheon.medium.com/

#### 4.5.2 量化技术

| 资料名称 | 作者/来源 | 类型 | 难度 | 推荐理由 |
|----------|----------|------|------|----------|
| **Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference** | Jacob et al. (Google) | 论文 | 高级 | 经典INT8量化论文，理解量化原理 |
| **GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers** | Frantar et al. | 论文 | 高级 | GPTQ量化方法论文，LLM INT4量化的里程碑 |
| **AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration** | Lin et al. (MIT) | 论文 | 高级 | AWQ量化方法论文，基于激活感知的权重量化 |
| **SmoothQuant: Accurate and Efficient Post-Training Quantization for LLMs** | Xiao et al. (MIT) | 论文 | 高级 | SmoothQuant INT8量化方法 |
| **FP8 Quantization for Deep Learning** | NVIDIA | 技术白皮书 | 高级 | FP8量化技术白皮书 |
| **A Survey of Quantization Methods for Efficient Neural Network Inference** | Amir Gholami et al. | 综述论文 | 中级-高级 | 量化技术综述，系统梳理量化方法 |
| **LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale** | Tim Dettmers et al. | 论文 | 高级 | LLM INT8量化方法，混合精度分解 |
| **AutoGPTQ GitHub** | AutoGPTQ Team | 开源项目 | 中级 | GPTQ量化实现，支持多种LLM |
| **AutoAWQ GitHub** | MIT HAN Lab | 开源项目 | 中级 | AWQ量化实现 |
| **BitsAndBytes GitHub** | Tim Dettmers | 开源项目 | 中级 | INT8/INT4量化库，HuggingFace集成 |

**在线资源链接：**
- AutoGPTQ: https://github.com/AutoGPTQ/AutoGPTQ
- AutoAWQ: https://github.com/casper-hansen/AutoAWQ
- BitsAndBytes: https://github.com/bitsandbytes-foundation/bitsandbytes
- HAN Lab量化: https://hanlab.mit.edu/projects/quantization

#### 4.5.3 知识蒸馏

| 资料名称 | 作者/来源 | 类型 | 难度 | 推荐理由 |
|----------|----------|------|------|----------|
| **Distilling the Knowledge in a Neural Network** | Geoffrey Hinton et al. | 论文 | 中级 | 知识蒸馏开山论文，理解软标签蒸馏原理 |
| **FitNets: Hints for Thin Deep Nets** | Romero et al. | 论文 | 中级 | 特征蒸馏方法，中间层知识传递 |
| **A Survey on Knowledge Distillation** | 多位作者 | 综述论文 | 中级 | 知识蒸馏综述，系统梳理蒸馏方法 |
| **MiniLLM: Knowledge Distillation of Large Language Models** | Gu et al. | 论文 | 高级 | LLM知识蒸馏方法，大模型蒸馏到小模型 |
| **DistilBERT: A distilled version of BERT** | Victor Sanh et al. (HuggingFace) | 论文 | 中级 | DistilBERT蒸馏实践，经典案例 |

#### 4.5.4 算子优化与开发

| 资料名称 | 作者/来源 | 类型 | 难度 | 推荐理由 |
|----------|----------|------|------|----------|
| **CUDA C++ Programming Guide — Optimization** | NVIDIA | 官方文档 | 高级 | CUDA优化指南，涵盖共享内存、寄存器、向量化等 |
| **CUDA C++ Best Practices Guide** | NVIDIA | 官方文档 | 高级 | CUDA最佳实践指南，直接面向性能优化 |
| **Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations** | Philippe Tillet et al. | 论文 | 高级 | Triton语言论文，理解GPU编程新范式 |
| **Triton Language Tutorial** | OpenAI | 官方教程 | 中级-高级 | Triton语言教程，编写GPU Kernel的新方法 |
| **Triton GitHub** | OpenAI | 开源项目 | 中级-高级 | Triton源码和示例，Flash Attention的Triton实现 |
| **FasterTransformer GitHub** | NVIDIA | 开源项目 | 高级 | NVIDIA推理加速库，大量算子融合实现 |
| **Ascend C Programming Guide** | 官方 | 官方文档 | 高级 | Ascend C算子开发指南，NPU算子编程 |
| **CUTLASS: CUDA Templates for Linear Algebra** | NVIDIA | 开源项目 | 高级 | NVIDIA矩阵运算模板库，高效GEMM实现参考 |
| **CUDA Mode Discord/YouTube** | Community | 社区/视频 | 中级-高级 | 活跃的CUDA编程社区，大量Kernel优化案例 |
| **GPU Optimization for Machine Learning** | Stanford CS149 | 课程材料 | 高级 | Stanford课程中的GPU优化章节 |

**在线资源链接：**
- Triton: https://github.com/openai/triton
- Triton教程: https://triton-lang.org/
- FasterTransformer: https://github.com/NVIDIA/FasterTransformer
- CUTLASS: https://github.com/NVIDIA/cutlass
- Ascend C: https://www.hiascend.com/document/detail/zh/AscendC

#### 4.5.5 计算图优化

| 资料名称 | 作者/来源 | 类型 | 难度 | 推荐理由 |
|----------|----------|------|------|----------|
| **Compilers: Principles, Techniques, and Tools (Dragon Book)** | Aho, Lam, Sethi, Ullman | 书籍 | 高级 | 编译器原理经典教材，理解图优化的理论基础 |
| **TVM: An Automated End-to-End Optimizing Compiler for DL** | Tianqi Chen et al. | 论文 | 高级 | TVM编译器论文，理解深度学习编译器 |
| **XLA: Optimizing Compiler for Machine Learning** | Google | 官方文档 | 高级 | XLA编译器文档，理解计算图优化 |
| **TVM Documentation** | Apache TVM | 官方文档 | 高级 | TVM文档，理解算子调优和图优化 |
| **TorchDynamo + TorchInductor** | PyTorch | 官方文档 | 高级 | PyTorch 2.0编译技术，理解图捕获和代码生成 |
| **MLIR: Multi-Level Intermediate Representation** | LLVM/Google | 开源项目 | 高级 | 多级中间表示，理解编译器IR设计 |
| **TensorFlow XLA Architecture** | Google | 技术文档 | 高级 | XLA架构详解，理解图级优化Pass |

**在线资源链接：**
- TVM: https://tvm.apache.org/
- XLA: https://www.tensorflow.org/xla
- TorchDynamo: https://pytorch.org/docs/stable/dynamo.html
- MLIR: https://mlir.llvm.org/

#### 4.5.6 显存优化

| 资料名称 | 作者/来源 | 类型 | 难度 | 推荐理由 |
|----------|----------|------|------|----------|
| **Training Deep Nets with Sublinear Memory Cost** | Tianqi Chen et al. | 论文 | 高级 | 梯度检查点原始论文，理解显存-计算权衡 |
| **Checkmate: Deep Learning Training via Optimal Rematerialization** | Jain et al. | 论文 | 高级 | 最优重计算策略，显存优化理论 |
| **Selective Recomputation** | DeepSpeed Blog | 博客 | 高级 | 选择性重计算策略，Attention层重计算 |
| **PyTorch Memory Format Guide** | PyTorch | 官方文档 | 中级 | PyTorch内存格式与分配器文档 |
| **Memory Management in PyTorch** | PyTorch Blog | 博客 | 中级 | PyTorch内存管理机制详解 |

**在线资源链接：**
- PyTorch Memory: https://pytorch.org/blog/understanding-gpu-memory-1/
- DeepSpeed Selective Recomputation: https://www.deepspeed.ai/tutorials/

---

### 4.6 系统架构与工程实践资料

#### 4.6.1 系统架构设计

| 资料名称 | 作者/来源 | 类型 | 难度 | 推荐理由 |
|----------|----------|------|------|----------|
| **Designing Data-Intensive Applications** | Martin Kleppmann | 书籍 | 中级 | 分布式系统设计圣经，涵盖数据复制、分区、一致性 |
| **System Design Interview (Vol 1 & 2)** | Alex Xu | 书籍 | 中级 | 系统设计面试经典，含ML系统设计案例 |
| **Machine Learning System Design Interview** | Ali Aminian | 书籍 | 中级 | ML系统设计专门书籍，含推荐系统、搜索系统等 |
| **ML System Design Course** | Stanford CS329S | 大学课程 | 中级-高级 | Stanford ML系统设计课程，涵盖ML全生命周期 |
| **Hidden Technical Debt in Machine Learning Systems** | D. Sculley et al. (Google) | 论文 | 中级 | ML系统技术债经典论文，理解ML系统工程挑战 |
| **Continuous Delivery for Machine Learning** | Sato et al. | 博客/论文 | 中级 | ML CI/CD实践 |
| **The ML Test Score: A Rubric for ML Production Readiness** | Google | 论文 | 中级 | ML生产就绪度评估框架 |

**在线资源链接：**
- CS329S: https://stanford-cs329s.github.io/
- ML技术债论文: https://proceedings.neurips.cc/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html

#### 4.6.2 开源项目源码阅读

| 资料名称 | 作者/来源 | 类型 | 难度 | 推荐理由 |
|----------|----------|------|------|----------|
| **vLLM Source Code** | vLLM Project | 开源项目 | 高级 | 推理引擎架构设计最佳参考，PagedAttention实现 |
| **DeepSpeed Source Code** | Microsoft | 开源项目 | 高级 | 训练系统架构设计参考，ZeRO/3D并行实现 |
| **Megatron-LM Source Code** | NVIDIA | 开源项目 | 高级 | 3D并行编排实现，张量/流水线并行工程实现 |
| **PyTorch Distributed Source** | PyTorch | 开源项目 | 高级 | DDP/FSDP/RPC实现，分布式训练底层机制 |
| **Ray Source Code** | Ray Project | 开源项目 | 中级-高级 | 分布式计算框架，弹性调度架构 |
| **NVIDIA Triton Source** | NVIDIA | 开源项目 | 中级-高级 | 推理服务器架构，多模型管理 |
| **SGLang Source** | LMSYS | 开源项目 | 高级 | 新一代推理引擎，RadixAttention等创新 |
| **TensorRT-LLM Source** | NVIDIA | 开源项目 | 高级 | NVIDIA推理引擎，算子融合/量化推理实现 |

**在线资源链接：**
- vLLM: https://github.com/vllm-project/vllm
- SGLang: https://github.com/sgl-project/sglang
- Ray: https://github.com/ray-project/ray
- Triton Server: https://github.com/triton-inference-server/server

---

### 4.7 性能分析与优化资料

#### 4.7.1 性能分析工具

| 资料名称 | 作者/来源 | 类型 | 难度 | 推荐理由 |
|----------|----------|------|------|----------|
| **NVIDIA Nsight Systems Documentation** | NVIDIA | 官方文档 | 中级-高级 | 系统级性能分析工具文档，Timeline分析 |
| **NVIDIA Nsight Compute Documentation** | NVIDIA | 官方文档 | 高级 | Kernel级性能分析工具文档，Roofline分析 |
| **PyTorch Profiler Documentation** | PyTorch | 官方文档 | 中级 | PyTorch内置性能分析器，CPU/GPU/通信分析 |
| **MindSpore Profiler Documentation** | 官方 | 官方文档 | 中级-高级 | MindSpore性能分析器，Ascend平台 |
| **Chrome Trace Event Format** | Google | 技术文档 | 中级 | 性能Trace格式标准，理解性能数据表示 |
| **torch.profiler Tutorial** | PyTorch | 官方教程 | 中级 | PyTorch Profiler使用教程，含实战案例 |

**在线资源链接：**
- Nsight Systems: https://docs.nvidia.com/nsight-systems/
- Nsight Compute: https://docs.nvidia.com/nsight-compute/
- PyTorch Profiler: https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html

#### 4.7.2 性能优化方法论

| 资料名称 | 作者/来源 | 类型 | 难度 | 推荐理由 |
|----------|----------|------|------|----------|
| **The Art of Performance Engineering** | MIT 6.172 | 大学课程 | 中级-高级 | MIT性能工程课程，系统讲解性能优化方法论 |
| **Performance Analysis and Tuning on Modern CPUs** | Jim Cownie | 书籍 | 高级 | 性能分析与调优实践 |
| **NVIDIA GPU Performance Background** | NVIDIA | 技术文档 | 中级-高级 | GPU性能背景知识，理解性能瓶颈 |
| **Optimizing Distributed Training** | DeepSpeed Blog | 博客 | 高级 | DeepSpeed团队分布式训练优化实践 |
| **Scaling PyTorch on HPC Clusters** | PyTorch | 官方文档 | 中级-高级 | HPC集群上PyTorch优化指南 |
| **MLPerf Training Analysis Reports** | MLCommons | 基准报告 | 高级 | MLPerf训练基准分析，理解SOTA训练优化 |

**在线资源链接：**
- MLPerf Training: https://mlcommons.org/benchmarks/training/
- GPU性能背景: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html
- DeepSpeed优化: https://www.deepspeed.ai/training/

#### 4.7.3 性能建模

| 资料名称 | 作者/来源 | 类型 | 难度 | 推荐理由 |
|----------|----------|------|------|----------|
| **Computing and Communication in Distributed Deep Learning** | 多位作者 | 论文 | 高级 | 分布式训练计算通信建模 |
| **Efficient Scaling of LLM Training** | NVIDIA | 技术博客 | 高级 | LLM训练扩展性分析与建模 |
| **Megatron-LM Compute Analysis** | NVIDIA Blog | 博客 | 高级 | Megatron-LM计算量与通信量分析 |
| **Transformer Math 201** | EleutherAI | 博客 | 中级 | Transformer计算量/显存/通信量详细推导 |
| **LLM Training Cost Calculator** | 多种工具 | 工具 | 中级 | LLM训练成本计算器，理解训练资源需求 |

**在线资源链接：**
- EleutherAI Transformer Math: https://blog.eleuther.ai/transformer-math/
- LLM计算器: https://github.com/kingbri1/llm-calculator

---

### 4.8 软硬协同优化资料

#### 4.8.1 硬件架构深度

| 资料名称 | 作者/来源 | 类型 | 难度 | 推荐理由 |
|----------|----------|------|------|----------|
| **Computer Architecture: A Quantitative Approach (6th Ed.)** | Hennessy & Patterson | 书籍 | 高级 | 计算机体系结构圣经，理解存储层次、并行计算 |
| **NVIDIA H100 Architecture Deep Dive** | NVIDIA | 技术文档 | 高级 | H100架构深度解析，TMA/Transformer Engine等新特性 |
| **NVIDIA Blackwell Architecture** | NVIDIA | 技术白皮书 | 高级 | 最新Blackwell架构，理解未来GPU演进方向 |
| **Ascend 910 Architecture Documentation** | 官方 | 技术文档 | 高级 | Ascend 910架构文档，理解NPU架构 |
| **DaVinci Architecture Whitepaper** | 官方 | 技术白皮书 | 高级 | 达芬奇架构白皮书，Cube/Vector单元详解 |
| **GPU Microarchitecture** | Chips and Cheese | 博客系列 | 高级 | GPU微架构深度分析博客 |

**在线资源链接：**
- H100: https://resources.nvidia.com/en-us-hopper-architecture
- Blackwell: https://www.nvidia.com/en-us/technologies/blackwell-architecture/
- Chips and Cheese: https://chipsandcheese.com/

#### 4.8.2 算子级优化实践

| 资料名称 | 作者/来源 | 类型 | 难度 | 推荐理由 |
|----------|----------|------|------|----------|
| **CUTLASS GEMM Examples** | NVIDIA | 开源项目 | 高级 | NVIDIA GEMM模板库，高效矩阵乘法实现 |
| **CUDA Flash Attention Implementation** | Tri Dao | 开源项目 | 高级 | Flash Attention的CUDA实现，学习算子优化最佳实践 |
| **Triton Flash Attention Tutorial** | OpenAI | 教程 | 高级 | 用Triton实现Flash Attention的教程 |
| **CUDA Memory Coalescing Tutorial** | NVIDIA | 官方教程 | 中级-高级 | 内存合并访问教程，理解内存访问优化 |
| **Warp-Level Primitives Guide** | NVIDIA | 官方文档 | 高级 | Warp级原语使用指南，__shfl/__ballot等 |
| **Tensor Core Programming Guide** | NVIDIA | 官方文档 | 高级 | Tensor Core编程指南，wmma API使用 |
| **AscendC Operator Development Guide** | 官方 | 官方文档 | 高级 | Ascend C算子开发完整指南 |
| **GPU Optimization Blog Series** | Lei Mao | 博客系列 | 高级 | GPU优化深度博客，涵盖各种算子优化案例 |
| **Simon Boehm GPU Optimization Blog** | Simon Boehm | 博客 | 高级 | GEMM优化从零到一，手把手优化矩阵乘法 |

**在线资源链接：**
- CUTLASS: https://github.com/NVIDIA/cutlass
- Triton教程: https://triton-lang.org/getting-started/tutorials/02-fused-attention.html
- Tensor Core: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#wmma
- Simon Boehm GEMM: https://siboehm.com/articles/22/CUDA-MMM
- Lei Mao: https://leimao.github.io/

#### 4.8.3 系统级协同优化

| 资料名称 | 作者/来源 | 类型 | 难度 | 推荐理由 |
|----------|----------|------|------|----------|
| **Making Deep Learning Go Brrrr From First Principles** | Horace He | 博客 | 高级 | 从第一性原理理解DL系统性能瓶颈，经典文章 |
| **A Full-Stack Approach to Accelerating LLM Training** | NVIDIA | 技术博客 | 高级 | NVIDIA全栈LLM训练加速方案 |
| **PyTorch Memory Format Guide** | PyTorch | 官方文档 | 中级 | PyTorch内存格式（Channels Last等）优化 |
| **Data Layout Optimization for GPU** | NVIDIA | 技术文档 | 高级 | GPU数据布局优化，内存对齐 |
| **Fused Kernel Design Patterns** | FasterTransformer | 开源项目 | 高级 | 算子融合设计模式，从FasterTransformer学习 |
| **NVIDIA cuDNN Developer Guide** | NVIDIA | 官方文档 | 高级 | cuDNN开发指南，理解高性能DL算子库设计 |

**在线资源链接：**
- Horace He博客: https://horace.io/brrr_intro.html
- cuDNN: https://docs.nvidia.com/deeplearning/cudnn/
- PyTorch Memory Format: https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html

#### 4.8.4 跨平台与多硬件适配

| 资料名称 | 作者/来源 | 类型 | 难度 | 推荐理由 |
|----------|----------|------|------|----------|
| **TVM: Multi-Backend Compilation** | Apache TVM | 开源项目 | 高级 | TVM多硬件后端编译，理解跨平台适配 |
| **ONNX: Open Neural Network Exchange** | ONNX Community | 开源项目 | 中级 | ONNX模型交换格式，跨框架/硬件模型部署 |
| **OpenXLA: ML Compiler Ecosystem** | Google/Intel/NVIDIA | 开源项目 | 高级 | OpenXLA编译器生态，多硬件统一编译 |
| **Intel oneDNN** | Intel | 开源项目 | 高级 | Intel深度学习算子库，CPU/GPU跨平台 |
| **AMD MIOpen** | AMD | 开源项目 | 高级 | AMD GPU深度学习算子库 |

**在线资源链接：**
- TVM: https://tvm.apache.org/
- ONNX: https://onnx.ai/
- OpenXLA: https://openxla.org/
- oneDNN: https://github.com/oneapi-src/oneDNN

---

## 5. 学习路径建议

### 5.1 推荐学习顺序

本学习路径按照知识依赖关系和由浅入深的原则设计，分为5个阶段：

```
阶段1: 基础筑基（1-2个月）
├── 深度学习基础（D2L + CS231n/CS224n）
├── CUDA/GPU编程基础（PMPP + CUDA Guide）
├── 分布式系统基础（DDIA + MIT 6.824）
└── Python/C++编程强化

阶段2: 训练系统核心（2-3个月）
├── 分布式训练策略（ZeRO/Megatron论文 + DeepSpeed源码）
├── 训练全流程（CS324 + HuggingFace教程）
├── 训练机制与稳定性（混合精度论文 + DDP/FSDP教程）
└── 训练框架源码阅读（PyTorch Distributed + DeepSpeed）

阶段3: 推理系统核心（1-2个月）
├── 推理优化技术（vLLM论文 + 源码）
├── 推理引擎（vLLM/TensorRT-LLM源码）
├── 推理服务化（Triton/Ray Serve）
└── 推理性能优化实践

阶段4: 加速技术深入（2-3个月）
├── Flash Attention（论文 + 源码 + Triton实现）
├── 量化技术（GPTQ/AWQ/SmoothQuant论文 + 实现）
├── 算子优化（CUTLASS + Triton + 自定义Kernel）
├── 计算图优化（TVM/XLA/TorchDynamo）
└── 显存优化（梯度检查点 + 选择性重计算）

阶段5: 系统级整合与进阶（2-3个月）
├── 高性能通信（NCCL + Ring AllReduce + 拓扑优化）
├── 软硬协同（GPU/NPU架构 + 算子映射 + 数据布局）
├── 性能分析（Nsight + PyTorch Profiler + Roofline）
├── 大规模训练运维（容错 + 弹性调度）
└── 综合项目实践
```

### 5.2 分阶段详细说明

#### 阶段1：基础筑基（1-2个月）

**目标**：建立深度学习、GPU编程、分布式系统的坚实基础

**核心任务**：
1. **深度学习基础**
   - 精读《动手学深度学习》(D2L)，完成所有代码练习
   - 学习CS231n/CS224n核心章节（Transformer部分必学）
   - 掌握PyTorch基本操作和训练流程

2. **CUDA/GPU编程基础**
   - 精读《Programming Massively Parallel Processors》前8章
   - 学习NVIDIA CUDA编程指南核心章节
   - 完成至少5个CUDA Kernel编写练习（矩阵乘法、归约、Scan等）
   - 学习CS179课程材料

3. **分布式系统基础**
   - 精读《Designing Data-Intensive Applications》前5章
   - 学习MIT 6.824前几个Lab（Raft共识算法）
   - 理解分布式系统核心概念：一致性、容错、分区

4. **编程能力强化**
   - C++：Effective Modern C++ + 并发编程
   - Python：高级特性 + 性能优化

**验收标准**：
- [ ] 能独立用PyTorch训练一个小型Transformer模型
- [ ] 能编写基本的CUDA Kernel（矩阵乘法、向量加法）
- [ ] 理解分布式系统核心概念（一致性、容错、CAP定理）
- [ ] 理解GPU架构基础（SM、Warp、SIMT、存储层次）

#### 阶段2：训练系统核心（2-3个月）

**目标**：深入理解大模型训练系统的核心原理和实现

**核心任务**：
1. **分布式训练策略**
   - 精读ZeRO论文系列（ZeRO, ZeRO-Offload, ZeRO-Infinity）
   - 精读Megatron-LM论文（张量并行 + 3D并行）
   - 学习FSDP文档和教程
   - 阅读DeepSpeed源码中的ZeRO实现
   - 阅读Megatron-LM源码中的张量并行实现

2. **训练全流程**
   - 学习CS324课程
   - 实践：用HuggingFace + DeepSpeed完成一次完整的SFT训练
   - 学习RLHF/DPO原理和实践
   - 理解预训练数据处理和训练监控

3. **训练机制与稳定性**
   - 精读混合精度训练论文
   - 学习PyTorch DDP/FSDP教程
   - 理解梯度累积、检查点保存恢复
   - 学习训练稳定性问题与应对策略

4. **训练框架源码阅读**
   - 阅读PyTorch torch.distributed源码（DDP部分）
   - 阅读DeepSpeed核心模块源码
   - 阅读Megatron-LM张量并行源码

**验收标准**：
- [ ] 能清晰解释ZeRO-1/2/3的原理和显存节省分析
- [ ] 能清晰解释张量并行（Column/Row Parallel）原理
- [ ] 能清晰解释3D并行的维度组合和通信分析
- [ ] 能用DeepSpeed/FSDP完成多卡分布式训练
- [ ] 能阅读并理解训练框架核心源码
- [ ] 理解混合精度训练原理和损失缩放机制

#### 阶段3：推理系统核心（1-2个月）

**目标**：深入理解大模型推理系统的核心优化技术

**核心任务**：
1. **推理优化技术**
   - 精读PagedAttention/vLLM论文
   - 精读Orca论文（Continuous Batching）
   - 学习Speculative Decoding论文
   - 理解KV Cache管理原理

2. **推理引擎源码**
   - 阅读vLLM源码（PagedAttention实现 + Continuous Batching）
   - 学习TensorRT-LLM文档和架构
   - 了解TGI、llama.cpp、SGLang

3. **推理服务化**
   - 学习NVIDIA Triton推理服务器
   - 学习Ray Serve
   - 理解推理服务API设计

4. **推理性能优化实践**
   - 部署vLLM推理服务
   - 进行推理性能Benchmark
   - 分析TTFT和TPS瓶颈

**验收标准**：
- [ ] 能清晰解释PagedAttention的原理和优势
- [ ] 能清晰解释Continuous Batching vs 静态Batching
- [ ] 能清晰解释Speculative Decoding原理
- [ ] 能部署和调优vLLM推理服务
- [ ] 理解推理性能指标（TTFT/TPS/吞吐量）

#### 阶段4：加速技术深入（2-3个月）

**目标**：掌握训推加速的核心技术和实现方法

**核心任务**：
1. **Flash Attention**
   - 精读Flash Attention v1/v2/v3论文
   - 阅读Flash Attention CUDA源码
   - 学习Triton实现的Flash Attention
   - 理解在线Softmax数学推导

2. **量化技术**
   - 精读GPTQ、AWQ、SmoothQuant论文
   - 实践：用AutoGPTQ/AutoAWQ量化一个LLM
   - 理解INT8/INT4/FP8量化原理
   - 学习LLM.int8()方法

3. **算子优化**
   - 学习CUTLASS GEMM实现
   - 用Triton编写自定义算子
   - 学习算子融合策略（FasterTransformer）
   - 练习编写自定义CUDA Kernel

4. **计算图优化**
   - 学习TVM图优化Pass
   - 学习XLA编译器
   - 学习TorchDynamo + TorchInductor
   - 理解常量折叠、算子重排等优化

5. **显存优化**
   - 学习梯度检查点原理
   - 学习选择性重计算策略
   - 理解PyTorch内存管理机制

**验收标准**：
- [ ] 能清晰解释Flash Attention的IO复杂度分析
- [ ] 能用Triton编写Flash Attention的简化版本
- [ ] 能用GPTQ/AWQ量化LLM并评估精度损失
- [ ] 能编写自定义CUDA/Triton算子
- [ ] 理解计算图优化的常见Pass
- [ ] 理解梯度检查点的显存-计算权衡

#### 阶段5：系统级整合与进阶（2-3个月）

**目标**：掌握高性能通信、软硬协同和性能分析，形成系统级能力

**核心任务**：
1. **高性能通信**
   - 深入学习NCCL文档和Ring AllReduce算法
   - 理解通信拓扑优化
   - 学习通信-计算重叠策略
   - 理解NVLink/RDMA等高速互联技术

2. **软硬协同优化**
   - 深入学习GPU架构（Hopper/Ampere白皮书）
   - 学习Ascend NPU架构和AscendC编程
   - 学习数据布局优化
   - 理解算子到硬件的映射

3. **性能分析**
   - 学习Nsight Systems/Compute使用
   - 学习PyTorch Profiler使用
   - 实践Roofline分析
   - 进行端到端性能瓶颈分析

4. **大规模训练运维**
   - 学习容错恢复机制
   - 学习弹性调度
   - 理解集群资源管理

5. **综合项目实践**
   - 完成一个端到端的训练优化项目
   - 完成一个端到端的推理优化项目
   - 完成一个自定义算子开发项目

**验收标准**：
- [ ] 能清晰解释Ring AllReduce算法和通信量分析
- [ ] 能使用Nsight Systems进行系统级性能分析
- [ ] 能进行Roofline分析并识别瓶颈类型
- [ ] 能分析训练系统的计算/通信/显存瓶颈
- [ ] 理解GPU/NPU架构特性对系统优化的影响
- [ ] 能完成端到端的训推系统优化项目

---

## 6. 推荐学习时间表

### 6.1 6个月紧凑学习计划（每天3-4小时）

| 月份 | 主题 | 核心内容 | 关键产出 |
|------|------|----------|----------|
| **第1月** | 基础筑基 | D2L + CUDA基础 + 分布式概念 | PyTorch训练 + 基本CUDA Kernel |
| **第2月** | 训练系统I | 分布式训练策略 + 训练全流程 | 多卡训练实践 + ZeRO/Megatron论文笔记 |
| **第3月** | 训练系统II | 训练机制 + 框架源码 + 推理基础 | DeepSpeed源码阅读 + vLLM部署 |
| **第4月** | 推理系统 | 推理优化技术 + 推理引擎源码 | vLLM源码分析 + 推理Benchmark |
| **第5月** | 加速技术 | Flash Attention + 量化 + 算子优化 | Triton算子开发 + 量化实践 |
| **第6月** | 系统整合 | 通信优化 + 软硬协同 + 性能分析 | 端到端优化项目 |

### 6.2 12个月深度学习计划（每天2-3小时）

| 阶段 | 月份 | 主题 | 核心内容 |
|------|------|------|----------|
| **基础** | 1-2 | 深度学习+GPU编程 | D2L, CS231n/CS224n, PMPP, CUDA Guide |
| **基础** | 3 | 分布式系统 | DDIA, MIT 6.824, CS149 |
| **训练** | 4-5 | 分布式训练 | ZeRO/Megatron论文, DeepSpeed源码, DDP/FSDP |
| **训练** | 6 | 训练全流程 | CS324, SFT/RLHF实践, 训练稳定性 |
| **推理** | 7-8 | 推理系统 | vLLM论文+源码, TensorRT-LLM, 推理服务化 |
| **加速** | 9-10 | 加速技术 | Flash Attention, 量化, 算子优化, 图优化 |
| **整合** | 11 | 通信+软硬协同 | NCCL, GPU/NPU架构, 性能分析 |
| **整合** | 12 | 综合实践 | 端到端训推系统优化项目 |

### 6.3 关键里程碑

| 里程碑 | 时间点 | 验收标准 |
|--------|--------|----------|
| M1: 基础过关 | 第1-2月 | 能训练Transformer + 写CUDA Kernel |
| M2: 训练系统 | 第4-5月 | 能用DeepSpeed/FSDP做分布式训练 |
| M3: 推理系统 | 第7-8月 | 能部署调优vLLM推理服务 |
| M4: 加速技术 | 第9-10月 | 能写Triton算子 + 量化LLM |
| M5: 系统整合 | 第11-12月 | 能做端到端性能分析与优化 |

---

## 附录：核心论文阅读清单

按优先级排序的必读论文清单：

### 第一优先级（核心必读）
1. **ZeRO: Memory Optimizations Toward Training Trillion Parameter Models** (Rajbhandari et al., 2019)
2. **Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism** (Shoeybi et al., 2019)
3. **Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM** (Narayanan et al., 2021)
4. **Efficient Memory Management for Large Language Model Serving with PagedAttention** (Kwon et al., 2023)
5. **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness** (Dao et al., 2022)
6. **Mixed Precision Training** (Micikevicius et al., 2018)

### 第二优先级（重要参考）
7. **FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning** (Dao, 2023)
8. **GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers** (Frantar et al., 2023)
9. **AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration** (Lin et al., 2024)
10. **Orca: A Distributed Serving System for Transformer-Based Generative Models** (Yu et al., OSDI 2022)
11. **Fast Inference from Transformers via Speculative Decoding** (Leviathan et al., 2023)
12. **Training Deep Nets with Sublinear Memory Cost** (Chen et al., 2016)
13. **Scaling Laws for Neural Language Models** (Kaplan et al., 2020)
14. **Direct Preference Optimization: Your Language Model is Secretly a Reward Model** (Rafailov et al., 2023)

### 第三优先级（扩展阅读）
15. **ZeRO-Offload: Democratizing Billion-Scale Model Training** (Ren et al., 2021)
16. **SmoothQuant: Accurate and Efficient Post-Training Quantization for LLMs** (Xiao et al., 2023)
17. **LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale** (Dettmers et al., 2022)
18. **FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-Precision** (Dao et al., 2024)
19. **Medusa: Simple LLM Inference Acceleration Framework** (Cai et al., 2024)
20. **Training Compute-Optimal Large Language Models (Chinchilla)** (Hoffmann et al., 2022)
21. **Roofline: An Insightful Visual Performance Model** (Williams et al., 2009)
22. **TVM: An Automated End-to-End Optimizing Compiler for DL** (Chen et al., 2018)
23. **Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations** (Tillet et al., 2019)
24. **Deep Gradient Compression** (Lin et al., 2018)
25. **On the Stability of Large Language Model Training** (Li et al., 2024)

---

## 附录：核心开源项目清单

| 项目名 | 组织 | 链接 | 学习重点 |
|--------|------|------|----------|
| **DeepSpeed** | Microsoft | https://github.com/microsoft/DeepSpeed | ZeRO实现, 3D并行编排 |
| **Megatron-LM** | NVIDIA | https://github.com/NVIDIA/Megatron-LM | 张量并行, 流水线并行 |
| **vLLM** | UC Berkeley | https://github.com/vllm-project/vllm | PagedAttention, Continuous Batching |
| **FlashAttention** | Tri Dao | https://github.com/Dao-AILab/flash-attention | IO感知注意力实现 |
| **Triton** | OpenAI | https://github.com/openai/triton | GPU编程语言, Kernel开发 |
| **CUTLASS** | NVIDIA | https://github.com/NVIDIA/cutlass | GEMM模板库, 矩阵乘优化 |
| **TensorRT-LLM** | NVIDIA | https://github.com/NVIDIA/TensorRT-LLM | 推理引擎, 算子融合 |
| **FasterTransformer** | NVIDIA | https://github.com/NVIDIA/FasterTransformer | 算子融合设计模式 |
| **PyTorch** | Meta | https://github.com/pytorch/pytorch | DDP/FSDP/Profiler |
| **NCCL** | NVIDIA | https://github.com/NVIDIA/nccl | 集合通信库 |
| **TVM** | Apache | https://github.com/apache/tvm | 深度学习编译器 |
| **Ray** | Anyscale | https://github.com/ray-project/ray | 分布式计算框架 |
| **SGLang** | LMSYS | https://github.com/sgl-project/sglang | 新一代推理引擎 |
| **AutoGPTQ** | Community | https://github.com/AutoGPTQ/AutoGPTQ | GPTQ量化实现 |
| **AutoAWQ** | MIT HAN Lab | https://github.com/casper-hansen/AutoAWQ | AWQ量化实现 |
| **BitsAndBytes** | T. Dettmers | https://github.com/bitsandbytes-foundation/bitsandbytes | INT8/INT4量化 |
| **LLaMA-Factory** | hiyouga | https://github.com/hiyouga/LLaMA-Factory | LLM微调框架 |
| **ColossalAI** | HPC-AI Tech | https://github.com/hpcaitech/ColossalAI | 大模型训练系统 |
| **Triton Server** | NVIDIA | https://github.com/triton-inference-server/server | 推理服务器 |
| **llama.cpp** | G. Gerganov | https://github.com/ggerganov/llama.cpp | CPU/GPU推理引擎 |

---

## 附录：推荐博客与技术资源

| 资源名 | 链接 | 推荐内容 |
|--------|------|----------|
| **HuggingFace Blog** | https://huggingface.co/blog | 训练/推理优化实践文章 |
| **DeepSpeed Blog** | https://www.deepspeed.ai/blog/ | 训练系统优化实践 |
| **NVIDIA Developer Blog** | https://developer.nvidia.com/blog | GPU优化技术文章 |
| **PyTorch Blog** | https://pytorch.org/blog/ | PyTorch新特性与优化 |
| **LMSYS Blog** | https://lmsys.org/blog/ | 推理系统研究 |
| **EleutherAI Blog** | https://blog.eleuther.ai/ | LLM技术深度分析 |
| **Horace He** | https://horace.io/brrr_intro.html | DL系统性能分析经典 |
| **Lei Mao** | https://leimao.github.io/ | GPU/CUDA深度技术 |
| **Simon Boehm** | https://siboehm.com/ | GEMM优化教程 |
| **Chipressa** | YouTube | GPU架构深度分析 |
| **GPU Mode** | https://www.youtube.com/@GPUMODE | CUDA/Triton编程社区 |
| **Andrew Gibiansky** | https://andrew.gibiansky.com/ | Ring AllReduce推导 |
| **Sebastian Raschka** | https://magazine.sebastianraschka.com/ | LLM训练实践指南 |
| **Lil'Log** | https://lilianweng.github.io/ | AI技术综述 |

---

> **文档说明**：本文档为AI Infra-AI训推系统岗位的完整学习计划，覆盖了岗位岗位能力要求中的全部4个专业知识领域和3个关键技能领域。学习资料涵盖书籍、大学课程、博客、论文和开源项目，从基础到高级全面覆盖。建议按照第5章的学习路径建议，结合个人基础和可用时间，制定个性化的学习计划。
>
> **最后更新**：2026-10-12