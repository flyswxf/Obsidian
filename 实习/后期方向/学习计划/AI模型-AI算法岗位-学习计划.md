## 目录

- [一、岗位概述](#一岗位概述)
- [二、知识体系全景图（知识树）](#二知识体系全景图知识树)
- [三、分领域详细知识点梳理](#三分领域详细知识点梳理)
  - [3.1 机器学习核心理论](#31-机器学习核心理论)
  - [3.2 自然语言处理（NLP）](#32-自然语言处理nlp)
  - [3.3 计算机视觉（CV）](#33-计算机视觉cv)
  - [3.4 语音处理](#34-语音处理)
  - [3.5 编程语言熟练度](#35-编程语言熟练度)
  - [3.6 算法设计与调优经验](#36-算法设计与调优经验)
  - [3.7 系统设计与工程开发能力](#37-系统设计与工程开发能力)
- [四、分领域学习资料规划](#四分领域学习资料规划)
  - [4.1 机器学习核心理论 — 学习资料](#41-机器学习核心理论--学习资料)
  - [4.2 自然语言处理（NLP）— 学习资料](#42-自然语言处理nlp--学习资料)
  - [4.3 计算机视觉（CV）— 学习资料](#43-计算机视觉cv--学习资料)
  - [4.4 语音处理 — 学习资料](#44-语音处理--学习资料)
  - [4.5 编程语言与工程能力 — 学习资料](#45-编程语言与工程能力--学习资料)
  - [4.6 算法设计与调优 — 学习资料](#46-算法设计与调优--学习资料)
  - [4.7 系统设计与工程开发 — 学习资料](#47-系统设计与工程开发--学习资料)
- [五、学习路径建议（阶段划分与推荐顺序）](#五学习路径建议阶段划分与推荐顺序)
- [附录：关键论文清单](#附录关键论文清单)

---

## 一、岗位概述

### 1.1 岗位定位

AI模型-AI算法岗位是一个**通才型 AI 算法岗位**，要求候选人具备横跨 NLP、CV、语音三大模态的深度学习能力，同时具备扎实的机器学习数学理论基础和工程落地能力。该岗位面向大模型时代，要求候选人不仅理解现有算法，还能从数学原理出发进行算法改进和系统优化。

### 1.2 掌握程度体系

| 等级 | 能力定位 |
|------|----------|
| **入门级** | 基础概念模糊，仅停留在名词认知层面 |
| **基础级** | 能区分基本任务类型，了解代表性算法，但对原理理解模糊 |
| **进阶级** | 能推导核心算法流程，理解架构设计思想，能使用工具包完成任务 |
| **精通级** | 深入理解数学原理，能针对复杂场景设计方案，具备工程实践见解 |
| **专家级** | 具备深厚数学底蕴，能从信息论/流形学习角度剖析算法本质，能主导前沿系统架构设计 |

### 1.3 学习目标

- **短期目标（3-6个月）**：达到各知识领域进阶级水平，能复现经典算法并使用工具包完成任务
- **中期目标（6-12个月）**：达到精通级水平，深入理解数学原理，能针对复杂场景设计方案
- **长期目标（12-18个月）**：向专家级迈进，具备从底层原理出发解决工程难题的能力

---

## 二、知识体系全景图（知识树）

```
AI模型-AI算法岗位
├── 1. 机器学习核心理论
│   ├── 1.1 监督学习基础
│   │   ├── 1.1.1 线性回归 / 逻辑回归
│   │   ├── 1.1.2 分类、回归、聚类任务区分
│   │   └── 1.1.3 评估指标（Accuracy/Precision/Recall/F1/AUC）
│   ├── 1.2 无监督学习
│   │   ├── 1.2.1 K-Means 聚类
│   │   ├── 1.2.2 PCA / t-SNE 特征降维
│   │   └── 1.2.3 自编码器
│   ├── 1.3 神经网络基础
│   │   ├── 1.3.1 前向传播与反向传播推导
│   │   ├── 1.3.2 激活函数（Sigmoid/ReLU/LeakyReLU/GELU/SwiGLU）
│   │   ├── 1.3.3 损失函数（MSE/CE/Focal Loss/Triplet Loss）
│   │   └── 1.3.4 正则化（Dropout/BatchNorm/LayerNorm/Weight Decay）
│   ├── 1.4 优化算法
│   │   ├── 1.4.1 SGD / Momentum / Nesterov
│   │   ├── 1.4.2 Adagrad / Adadelta / RMSprop
│   │   ├── 1.4.3 Adam / AdamW / Lion
│   │   ├── 1.4.4 学习率调度策略（Cosine/Warmup/OneCycle）
│   │   └── 1.4.5 收敛性分析与学习率自适应机制
│   ├── 1.5 深度学习架构
│   │   ├── 1.5.1 MLP / CNN / RNN / LSTM / GRU
│   │   ├── 1.5.2 Seq2Seq 与 Encoder-Decoder
│   │   ├── 1.5.3 Attention 机制（Self/Cross/Multi-Head）
│   │   └── 1.5.4 Transformer 架构全解
│   ├── 1.6 大模型理论与训练
│   │   ├── 1.6.1 预训练目标（MLM/CLM/Prefix-LM）
│   │   ├── 1.6.2 GPT vs BERT vs T5 架构对比
│   │   ├── 1.6.3 SFT（监督微调）流程
│   │   ├── 1.6.4 RLHF / DPO 对齐训练
│   │   ├── 1.6.5 位置编码（绝对/相对/RoPE/ALiBi）
│   │   ├── 1.6.6 LayerNorm vs RMSNorm
│   │   └── 1.6.7 Scaling Laws 与涌现能力
│   ├── 1.7 分布式训练
│   │   ├── 1.7.1 数据并行（DDP）
│   │   ├── 1.7.2 模型并行（张量并行/流水线并行）
│   │   ├── 1.7.3 ZeRO 优化（ZeRO-1/2/3）
│   │   ├── 1.7.4 FSDP（完全分片数据并行）
│   │   └── 1.7.5 混合精度训练（FP16/BF16/FP8）
│   ├── 1.8 高级优化与效率
│   │   ├── 1.8.1 Flash Attention / Flash Attention 2/3
│   │   ├── 1.8.2 KV Cache 与 Paged Attention
│   │   ├── 1.8.3 模型量化（GPTQ/AWQ/SmoothQuant）
│   │   ├── 1.8.4 知识蒸馏
│   │   └── 1.8.5 梯度检查点（Gradient Checkpointing）
│   └── 1.9 数学基础
│       ├── 1.9.1 线性代数（矩阵分解/SVD/特征值）
│       ├── 1.9.2 概率论与统计（贝叶斯/极大似然/信息熵）
│       ├── 1.9.3 信息论（KL散度/交叉熵/互信息）
│       ├── 1.9.4 最优化理论（凸优化/梯度下降收敛性）
│       └── 1.9.5 流形学习与表示学习
│
├── 2. 自然语言处理（NLP）
│   ├── 2.1 文本预处理
│   │   ├── 2.1.1 分词（Jieba/NLTK/SpaCy）
│   │   ├── 2.1.2 去停用词 / 文本清洗
│   │   ├── 2.1.3 TF-IDF / BoW / N-gram
│   │   └── 2.1.4 子词算法（BPE/WordPiece/SentencePiece/Unigram）
│   ├── 2.2 词向量与表示学习
│   │   ├── 2.2.1 Word2Vec（CBOW/Skip-gram）
│   │   ├── 2.2.2 负采样与层次Softmax
│   │   ├── 2.2.3 GloVe
│   │   ├── 2.2.4 FastText
│   │   └── 2.2.5 上下文词向量（ELMo → BERT）
│   ├── 2.3 序列建模
│   │   ├── 2.3.1 RNN / LSTM / GRU 序列建模
│   │   ├── 2.3.2 Seq2Seq 与 Encoder-Decoder
│   │   ├── 2.3.3 Attention 机制（Bahdanau/Luong Attention）
│   │   └── 2.3.4 Self-Attention 数学本质（Q/K/V/缩放点积）
│   ├── 2.4 Transformer 与预训练模型
│   │   ├── 2.4.1 Transformer 架构详解
│   │   ├── 2.4.2 BERT（MLM + NSP 预训练）
│   │   ├── 2.4.3 GPT 系列（GPT-1 到 GPT-4 演进）
│   │   ├── 2.4.4 T5 / BART（Encoder-Decoder 统一）
│   │   ├── 2.4.5 LLaMA 系列（RoPE/RMSNorm/SwiGLU）
│   │   └── 2.4.6 大模型微调技术（SFT/LoRA/QLoRA/Prefix Tuning）
│   ├── 2.5 NLP 核心任务
│   │   ├── 2.5.1 文本分类 / 情感分析
│   │   ├── 2.5.2 命名实体识别（NER）
│   │   ├── 2.5.3 机器翻译
│   │   ├── 2.5.4 问答系统（QA / RAG）
│   │   ├── 2.5.5 文本摘要（抽取式 / 生成式）
│   │   └── 2.5.6 多文档摘要
│   ├── 2.6 解码策略
│   │   ├── 2.6.1 Greedy Search
│   │   ├── 2.6.2 Beam Search
│   │   ├── 2.6.3 Nucleus Sampling（Top-p）
│   │   ├── 2.6.4 Top-K Sampling
│   │   ├── 2.6.5 Temperature / Repetition Penalty
│   │   └── 2.6.6 Contrastive Search
│   ├── 2.7 评估指标
│   │   ├── 2.7.1 BLEU / ROUGE
│   │   ├── 2.7.2 F1-score / Precision / Recall
│   │   ├── 2.7.3 困惑度（Perplexity）
│   │   └── 2.7.4 人类评估与 LLM-as-Judge
│   ├── 2.8 前沿与工程
│   │   ├── 2.8.1 模型幻觉分析与缓解
│   │   ├── 2.8.2 长文本线性注意力（Linear Attention/Mamba/RWKV）
│   │   ├── 2.8.3 跨语言统一表征
│   │   ├── 2.8.4 跨模态统一表征
│   │   ├── 2.8.5 RAG（检索增强生成）
│   │   └── 2.8.6 Agent / Tool Use
│   └── 2.9 NLP 工具生态
│       ├── 2.9.1 Hugging Face Transformers
│       ├── 2.9.2 NLTK / SpaCy
│       ├── 2.9.3 Jieba 分词
│       └── 2.9.4 LangChain / LlamaIndex
│
├── 3. 计算机视觉（CV）
│   ├── 3.1 图像基础
│   │   ├── 3.1.1 RGB / 灰度图 / 色彩空间转换
│   │   ├── 3.1.2 图像滤波与边缘检测（Sobel/Canny）
│   │   ├── 3.1.3 直方图均衡化
│   │   └── 3.1.4 数据增强（翻转/裁剪/颜色抖动/Mixup/CutMix）
│   ├── 3.2 卷积神经网络
│   │   ├── 3.2.1 卷积操作数学细节（步长/填充/膨胀）
│   │   ├── 3.2.2 感受野计算
│   │   ├── 3.2.3 池化层（Max/Avg/Global Average Pooling）
│   │   ├── 3.2.4 1×1 卷积与瓶颈结构
│   │   └── 3.2.5 深度可分离卷积（MobileNet）
│   ├── 3.3 经典架构
│   │   ├── 3.3.1 LeNet / AlexNet / VGG
│   │   ├── 3.3.2 ResNet（残差连接 / 恒等映射）
│   │   ├── 3.3.3 Inception / GoogLeNet
│   │   ├── 3.3.4 MobileNet V1/V2/V3
│   │   ├── 3.3.5 EfficientNet
│   │   └── 3.3.6 DenseNet
│   ├── 3.4 目标检测
│   │   ├── 3.4.1 两阶段检测（R-CNN / Fast R-CNN / Faster R-CNN）
│   │   ├── 3.4.2 单阶段检测（YOLO 系列 / SSD / RetinaNet）
│   │   ├── 3.4.3 Anchor-free 检测（FCOS / CenterNet）
│   │   ├── 3.4.4 评估指标（mAP / IoU / PR曲线）
│   │   ├── 3.4.5 NMS（非极大值抑制）与 Soft-NMS
│   │   └── 3.4.6 YOLO Loss（坐标/类别/置信度损失）
│   ├── 3.5 图像分割
│   │   ├── 3.5.1 语义分割（FCN / U-Net / DeepLab）
│   │   ├── 3.5.2 实例分割（Mask R-CNN）
│   │   ├── 3.5.3 全景分割
│   │   └── 3.5.4 Mask R-CNN 多分支损失设计
│   ├── 3.6 Vision Transformer
│   │   ├── 3.6.1 ViT（图像 Patch 化 + 线性投影 + 位置编码）
│   │   ├── 3.6.2 Swin Transformer（层级窗口注意力）
│   │   ├── 3.6.3 DeiT（数据高效训练 ViT）
│   │   └── 3.6.4 CNN vs Transformer 优劣分析
│   ├── 3.7 多模态与自监督
│   │   ├── 3.7.1 CLIP（对比学习 / 图文对齐）
│   │   ├── 3.7.2 MAE（掩码自编码视觉预训练）
│   │   ├── 3.7.3 SimCLR / MoCo（对比学习）
│   │   ├── 3.7.4 DINO / DINOv2（自监督 ViT）
│   │   └── 3.7.5 BLIP / BLIP-2（视觉语言模型）
│   ├── 3.8 生成式模型
│   │   ├── 3.8.1 GAN（Generator/Discriminator / DCGAN / StyleGAN）
│   │   ├── 3.8.2 VAE（变分推断 / 重参数化技巧）
│   │   ├── 3.8.3 Diffusion Model（DDPM / DDIM / Stable Diffusion）
│   │   ├── 3.8.4 Score-based Generative Models
│   │   └── 3.8.5 条件生成（ControlNet / IP-Adapter）
│   ├── 3.9 CV 工程优化
│   │   ├── 3.9.1 TensorRT 加速推理
│   │   ├── 3.9.2 模型剪枝与量化
│   │   ├── 3.9.3 ONNX 导出与部署
│   │   └── 3.9.4 OpenCV 图像处理流水线
│   └── 3.10 CV 工具生态
│       ├── 3.10.1 OpenCV
│       ├── 3.10.2 PyTorch torchvision / timm
│       ├── 3.10.3 Detectron2（FAIR）
│       ├── 3.10.4 MMDetection（OpenMMLab）
│       └── 3.10.5 Ultralytics YOLO
│
├── 4. 语音处理
│   ├── 4.1 音频信号基础
│   │   ├── 4.1.1 模拟信号到数字音频（采样定理/奈奎斯特频率）
│   │   ├── 4.1.2 采样率 / 位深 / 声道数
│   │   ├── 4.1.3 分帧与加窗（Hamming/Hann 窗）
│   │   └── 4.1.4 预加重
│   ├── 4.2 声学特征提取
│   │   ├── 4.2.1 傅里叶变换（DFT/FFT/STFT）
│   │   ├── 4.2.2 频谱图与梅尔频谱图
│   │   ├── 4.2.3 Fbank（Filter Bank）特征
│   │   ├── 4.2.4 MFCC（Mel-Frequency Cepstral Coefficients）
│   │   ├── 4.2.5 Fbank vs MFCC 差异分析
│   │   └── 4.2.6 倒谱与 DCT 变换
│   ├── 4.3 语音识别（ASR）
│   │   ├── 4.3.1 ASR 基本流程（VAD → 特征 → 声学模型 → 解码）
│   │   ├── 4.3.2 VAD（语音端点检测）与 KWS（关键词唤醒）
│   │   ├── 4.3.3 CTC Loss（连接时序分类）
│   │   ├── 4.3.4 RNN-T（RNN Transducer）
│   │   ├── 4.3.5 LAS（Listen, Attend and Spell）
│   │   ├── 4.3.6 流式识别与非流式识别
│   │   └── 4.3.7 Conformer 架构
│   ├── 4.4 语音合成（TTS）
│   │   ├── 4.4.1 TTS 基本流程（文本前端 → 声学模型 → 声码器）
│   │   ├── 4.4.2 Tacotron / Tacotron2（自回归）
│   │   ├── 4.4.3 FastSpeech / FastSpeech2（非自回归）
│   │   ├── 4.4.4 声码器（WaveNet / WaveGlow / HiFi-GAN）
│   │   ├── 4.4.5 韵律建模与情感表达
│   │   └── 4.4.6 VITS（端到端 TTS）
│   ├── 4.5 语音预训练模型
│   │   ├── 4.5.1 Wav2vec 2.0（对比学习预训练）
│   │   ├── 4.5.2 HuBERT（隐含单元 BERT）
│   │   ├── 4.5.3 Whisper（OpenAI 多语言 ASR）
│   │   └── 4.5.4 语音大模型（AudioLLM）
│   ├── 4.6 语音处理前沿
│   │   ├── 4.6.1 流式识别延迟优化
│   │   ├── 4.6.2 多语种对齐与噪声鲁棒性
│   │   ├── 4.6.3 语音克隆与零样本 TTS
│   │   ├── 4.6.4 语音情感识别
│   │   └── 4.6.5 多模态融合（语音+视觉）
│   └── 4.7 语音工具生态
│       ├── 4.7.1 Hugging Face Audio / Transformers
│       ├── 4.7.2 ESPnet / Kaldi
│       ├── 4.7.3 librosa（音频处理）
│       ├── 4.7.4 WeNet（出门问问端到端语音识别）
│       └── 4.7.5 Coqui TTS / Mozilla TTS
│
├── 5. 编程语言熟练度
│   ├── 5.1 Python 深度学习编程
│   │   ├── 5.1.1 Python 面向对象（类/继承/多态/装饰器）
│   │   ├── 5.1.2 NumPy 向量化操作与广播机制
│   │   ├── 5.1.3 PyTorch Tensor 操作与 Autograd
│   │   ├── 5.1.4 自定义损失函数与 DataLoader
│   │   ├── 5.1.5 PyTorch nn.Module 自定义模型
│   │   └── 5.1.6 性能优化（profiling/内存管理/GPU利用率）
│   ├── 5.2 C++ 推理加速
│   │   ├── 5.2.1 C++ 基础与内存管理
│   │   ├── 5.2.2 CUDA 编程基础
│   │   ├── 5.2.3 TensorRT C++ API
│   │   └── 5.2.4 内存泄漏排查与性能 profiling
│   └── 5.3 工程实践
│       ├── 5.3.1 Git 版本控制
│       ├── 5.3.2 Docker 容器化
│       ├── 5.3.3 CI/CD 流水线
│       └── 5.3.4 编码规范与代码审查
│
├── 6. 算法设计与调优经验
│   ├── 6.1 经典算法复现
│   │   ├── 6.1.1 从零实现线性回归/逻辑回归
│   │   ├── 6.1.2 从零实现 CNN / RNN / Transformer
│   │   └── 6.1.3 复现 ResNet / BERT / GPT
│   ├── 6.2 超参数调优
│   │   ├── 6.2.1 学习率调度策略调优
│   │   ├── 6.2.2 Batch Size / Warmup / Weight Decay 调优
│   │   ├── 6.2.3 网格搜索 / 贝叶斯优化 / Optuna
│   │   └── 6.2.4 消融实验设计
│   ├── 6.3 算法改进
│   │   ├── 6.3.1 注意力机制改进（Linear Attention/Flash Attention）
│   │   ├── 6.3.2 损失函数设计（Focal Loss/Dice Loss/Contrastive Loss）
│   │   ├── 6.3.3 数据策略（数据增强/课程学习/难例挖掘）
│   │   └── 6.3.4 理论分析与实验对比方法论
│   └── 6.4 前沿研究
│       ├── 6.4.1 论文阅读与复现能力
│       ├── 6.4.2 顶会论文追踪（NeurIPS/ICML/ICLR/CVPR/ACL/Interspeech）
│       └── 6.4.3 新型训练范式（MoE/RAG/Agent）
│
└── 7. 系统设计与工程开发能力
    ├── 7.1 系统架构设计
    │   ├── 7.1.1 单体架构 vs 微服务架构
    │   ├── 7.1.2 模型服务化（REST/gRPC/模型推理服务）
    │   ├── 7.1.3 API 设计与版本管理
    │   └── 7.1.4 系统容量规划
    ├── 7.2 分布式系统
    │   ├── 7.2.1 分布式训练架构设计
    │   ├── 7.2.2 数据分片与负载均衡
    │   ├── 7.2.3 一致性协议（Raft/Paxos）
    │   └── 7.2.4 高可用与容灾
    ├── 7.3 性能优化
    │   ├── 7.3.1 推理加速（TensorRT/vLLM/TGI）
    │   ├── 7.3.2 模型压缩（量化/剪枝/蒸馏）
    │   ├── 7.3.3 缓存策略（KV Cache/Redis）
    │   └── 7.3.4 GPU 资源调度
    ├── 7.4 数据工程
    │   ├── 7.4.1 数据管道设计
    │   ├── 7.4.2 数据标注与质量管理
    │   ├── 7.4.3 特征存储与在线/离线特征
    │   └── 7.4.4 数据版本管理（DVC）
    └── 7.5 MLOps
        ├── 7.5.1 模型生命周期管理（MLflow/Weights & Biases）
        ├── 7.5.2 模型监控与漂移检测
        ├── 7.5.3 A/B 测试与灰度发布
        └── 7.5.4 自动化再训练流水线
```

---

## 三、分领域详细知识点梳理

### 3.1 机器学习核心理论

#### 3.1.1 监督学习基础

- **线性回归**：最小二乘法、正则化（L1/Lasso、L2/Ridge）、闭式解与梯度下降解
- **逻辑回归**：Sigmoid 函数、极大似然估计、对数损失、多分类 Softmax 回归
- **任务类型区分**：分类（二分类/多分类/多标签）、回归、聚类的基本定义与适用场景
- **评估指标体系**：
  - 分类：Accuracy、Precision、Recall、F1-score、ROC-AUC、PR曲线
  - 回归：MSE、MAE、R²
  - 交叉验证：K折交叉验证、留一法
- **偏差-方差权衡**：欠拟合与过拟合的识别与处理

#### 3.1.2 无监督学习

- **K-Means 聚类**：算法流程、K值选择（肘部法则/轮廓系数）、K-Means++ 初始化
- **层次聚类**：凝聚式与分裂式、距离度量（单连接/全连接/平均连接）
- **DBSCAN**：密度聚类、核心点/边界点/噪声点
- **PCA（主成分分析）**：协方差矩阵特征分解、降维的几何意义、方差保留比例
- **t-SNE / UMAP**：非线性降维、高维数据可视化
- **自编码器**：编码-解码结构、瓶颈层、变分自编码器（VAE）

#### 3.1.3 神经网络基础

- **前向传播**：线性变换 z = Wx + b、非线性激活 a = σ(z)、逐层传递
- **反向传播**：链式法则、计算梯度 ∂L/∂W、∂L/∂b、计算图
- **激活函数**：
  - Sigmoid：σ(x) = 1/(1+e⁻ˣ)，梯度消失问题
  - Tanh：tanh(x)，零均值但仍有梯度消失
  - ReLU：max(0, x)，解决梯度消失，但存在神经元死亡
  - LeakyReLU / ELU / GELU：ReLU 改进变体
  - SwiGLU：LLaMA 使用，门控线性单元
- **损失函数**：
  - MSE（均方误差）：回归任务
  - Cross-Entropy（交叉熵）：分类任务
  - Focal Loss：解决类别不平衡
  - Triplet Loss / Contrastive Loss：度量学习
- **正则化技术**：
  - Dropout：随机失活、推理时缩放
  - BatchNorm：批归一化、训练/推理模式差异
  - LayerNorm：层归一化、Transformer 标配
  - RMSNorm：LLaMA 使用、去掉中心化
  - Weight Decay：L2 正则化

#### 3.1.4 优化算法

- **SGD（随机梯度下降）**：小批量更新、噪声引入正则化效果
- **Momentum**：动量加速、指数加权移动平均
- **Nesterov Momentum**：预更新梯度方向
- **Adagrad**：自适应学习率、历史梯度平方累积、稀疏数据处理优势、学习率单调递减问题
- **Adadelta / RMSprop**：指数衰减平均解决 Adagrad 学习率衰减问题
- **Adam**：一阶矩+二阶矩估计、偏差校正、最常用优化器
- **AdamW**：解耦权重衰减、大模型训练标配
- **Lion**：符号驱动、内存效率更高
- **学习率调度**：
  - Warmup：线性预热防止初期不稳定
  - Cosine Annealing：余弦退火
  - OneCycleLR：超收敛策略
  - Step Decay：阶梯式衰减
- **收敛性分析**：凸优化收敛保证、非凸优化经验收敛、梯度爆炸/消失的数学根源

#### 3.1.5 深度学习架构

- **MLP（多层感知机）**：全连接层堆叠
- **CNN（卷积神经网络）**：局部连接、参数共享、平移不变性
- **RNN（循环神经网络）**：隐状态传递、时间展开、梯度消失/爆炸
- **LSTM（长短期记忆）**：遗忘门、输入门、输出门、细胞状态、解决长序列依赖的门控机制
- **GRU（门控循环单元）**：LSTM 简化版、重置门、更新门
- **Seq2Seq**：Encoder-Decoder 架构、固定长度上下文向量瓶颈
- **Attention 机制**：
  - Bahdanau Attention：加性注意力
  - Luong Attention：乘性注意力（点积）
  - Self-Attention：Q/K/V 向量交互、缩放点积注意力
  - Multi-Head Attention：多组 Q/K/V 并行计算、不同子空间关注不同模式
- **Transformer 架构**：
  - Encoder：Self-Attention + FFN + Residual + LayerNorm
  - Decoder：Masked Self-Attention + Cross-Attention + FFN
  - 位置编码：正弦/余弦编码、学习式位置编码
  - Pre-LN vs Post-LN：归一化位置对训练稳定性的影响

#### 3.1.6 大模型理论与训练

- **预训练目标**：
  - MLM（掩码语言模型）：BERT，双向编码
  - CLM（因果语言模型）：GPT，自回归生成
  - Prefix-LM：Prefix 部分双向、其余自回归
  - Span Corruption：T5，文本片段替换
- **GPT vs BERT vs T5**：
  - GPT：Decoder-only、自回归、生成任务强
  - BERT：Encoder-only、双向编码、理解任务强
  - T5：Encoder-Decoder、统一框架、迁移灵活
- **SFT（监督微调）流程**：
  - 指令数据构建、格式化
  - 全参数微调 vs 参数高效微调
  - 学习率选择、训练轮数控制
- **RLHF / DPO 对齐训练**：
  - RLHF：SFT → Reward Model → PPO 强化学习
  - DPO：直接偏好优化、简化 RLHF 流程
  - 安全对齐与价值观对齐
- **位置编码**：
  - 绝对位置编码：正弦/余弦、学习式
  - 相对位置编码：T5 Relative Position Bias
  - RoPE（旋转位置编码）：LLaMA 使用、通过旋转矩阵编码相对位置
  - ALiBi：线性偏置、外推性好
- **LayerNorm vs RMSNorm**：
  - LayerNorm：均值中心化 + 方差归一化
  - RMSNorm：仅方差归一化、计算更快
- **Scaling Laws**：模型大小/数据量/计算量的幂律关系、Chinchilla 最优配比
- **涌现能力**：规模阈值以上的能力突现

#### 3.1.7 分布式训练

- **数据并行（DDP）**：每GPU完整模型副本、梯度AllReduce同步
- **模型并行**：
  - 张量并行：将单层权重矩阵切分到多GPU（Megatron-LM）
  - 流水线并行：将不同层分配到不同GPU（GPipe/PipeDream）
- **ZeRO 优化**：
  - ZeRO-1：优化器状态分片
  - ZeRO-2：优化器状态+梯度分片
  - ZeRO-3：优化器状态+梯度+参数分片（等价于 FSDP）
- **FSDP（完全分片数据并行）**：PyTorch 原生实现、参数按需All-Gather
- **3D 并行**：数据并行+张量并行+流水线并行组合
- **混合精度训练**：
  - FP16：动态损失缩放防止溢出
  - BF16：无需损失缩放、动态范围大
  - FP8：最新精度格式、极致压缩
- **梯度累积**：小Batch Size模拟大Batch
- **梯度检查点**：用计算换内存、只保存部分中间激活

#### 3.1.8 高级优化与效率

- **Flash Attention**：
  - 核心思想：分块计算注意力、利用SRAM减少HBM访问
  - Flash Attention 2：优化并行度、减少非矩阵乘法
  - Flash Attention 3：异步化、FP8支持
- **KV Cache**：推理时缓存历史Key/Value、避免重复计算
- **Paged Attention（vLLM）**：分页管理KV Cache、减少内存碎片
- **模型量化**：
  - 训练后量化（PTQ）：GPTQ、AWQ、SmoothQuant
  - 量化感知训练（QAT）
  - INT8/INT4 量化
- **知识蒸馏**：Teacher-Student 框架、软标签蒸馏
- **推测解码**：小模型草稿+大模型验证、加速推理

#### 3.1.9 数学基础

- **线性代数**：矩阵运算、SVD分解、特征值与特征向量、矩阵求导
- **概率论与统计**：贝叶斯定理、极大似然估计、先验/后验分布、高斯分布
- **信息论**：信息熵、交叉熵、KL散度、互信息、JS散度
- **最优化理论**：凸优化基础、KKT条件、拉格朗日对偶、梯度下降收敛性证明
- **流形学习**：高维数据的低维流形假设、ISOMAP/LLE/Laplacian Eigenmaps

---

### 3.2 自然语言处理（NLP）

#### 3.2.1 文本预处理

- **分词**：
  - 中文分词：Jieba（HMM/Viterbi）、HanLP、pkuseg
  - 英文分词：NLTK word_tokenize、SpaCy tokenizer
  - 子词分词：
    - BPE（Byte Pair Encoding）：GPT 系列使用、从字符级逐步合并高频对
    - WordPiece：BERT 使用、类似 BPE 但基于似然增益选择合并
    - SentencePiece：支持 BPE 和 Unigram、语言无关
    - Unigram Language Model：概率式子词切分
- **文本清洗**：去除HTML标签、特殊字符、统一编码、大小写规范化
- **去停用词**：常见停用词表（NLTK/SpaCy 内置）、中文停用词表
- **TF-IDF**：词频-逆文档频率、衡量词重要性
- **BoW（词袋模型）**与 **N-gram**：统计语言模型基础

#### 3.2.2 词向量与表示学习

- **Word2Vec**：
  - CBOW（连续词袋）：上下文预测中心词
  - Skip-gram（跳字模型）：中心词预测上下文
  - 负采样：近似Softmax、加速训练
  - 层次Softmax：Huffman树加速
- **GloVe**：全局矩阵分解+局部上下文窗口、共现矩阵
- **FastText**：子词级词向量、OOV词处理
- **上下文词向量演进**：Word2Vec（静态）→ ELMo（双向LSTM）→ BERT（双向Transformer）

#### 3.2.3 序列建模

- **RNN 序列建模**：隐状态传递、时间步展开
- **LSTM 长序列依赖**：门控机制（遗忘门/输入门/输出门）、细胞状态长期记忆
- **Seq2Seq**：
  - Encoder 将输入序列编码为上下文向量
  - Decoder 从上下文向量生成输出序列
  - 固定长度上下文瓶颈问题
- **Attention 机制**：
  - Bahdanau Attention：加性注意力 a(s_{i-1}, h_j)
  - Luong Attention：乘性注意力 score = s_i · h_j
  - 解决信息瓶颈：动态加权聚合所有Encoder隐状态
- **Self-Attention 数学本质**：
  - Q = XW_Q, K = XW_K, V = XW_V
  - Attention(Q,K,V) = softmax(QK^T/√d_k)V
  - 缩放因子 √d_k 的作用：防止点积过大导致softmax梯度消失
  - Cross-Attention vs Self-Attention 的差异

#### 3.2.4 Transformer 与预训练模型

- **Transformer 架构详解**：
  - Multi-Head Self-Attention：多组Q/K/V并行、拼接后线性投影
  - Feed-Forward Network：两层线性变换+激活函数
  - Residual Connection：残差连接缓解梯度消失
  - Layer Normalization：层归一化稳定训练
  - Position Encoding：正弦/余弦位置编码
- **BERT**：
  - MLM（掩码语言模型）：随机掩码15%的token进行预测
  - NSP（下一句预测）：句子对关系判断
  - 预训练→微调范式
- **GPT 系列**：
  - GPT-1：无监督预训练+有监督微调
  - GPT-2：Zero-shot、更大模型
  - GPT-3：Few-shot Learning、In-context Learning
  - GPT-4：多模态、RLHF对齐
- **T5 / BART**：Encoder-Decoder统一框架、Span Corruption预训练
- **LLaMA 系列**：
  - LLaMA 1/2/3：RoPE位置编码、RMSNorm、SwiGLU激活
  - 开源权重、训练细节透明
- **大模型微调技术**：
  - 全参数SFT
  - LoRA：低秩矩阵分解、冻结原始权重
  - QLoRA：4-bit量化+LoRA、消费级GPU可训练
  - Prefix Tuning / P-Tuning：软前缀
  - Adapter：插入小型适配模块

#### 3.2.5 NLP 核心任务

- **文本分类**：情感分析、主题分类、BERT微调
- **命名实体识别（NER）**：序列标注、BIO/BIOES标注体系、CRF层
- **机器翻译**：Seq2Seq+Attention、Transformer、评估指标BLEU
- **问答系统**：
  - 抽取式QA：从文档中抽取答案片段
  - 生成式QA：生成式回答
  - RAG（检索增强生成）：检索+生成结合
- **文本摘要**：
  - 抽取式：选择关键句子
  - 生成式：Seq2Seq生成摘要
  - 多文档摘要：信息整合与去重
- **模型幻觉分析**：幻觉成因（训练数据/解码策略/知识缺失）、缓解策略

#### 3.2.6 解码策略

- **Greedy Search**：每步选概率最高的token、容易陷入局部最优
- **Beam Search**：维护Top-K候选序列、平衡质量与多样性
- **Nucleus Sampling（Top-p）**：从累积概率超过p的最小token集合中采样
- **Top-K Sampling**：从概率最高的K个token中采样
- **Temperature**：调节softmax分布的尖锐程度
- **Contrastive Search**：平衡生成质量与多样性

#### 3.2.7 评估指标

- **BLEU**：n-gram精度+长度惩罚、机器翻译标准指标
- **ROUGE**：召回率导向、文本摘要标准指标
- **F1-score**：精确率与召回率的调和平均
- **Perplexity（困惑度）**：语言模型评估、e^(交叉熵)
- **LLM-as-Judge**：使用强模型评估弱模型输出

#### 3.2.8 前沿与工程

- **长文本处理**：
  - 线性注意力：Linear Attention、Performer
  - 状态空间模型：Mamba、RWKV
  - 稀疏注意力：Longformer、BigBird
- **跨语言统一表征**：mBERT、XLM-R、多语言预训练
- **跨模态统一表征**：CLIP、Flamingo、LLaVA
- **RAG 系统**：向量数据库+检索+生成、RAG优化策略
- **Agent / Tool Use**：ReAct、Toolformer、Function Calling

#### 3.2.9 NLP 工具生态

- **Hugging Face Transformers**：模型加载、微调、推理一站式
- **NLTK / SpaCy**：传统NLP工具包
- **Jieba**：中文分词
- **LangChain / LlamaIndex**：LLM应用开发框架

---

### 3.3 计算机视觉（CV）

#### 3.3.1 图像基础

- **色彩空间**：RGB/HSV/LAB/灰度图转换公式
- **图像滤波**：高斯滤波、均值滤波、中值滤波
- **边缘检测**：Sobel算子、Canny边缘检测
- **数据增强**：
  - 几何变换：翻转、旋转、裁剪、缩放
  - 颜色变换：亮度/对比度/饱和度抖动
  - 高级增强：Mixup、CutMix、AutoAugment、RandAugment

#### 3.3.2 卷积神经网络

- **卷积操作数学细节**：
  - 输出尺寸计算：O = (I - K + 2P) / S + 1
  - 步长（Stride）与填充（Padding）
  - 膨胀卷积（Dilated Convolution）
- **感受野计算**：逐层感受野递推公式、有效感受野
- **池化层**：Max Pooling、Average Pooling、Global Average Pooling
- **1×1 卷积**：通道降维/升维、跨通道信息融合
- **深度可分离卷积**：Depthwise + Pointwise、MobileNet核心、参数量计算

#### 3.3.3 经典架构

- **LeNet**：最早的CNN、手写数字识别
- **AlexNet**：ReLU、Dropout、GPU训练、ImageNet突破
- **VGG**：统一3×3卷积堆叠、深层网络
- **ResNet**：
  - 残差连接：H(x) = F(x) + x
  - 恒等映射：解决退化问题
  - 瓶颈结构：1×1→3×3→1×1
- **Inception / GoogLeNet**：多尺度特征融合、1×1降维
- **MobileNet V1/V2/V3**：
  - V1：深度可分离卷积
  - V2：倒残差结构+线性瓶颈
  - V3：SE模块+硬件搜索激活函数
- **EfficientNet**：复合缩放（深度/宽度/分辨率协同缩放）
- **DenseNet**：密集连接、特征复用

#### 3.3.4 目标检测

- **两阶段检测**：
  - R-CNN：Selective Search + CNN + SVM
  - Fast R-CNN：ROI Pooling、端到端训练
  - Faster R-CNN：RPN（区域提议网络）、Anchor机制
- **单阶段检测**：
  - YOLO 系列：YOLOv1→YOLOv11演进、网格划分+直接回归
  - SSD：多尺度特征图检测
  - RetinaNet：Focal Loss解决类别不平衡
- **Anchor-free 检测**：FCOS、CenterNet
- **评估指标**：
  - IoU（交并比）：预测框与真实框重叠度
  - mAP：不同IoU阈值下的平均精度均值
  - PR曲线：精确率-召回率曲线
- **NMS（非极大值抑制）**：去除重叠检测框、Soft-NMS改进
- **YOLO Loss**：
  - 坐标损失：边界框回归
  - 类别损失：分类交叉熵
  - 置信度损失：物体存在概率
  - 样本不平衡处理

#### 3.3.5 图像分割

- **语义分割**：
  - FCN：全卷积网络、端到端像素级分类
  - U-Net：编码器-解码器+跳跃连接、医学图像分割
  - DeepLab V1/V2/V3/V3+：空洞卷积+ASPP模块
- **实例分割**：
  - Mask R-CNN：Faster R-CNN + 分割分支
  - 多分支损失：分类损失+检测损失+分割损失
- **全景分割**：语义分割+实例分割统一

#### 3.3.6 Vision Transformer

- **ViT（Vision Transformer）**：
  - 图像Patch化：224×224 → 16×16 patches
  - 线性投影：Patch → Embedding
  - 位置编码：学习式1D位置编码
  - 分类Token：[CLS] token
- **Swin Transformer**：
  - 层级结构：多尺度特征
  - 窗口注意力：局部窗口内计算Self-Attention
  - 移位窗口：跨窗口信息交流
- **DeiT**：数据高效训练ViT、知识蒸馏
- **CNN vs Transformer 优劣**：
  - CNN：局部归纳偏置、数据效率高
  - Transformer：全局感受野、大数据下更优
  - 混合架构：EfficientFormer、CoAtNet

#### 3.3.7 多模态与自监督

- **CLIP（Contrastive Language-Image Pre-training）**：
  - 双编码器架构：图像编码器+文本编码器
  - 对比学习：最大化匹配图文对的相似度
  - 零样本分类能力
- **MAE（Masked Autoencoders）**：
  - 随机掩码75%的图像Patch
  - 编码器只处理可见Patch
  - 解码器重建被掩码的Patch
  - 高效预训练策略
- **SimCLR / MoCo**：对比学习自监督预训练
- **DINO / DINOv2**：自监督ViT、无需标签学习强大特征
- **BLIP / BLIP-2**：视觉语言理解与生成、Q-Former连接视觉与语言

#### 3.3.8 生成式模型

- **GAN（生成对抗网络）**：
  - Generator vs Discriminator博弈
  - DCGAN、StyleGAN系列
  - 模式崩溃问题与训练技巧
- **VAE（变分自编码器）**：
  - 变分推断、ELBO
  - 重参数化技巧（Reparameterization Trick）
- **Diffusion Model**：
  - DDPM（去噪扩散概率模型）：前向加噪+反向去噪
  - DDIM：确定性采样、加速生成
  - Stable Diffusion：Latent Diffusion、文本条件生成
  - Score-based Generative Models
  - 条件生成：ControlNet、IP-Adapter

#### 3.3.9 CV 工程优化

- **TensorRT**：层融合、精度校准、推理加速
- **模型剪枝**：结构化剪枝/非结构化剪枝
- **模型量化**：INT8/FP16量化
- **ONNX**：模型格式转换与跨框架部署
- **OpenCV**：图像IO、预处理、后处理流水线

#### 3.3.10 CV 工具生态

- **OpenCV**：计算机视觉基础库
- **torchvision / timm**：PyTorch视觉模型库
- **Detectron2**：FAIR目标检测/分割框架
- **MMDetection**：OpenMMLab检测工具箱
- **Ultralytics YOLO**：YOLO系列官方实现

---

### 3.4 语音处理

#### 3.4.1 音频信号基础

- **模拟到数字转换**：
  - 采样定理（奈奎斯特定理）：采样率 ≥ 2 × 最高频率
  - 量化：连续幅度→离散值
  - 编码：PCM编码
- **音频参数**：
  - 采样率：16kHz（语音常用）、44.1kHz（CD音质）
  - 位深：16bit、24bit、32bit
  - 声道数：单声道/立体声
- **分帧与加窗**：
  - 分帧：将连续音频切分为短时帧（通常20-30ms）
  - 加窗：Hamming窗/Hann窗减少频谱泄漏
  - 帧移：通常为帧长的50%重叠
- **预加重**：高通滤波器增强高频成分

#### 3.4.2 声学特征提取

- **傅里叶变换**：
  - DFT（离散傅里叶变换）：时域→频域
  - FFT（快速傅里叶变换）：DFT的高效算法
  - STFT（短时傅里叶变换）：分帧+FFT→时频图
- **频谱图**：时间×频率×幅度的三维表示
- **梅尔频谱图**：
  - 梅尔刻度：模拟人耳对频率的非线性感知
  - Fbank（Filter Bank）特征：梅尔滤波器组输出
- **MFCC（Mel-Frequency Cepstral Coefficients）**：
  - Fbank → DCT（离散余弦变换）→ MFCC
  - 去相关性、降低特征维度
  - 通常取前12-13维 + 能量 + 一阶/二阶差分
- **Fbank vs MFCC 差异**：
  - Fbank：保留更多频谱信息、深度学习常用
  - MFCC：去相关、传统GMM-HMM系统常用
  - DCT的作用：将频谱从频域转换到倒谱域

#### 3.4.3 语音识别（ASR）

- **ASR 基本流程**：
  1. 语音输入 → VAD端点检测
  2. 特征提取（Fbank/MFCC）
  3. 声学模型（AM）：音频特征→音素/字符概率
  4. 语言模型（LM）：文本序列概率
  5. 解码搜索：AM + LM → 最终文本
- **VAD（语音端点检测）**：区分语音段与静音段、能量/过零率/深度学习VAD
- **KWS（关键词唤醒）**：低功耗实时检测预设关键词
- **CTC Loss（连接时序分类）**：
  - 解决音频序列与文本序列不等长对齐问题
  - Blank标签：允许重复和空白
  - 前向-后向算法计算所有对齐路径的概率
  - 不需要帧级别标注
- **RNN-T（RNN Transducer）**：
  - 编码网络 + 预测网络 + 联合网络
  - 流式友好、不需要完整音频输入
  - 适合实时ASR
- **LAS（Listen, Attend and Spell）**：
  - Listener（编码器）+ Speller（带Attention的解码器）
  - Attention机制解决对齐
  - 非流式、精度高
- **流式 vs 非流式识别**：
  - 流式：逐帧输入、实时输出、延迟低
  - 非流式：完整音频输入后输出、精度高
  - 延迟优化：chunk-based处理、限制Attention范围
- **Conformer**：卷积+Transformer混合架构、局部+全局特征建模

#### 3.4.4 语音合成（TTS）

- **TTS 基本流程**：
  1. 文本前端：文本规范化→音素转换→韵律预测
  2. 声学模型：文本特征→梅尔频谱图
  3. 声码器：梅尔频谱图→波形
- **Tacotron / Tacotron2**：
  - 自回归生成、Attention对齐
  - 编码器+Attention+解码器架构
  - 生成质量高但速度慢
- **FastSpeech / FastSpeech2**：
  - 非自回归、一次性生成所有帧
  - 时长预测器解决对齐
  - 速度快、稳定性高
- **声码器**：
  - WaveNet：自回归波形生成、质量极高但极慢
  - WaveGlow：基于流的并行声码器
  - HiFi-GAN：GAN式声码器、速度快质量好
- **韵律建模**：F0（基频）、能量、时长的预测与控制
- **情感表达**：情感嵌入、多情感TTS、跨情感迁移
- **VITS**：端到端TTS、声学模型+声码器统一、变分推断

#### 3.4.5 语音预训练模型

- **Wav2vec 2.0**：
  - 对比学习预训练：量化后的潜在表示对比学习
  - 掩码预测：预测被掩码的潜在表示
  - 预训练→微调范式、低资源ASR效果好
- **HuBERT**：
  - Hidden-Unit BERT
  - k-means聚类产生伪标签
  - 预测被掩码帧的聚类标签
  - 更稳定的训练过程
- **Whisper（OpenAI）**：
  - 68万小时多语言数据训练
  - 多任务：ASR+翻译+语言识别+VAD
  - 长语音处理：30秒窗口+滑动
  - 噪声鲁棒性强
  - 多语种对齐
- **AudioLLM**：语音大模型、端到端语义理解与交互

#### 3.4.6 语音处理前沿

- **流式识别延迟优化**：chunk大小选择、注意力限制、流式Conformer
- **多语种对齐**：共享编码器、多语种Whisper
- **噪声鲁棒性**：数据增强（噪声混合）、域自适应
- **语音克隆与零样本TTS**：speaker embedding、少量样本克隆
- **语音情感识别**：语音→情感分类、多模态情感识别
- **多模态融合**：语音+视觉（唇语识别/视听觉ASR）、AudioLLM

#### 3.4.7 语音工具生态

- **Hugging Face Audio**：音频模型库、ASR/TTS模型
- **ESPnet**：端到端语音处理工具包
- **Kaldi**：经典语音识别工具包（GMM-HMM时代）
- **librosa**：Python音频处理库
- **WeNet**：出门问问端到端语音识别、流式+非流式统一
- **Coqui TTS**：开源TTS引擎

---

### 3.5 编程语言熟练度

#### 3.5.1 Python 深度学习编程

- **Python 面向对象**：类定义、继承、多态、魔术方法、装饰器（@property/@staticmethod/@classmethod）
- **NumPy**：数组操作、广播机制、向量化计算（避免Python循环）、内存布局
- **PyTorch 核心**：
  - Tensor 操作：创建、索引、变形、设备迁移
  - Autograd：自动微分、计算图、梯度计算
  - nn.Module：自定义模型、参数管理、模块嵌套
  - DataLoader：数据加载、批处理、多进程加载
  - 自定义损失函数：继承nn.Module、forward实现
  - 训练循环：forward→loss→backward→step
  - 混合精度训练：torch.cuda.amp
  - 分布式训练：DDP、FSDP API
- **性能优化**：
  - Profiling：torch.profiler、cProfile
  - 内存管理：显存监控、梯度检查点
  - GPU利用率优化：DataLoader workers、pin_memory
  - 向量化操作：避免Python循环、使用Tensor操作

#### 3.5.2 C++ 推理加速

- **C++ 基础**：指针与引用、RAII、智能指针、模板
- **内存管理**：堆栈分配、内存泄漏检测（Valgrind/AddressSanitizer）
- **CUDA 编程**：核函数、线程层次、共享内存、warp
- **TensorRT**：C++推理API、引擎构建、动态shape
- **性能 Profiling**：nsight、nvprof

#### 3.5.3 工程实践

- **Git**：分支管理、Merge Request、代码审查
- **Docker**：镜像构建、容器编排、GPU支持
- **CI/CD**：自动化测试、构建、部署
- **编码规范**：PEP 8、类型注解、文档字符串

---

### 3.6 算法设计与调优经验

#### 3.6.1 经典算法复现

- 从零实现：线性回归、逻辑回归、K-Means
- 从零实现：CNN、RNN、LSTM、Transformer
- 复现经典论文：ResNet、BERT、GPT-2、ViT

#### 3.6.2 超参数调优

- 学习率调度策略调优：Warmup步数、Cosine vs Linear
- Batch Size 影响：大Batch训练的泛化问题、梯度累积
- 权重衰减系数选择
- 自动化调优：Optuna、Ray Tune、Weights & Biases Sweeps
- 消融实验设计：控制变量、统计显著性

#### 3.6.3 算法改进

- 注意力机制改进：Linear Attention、Flash Attention、稀疏注意力
- 损失函数设计：Focal Loss、Dice Loss、Contrastive Loss、Triplet Loss
- 数据策略：数据增强、课程学习、难例挖掘、主动学习
- 理论分析与实验对比：消融实验、统计检验、可视化分析

#### 3.6.4 前沿研究

- 论文阅读方法论：精读vs泛读、代码复现验证
- 顶会追踪：NeurIPS、ICML、ICLR、CVPR、ICCV、ACL、EMNLP、Interspeech
- 新型训练范式：MoE（混合专家）、RAG、Agent、多模态大模型

---

### 3.7 系统设计与工程开发能力

#### 3.7.1 系统架构设计

- 模型服务化：REST API / gRPC、模型推理服务（Triton/vLLM/TGI）
- API设计：RESTful设计原则、版本管理、限流
- 系统容量规划：QPS估算、GPU资源需求

#### 3.7.2 分布式系统

- 分布式训练架构：3D并行设计、通信优化
- 数据分片与负载均衡
- 一致性协议：Raft/Paxos基础
- 高可用：故障恢复、检查点恢复

#### 3.7.3 性能优化

- 推理加速：TensorRT、vLLM、TGI、ONNX Runtime
- 模型压缩：量化、剪枝、蒸馏的组合策略
- 缓存策略：KV Cache管理、Redis缓存
- GPU资源调度：多模型共享GPU、批处理调度

#### 3.7.4 数据工程

- 数据管道：采集→清洗→标注→训练→评估
- 数据标注：主动学习、众包标注、标注质量校验
- 特征存储：在线/离线特征一致性
- 数据版本管理：DVC、数据血缘追踪

#### 3.7.5 MLOps

- 模型生命周期管理：MLflow、W&B
- 模型监控：性能漂移检测、数据漂移检测
- A/B测试与灰度发布
- 自动化再训练流水线

---

## 四、分领域学习资料规划

### 4.1 机器学习核心理论 — 学习资料

#### 📚 书籍

| 序号 | 名称 | 作者/来源 | 难度 | 推荐理由 |
|------|------|-----------|------|----------|
| 1 | **Pattern Recognition and Machine Learning (PRML)** | Christopher Bishop, Springer 2006 | ⭐⭐⭐⭐ | 机器学习经典教材，贝叶斯视角，数学严谨。涵盖线性模型、神经网络、核方法、图模型等。适合建立扎实的ML理论基础 |
| 2 | **Deep Learning: Foundations and Concepts** | Christopher Bishop & Hugh Bishop, Springer 2024 | ⭐⭐⭐ | 2024年最新深度学习教材，Springer 2024-2025最畅销AI书籍。涵盖现代深度学习架构与技术，内容更新，适合系统学习DL |
| 3 | **Deep Learning**（"花书"） | Ian Goodfellow, Yoshua Bengio, Aaron Courville, MIT Press 2016 | ⭐⭐⭐⭐ | 深度学习领域"圣经"，涵盖线性代数、概率论、优化、CNN/RNN/Autoencoder/GAN等全面内容。理论深度极高 |
| 4 | **The Elements of Statistical Learning (ESL)** | Hastie, Tibshirani, Friedman, Springer 2009 | ⭐⭐⭐⭐⭐ | 统计学习经典，数学极其严谨。适合专家级能力追求者，从统计视角理解ML本质 |
| 5 | **Mathematics for Machine Learning** | Marc Peter Deisenroth et al., Cambridge 2020 | ⭐⭐ | ML数学基础教材，涵盖线性代数、向量微积分、概率论、优化，适合补齐数学基础 |
| 6 | **动手学深度学习（Dive into Deep Learning）** | Aston Zhang, Zachary Lipton, Li Mu, Alex Smola | ⭐⭐⭐ | 代码+理论结合，PyTorch版。涵盖从基础到Transformer和大模型的完整内容，实践导向 |

#### 🎓 大学课程

| 序号 | 名称 | 来源 | 难度 | 推荐理由 |
|------|------|------|------|----------|
| 1 | **CS229: Machine Learning** | Stanford University (Andrew Ng) | ⭐⭐⭐ | 机器学习入门经典课程，涵盖监督/无监督学习、学习理论。有完整视频和讲义。官网：https://cs229.stanford.edu/ |
| 2 | **CS336: Language Modeling from Scratch** | Stanford University (Percy Liang, Tatsunori Hashimoto) | ⭐⭐⭐⭐⭐ | 2025年最新课程，从零构建语言模型。涵盖Tokenization、Transformer架构、Scaling Laws、分布式训练、Flash Attention等前沿内容。作业实操性极强。GitHub: https://github.com/stanford-cs336/ |
| 3 | **MIT 6.S191: Introduction to Deep Learning** | MIT | ⭐⭐⭐ | MIT深度学习入门课程，每年更新。涵盖CNN、RNN、Transformer、扩散模型等。官网：http://introtodeeplearning.com/ |
| 4 | **Neural Networks: Zero to Hero** | Andrej Karpathy | ⭐⭐⭐⭐ | 从零构建神经网络→GPT的全系列课程。包括micrograd、makemore、MLP、RNN、Transformer、GPT。代码+理论完美结合。官网：https://karpathy.ai/zero-to-hero.html |
| 5 | **MIT 6.790 Machine Learning** | MIT | ⭐⭐⭐⭐ | MIT研究生ML课程，理论深度高。推荐书单与PRML和Deep Learning Foundations配合。官网：https://gradml.mit.edu/ |

#### 📝 博客与在线资源

| 序号 | 名称 | 来源 | 难度 | 推荐理由 |
|------|------|------|------|----------|
| 1 | **The Illustrated Transformer** | Jay Alammar | ⭐⭐ | 可视化讲解Transformer架构，图解Q/K/V机制，入门必读。链接：https://jalammar.github.io/illustrated-transformer/ |
| 2 | **The Annotated Transformer** | Harvard NLP | ⭐⭐⭐⭐ | 逐行代码注释Transformer原始论文，代码+论文对照学习。链接：http://nlp.seas.harvard.edu/annotated-transformer/ |
| 3 | **Attention? Attention!** | Lilian Weng | ⭐⭐⭐⭐ | 系统梳理Attention机制所有变体，从Bahdanau到Self-Attention到Flash Attention。链接：https://lilianweng.github.io/posts/2018-06-24-attention/ |
| 4 | **LLM Training: All You Need to Know** | Sebastian Raschka | ⭐⭐⭐⭐ | 详解大模型训练全流程：预训练→SFT→RLHF→LoRA。链接：https://magazine.sebastianraschka.com/p/llm-training-sft-rlhf |
| 5 | **Let's build GPT: from scratch** | Andrej Karpathy (YouTube) | ⭐⭐⭐⭐ | 2小时视频从零用代码实现GPT，深入理解Transformer每一步。链接：https://www.youtube.com/watch?v=kCc8FmEb1nY |
| 6 | **Transformer Math 101** | EleutherAI Blog | ⭐⭐⭐⭐ | 详解Transformer中的数学：参数量计算、FLOPs估算、显存占用。链接：https://blog.eleuther.ai/transformer-math/ |

#### 🔧 开源项目/代码库

| 序号 | 名称 | GitHub 链接 | 难度 | 推荐理由 |
|------|------|------------|------|----------|
| 1 | **nanoGPT** | https://github.com/karpathy/nanoGPT | ⭐⭐⭐ | Karpathy从零实现GPT的最简代码库（~300行），理解GPT核心架构的最佳实践项目 |
| 2 | **FlashAttention** | https://github.com/Dao-AILab/flash-attention | ⭐⭐⭐⭐⭐ | Flash Attention官方实现，理解IO-aware注意力计算、GPU内存优化的最佳代码库 |
| 3 | **DeepSpeed** | https://github.com/microsoft/DeepSpeed | ⭐⭐⭐⭐ | 微软分布式训练框架，ZeRO优化器官方实现。学习分布式训练、混合精度、ZeRO的工程实践 |
| 4 | **Megatron-LM** | https://github.com/NVIDIA/Megatron-LM | ⭐⭐⭐⭐⭐ | NVIDIA张量并行+流水线并行框架，学习3D并行、大模型训练架构设计的核心项目 |
| 5 | **llm.c** | https://github.com/karpathy/llm.c | ⭐⭐⭐⭐ | Karpathy用C/CUDA实现GPT训练，理解底层GPU计算和优化的绝佳项目 |
| 6 | **PyTorch FSDP Tutorial** | https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html | ⭐⭐⭐⭐ | PyTorch官方FSDP教程，学习完全分片数据并行的官方指南 |

---

### 4.2 自然语言处理（NLP）— 学习资料

#### 📚 书籍

| 序号 | 名称 | 作者/来源 | 难度 | 推荐理由 |
|------|------|-----------|------|----------|
| 1 | **Speech and Language Processing (3rd ed. draft)** | Dan Jurafsky & James H. Martin, Stanford 2026 | ⭐⭐⭐⭐ | NLP领域"圣经"，2026最新版草稿免费在线。涵盖从传统NLP（分词/句法分析）到现代深度学习NLP（BERT/GPT/LLM）的全部内容。链接：https://web.stanford.edu/~jurafsky/slp3/ |
| 2 | **Natural Language Processing with Transformers** | Lewis Tunstall, Leandro von Werra, Thomas Wolf (O'Reilly 2022) | ⭐⭐⭐ | Hugging Face团队著作，聚焦Transformer在NLP中的应用，涵盖BERT/GPT/T5微调、数据集处理、模型部署全流程 |
| 3 | **Neural Network Methods in Natural Language Processing** | Yoav Goldberg, Morgan & Claypool 2017 | ⭐⭐⭐⭐ | NLP深度学习方法经典教材，涵盖词向量、CNN/RNN for NLP、Attention等，理论深度高 |
| 4 | **Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow** | Aurélien Géron, O'Reilly 2022 (3rd ed.) | ⭐⭐ | 实践导向，NLP章节涵盖文本分类、RNN、Transformer等，适合入门 |

#### 🎓 大学课程

| 序号 | 名称 | 来源 | 难度 | 推荐理由 |
|------|------|------|------|----------|
| 1 | **CS224N: Natural Language Processing with Deep Learning** | Stanford University (Christopher Manning) | ⭐⭐⭐⭐ | NLP领域最权威课程。涵盖词向量（Word2Vec）、神经网络基础、依存句法分析、RNN/LSTM、机器翻译、Attention、Transformer、BERT/GPT、问答系统、共指消解等。有完整视频、讲义、作业。官网：https://web.stanford.edu/class/cs224n/ |
| 2 | **CS336: Language Modeling from Scratch** | Stanford University | ⭐⭐⭐⭐⭐ | 从零构建语言模型，涵盖Tokenization、Transformer架构、预训练、SFT、分布式训练等。是学习大模型NLP的最新最佳课程 |
| 3 | **11-711 Advanced NLP** | CMU (Graham Neubig) | ⭐⭐⭐⭐⭐ | CMU高级NLP课程，涵盖最新NLP研究前沿。官网：https://phontron.com/class/anlp2024/ |

#### 📝 博客与在线资源

| 序号 | 名称 | 来源 | 难度 | 推荐理由 |
|------|------|------|------|----------|
| 1 | **The Illustrated BERT, ELMo, and co.** | Jay Alammar | ⭐⭐ | 可视化讲解BERT/ELMo/GPT架构差异，理解预训练模型演进的最佳入门资料 |
| 2 | **How GPT3 Works - Visualizations and Animations** | Leo Gao | ⭐⭐⭐ | 可视化讲解GPT-3工作原理，从Token到生成的完整流程 |
| 3 | **The Illustrated Word2Vec** | Jay Alammar | ⭐⭐ | 可视化讲解Word2Vec，直觉理解词向量 |
| 4 | **Hugging Face NLP Course** | Hugging Face | ⭐⭐⭐ | 官方NLP课程，涵盖Pipeline、微调BERT/GPT、数据集处理、Tokenizer训练。链接：https://huggingface.co/learn/nlp-course |
| 5 | **RLHF / DPO Explained** | Hugging Face Blog | ⭐⭐⭐⭐ | 详解RLHF和DPO对齐训练流程，从理论到代码实现 |
| 6 | **LLaMA Fine-tuning Guide** | Sebastian Raschka | ⭐⭐⭐⭐ | 详解LLaMA架构（RoPE/RMSNorm/SwiGLU）及LoRA/QLoRA微调实践 |

#### 🔧 开源项目/代码库

| 序号 | 名称 | GitHub 链接 | 难度 | 推荐理由 |
|------|------|------------|------|----------|
| 1 | **Hugging Face Transformers** | https://github.com/huggingface/transformers | ⭐⭐⭐ | NLP生态核心库，涵盖数千个预训练模型。学习模型架构实现、微调API、Tokenizer实现的最佳代码库 |
| 2 | **PEFT (Parameter-Efficient Fine-Tuning)** | https://github.com/huggingface/peft | ⭐⭐⭐⭐ | Hugging Face参数高效微调库，LoRA/QLoRA/Prefix Tuning/Adapter的官方实现 |
| 3 | **nanoGPT** | https://github.com/karpathy/nanoGPT | ⭐⭐⭐ | 从零实现GPT训练和微调，理解GPT架构的最简代码库 |
| 4 | **LLaMA (Meta)** | https://github.com/meta-llama/llama | ⭐⭐⭐⭐ | Meta LLaMA官方实现，学习RoPE/RMSNorm/SwiGLU等现代LLM技术 |
| 5 | **tokenizers** | https://github.com/huggingface/tokenizers | ⭐⭐⭐ | Hugging Face分词库，BPE/WordPiece/SentencePiece的Rust高性能实现 |
| 6 | **vLLM** | https://github.com/vllm-project/vllm | ⭐⭐⭐⭐⭐ | 高性能LLM推理引擎，Paged Attention的官方实现。学习KV Cache管理、连续批处理等推理优化技术 |

---

### 4.3 计算机视觉（CV）— 学习资料

#### 📚 书籍

| 序号 | 名称 | 作者/来源 | 难度 | 推荐理由 |
|------|------|-----------|------|----------|
| 1 | **Computer Vision: Algorithms and Applications (2nd ed.)** | Richard Szeliski, Springer 2022 | ⭐⭐⭐⭐ | CV领域经典教材，涵盖图像形成、特征检测、分割、3D重建、深度学习等。免费在线：https://szeliski.org/Book/ |
| 2 | **Foundations of Computer Vision** | Antonio Torralba et al., MIT Press 2024 | ⭐⭐⭐ | MIT最新CV教材，从图像处理和机器学习视角讲解CV基础。免费在线：https://visionbook.mit.edu/ |
| 3 | **Deep Learning for Computer Vision** | Rajalingappaa Shanmugamani, Packt 2018 | ⭐⭐⭐ | 实践导向，涵盖图像分类、目标检测、分割、GAN等CV深度学习应用 |

#### 🎓 大学课程

| 序号 | 名称 | 来源 | 难度 | 推荐理由 |
|------|------|------|------|----------|
| 1 | **CS231n: Deep Learning for Computer Vision** | Stanford University | ⭐⭐⭐⭐ | CV深度学习最权威课程。涵盖CNN基础、反向传播、优化、CNN架构（AlexNet/VGG/ResNet）、目标检测（R-CNN/YOLO）、分割、ViT、生成模型（GAN/Diffusion）、自监督学习等。有完整视频、讲义、作业。官网：https://cs231n.github.io/ |
| 2 | **EECS 498-007: Deep Learning for Computer Vision** | University of Michigan (Justin Johnson) | ⭐⭐⭐⭐ | Michigan CV课程，内容与CS231n互补，涵盖更多ViT和前沿内容。视频在YouTube |
| 3 | **MIT 6.S191: Deep Learning** | MIT | ⭐⭐⭐ | 每年更新，包含CV相关章节（CNN、ViT、Diffusion Models） |

#### 📝 博客与在线资源

| 序号 | 名称 | 来源 | 难度 | 推荐理由 |
|------|------|------|------|----------|
| 1 | **An Image is Worth 16x16 Words: ViT Explained** | Multiple blogs | ⭐⭐⭐ | ViT论文解读，理解图像Patch化和Transformer在CV中的应用 |
| 2 | **CLIP: Connecting Text and Images** | OpenAI Blog | ⭐⭐⭐⭐ | CLIP官方博客，理解对比学习在多模态中的应用 |
| 3 | **What are Diffusion Models?** | Lilian Weng | ⭐⭐⭐⭐⭐ | 最全面的Diffusion Model解读博客，从DDPM到Score-based到Stable Diffusion。链接：https://lilianweng.github.io/posts/2021-07-11-diffusion-models/ |
| 4 | **The Annotated Diffusion Model** | Hugging Face | ⭐⭐⭐⭐ | 逐行代码实现DDPM，代码+理论对照学习 |
| 5 | **Understanding YOLO Architecture** | Multiple blogs | ⭐⭐⭐ | YOLO架构详解，理解单阶段目标检测 |

#### 🔧 开源项目/代码库

| 序号 | 名称 | GitHub 链接 | 难度 | 推荐理由 |
|------|------|------------|------|----------|
| 1 | **Detectron2** | https://github.com/facebookresearch/detectron2 | ⭐⭐⭐⭐ | FAIR目标检测/分割框架，包含Faster R-CNN/Mask R-CNN/RetinaNet等。学习检测分割架构和Loss设计的最佳实践 |
| 2 | **MMDetection** | https://github.com/open-mmlab/mmdetection | ⭐⭐⭐⭐ | OpenMMLab检测工具箱，模块化设计，支持数十种检测算法。学习不同检测架构对比 |
| 3 | **Ultralytics YOLO** | https://github.com/ultralytics/ultralytics | ⭐⭐⭐ | YOLO系列最新官方实现（YOLOv8/v11），学习端到端检测和部署 |
| 4 | **MAE (Facebook Research)** | https://github.com/facebookresearch/mae | ⭐⭐⭐⭐ | MAE官方PyTorch实现，学习掩码自编码视觉预训练 |
| 5 | **CLIP (OpenAI)** | https://github.com/openai/CLIP | ⭐⭐⭐⭐ | CLIP官方实现，学习对比学习多模态预训练 |
| 6 | **Stable Diffusion** | https://github.com/CompVis/stable-diffusion | ⭐⭐⭐⭐⭐ | Stable Diffusion官方实现，学习Latent Diffusion和条件生成 |
| 7 | **timm** | https://github.com/huggingface/pytorch-image-models | ⭐⭐⭐ | Hugging Face视觉模型库，包含ViT/Swin/ResNet/EfficientNet等数百个CV模型实现 |

---

### 4.4 语音处理 — 学习资料

#### 📚 书籍

| 序号 | 名称 | 作者/来源 | 难度 | 推荐理由 |
|------|------|-----------|------|----------|
| 1 | **Speech and Language Processing (3rd ed. draft)** | Jurafsky & Martin | ⭐⭐⭐⭐ | 语音识别章节涵盖ASR基础、声学模型、语言模型、CTC等。同时覆盖NLP，一书两用 |
| 2 | **Automatic Speech Recognition: A Deep Learning Approach** | Dong Yu & Li Deng, Springer 2015 | ⭐⭐⭐⭐ | 深度学习ASR专著，涵盖DNN-HMM、CD-DNN-HMM、深度学习在语音中的应用 |
| 3 | **Mathematical Theory of Speech Recognition** | multiple authors | ⭐⭐⭐⭐⭐ | ASR数学理论，适合专家级能力追求者 |

#### 🎓 大学课程

| 序号 | 名称 | 来源 | 难度 | 推荐理由 |
|------|------|------|------|----------|
| 1 | **Deep Learning for Human Language Processing (DLHLP)** | National Taiwan University (Hung-yi Lee) | ⭐⭐⭐⭐ | 台大李宏毅教授语音处理课程，涵盖语音特征提取、CTC、RNN-T、LAS、Tacotron、FastSpeech、Wav2vec等。视频在YouTube。链接：https://speech.ee.ntu.edu.tw/~hylee/dlhlp/2020-spring.html |
| 2 | **YSDA Speech Course** | Yandex Data School | ⭐⭐⭐⭐ | 完整语音处理课程，涵盖ASR、TTS、CTC/RNN-T解码。GitHub: https://github.com/yandexdataschool/speech_course |
| 3 | **Hugging Face Audio Course** | Hugging Face | ⭐⭐⭐ | 官方音频处理课程，涵盖ASR、TTS、音频分类。链接：https://huggingface.co/learn/audio-course |

#### 📝 博客与在线资源

| 序号 | 名称 | 来源 | 难度 | 推荐理由 |
|------|------|------|------|----------|
| 1 | **Speech Recognition (ASR) — CTC, RNN-T, Attention** | Tai Bui | ⭐⭐⭐ | 系统对比CTC/RNN-T/Attention三种ASR方法，适合理解端到端ASR |
| 2 | **Whisper Explained** | OpenAI Blog / Multiple | ⭐⭐⭐⭐ | Whisper架构详解，理解多语言ASR、长语音处理、噪声鲁棒性 |
| 3 | **Wav2vec 2.0 Explained** | Multiple blogs | ⭐⭐⭐⭐ | Wav2vec 2.0对比学习预训练详解 |
| 4 | **Faster-Whisper Guide** | Multiple | ⭐⭐⭐ | Faster-Whisper（CTranslate2加速版）使用指南，理解推理优化 |

#### 🔧 开源项目/代码库

| 序号 | 名称 | GitHub 链接 | 难度 | 推荐理由 |
|------|------|------------|------|----------|
| 1 | **Whisper (OpenAI)** | https://github.com/openai/whisper | ⭐⭐⭐⭐ | OpenAI Whisper官方实现，学习多语言ASR、长语音处理、多任务学习 |
| 2 | **ESPnet** | https://github.com/espnet/espnet | ⭐⭐⭐⭐⭐ | 端到端语音处理工具包，涵盖ASR/TTS/语音翻译/语音增强。学习CTC/RNN-T/LAS/Conformer等架构 |
| 3 | **WeNet** | https://github.com/wenet-e2e/wenet | ⭐⭐⭐⭐ | 出门问问端到端语音识别，流式+非流式统一架构。学习生产级ASR系统设计 |
| 4 | **Wav2vec 2.0 (Fairseq)** | https://github.com/facebookresearch/fairseq | ⭐⭐⭐⭐⭐ | Wav2vec 2.0官方实现，学习语音自监督预训练 |
| 5 | **Coqui TTS** | https://github.com/coqui-ai/TTS | ⭐⭐⭐ | 开源TTS引擎，包含Tacotron/FastSpeech/VITS等多种模型实现 |
| 6 | **librosa** | https://github.com/librosa/librosa | ⭐⭐ | Python音频处理库，学习音频特征提取（STFT/MFCC/Fbank） |

---

### 4.5 编程语言与工程能力 — 学习资料

#### 📚 书籍

| 序号 | 名称 | 作者/来源 | 难度 | 推荐理由 |
|------|------|-----------|------|----------|
| 1 | **Fluent Python (2nd ed.)** | Luciano Ramalho, O'Reilly 2022 | ⭐⭐⭐⭐ | Python进阶圣经，涵盖数据模型、序列、字典、函数装饰器、面向对象、并发等。提升Python编程到工程级水平 |
| 2 | **Effective Python (2nd ed.)** | Brett Slatkin, Addison-Wesley 2019 | ⭐⭐⭐ | 90条Python最佳实践，涵盖列表推导、生成器、并发、内存管理等 |
| 3 | **Deep Learning with PyTorch** | Eli Stevens, Luca Antiga, Thomas Viehmann, Manning 2020 | ⭐⭐⭐ | PyTorch实战指南，从Tensor操作到模型训练到部署全流程 |
| 4 | **Programming Massively Parallel Processors (4th ed.)** | David Kirk & Wen-mei Hwu, Morgan Kaufmann 2022 | ⭐⭐⭐⭐⭐ | CUDA/GPU编程圣经，理解GPU架构和并行计算原理。C++推理加速必读 |

#### 🎓 大学课程

| 序号 | 名称 | 来源 | 难度 | 推荐理由 |
|------|------|------|------|----------|
| 1 | **PyTorch Official Tutorials** | PyTorch | ⭐⭐⭐ | 官方教程，涵盖基础Tensor操作、训练循环、DDP/FSDP、混合精度等。链接：https://docs.pytorch.org/tutorials/ |
| 2 | **Zero to Mastery Learn PyTorch** | Daniel Bourke | ⭐⭐ | 免费PyTorch课程，从零到实战。链接：https://www.learnpytorch.io/ |
| 3 | **CS149: Parallel Computing** | Stanford | ⭐⭐⭐⭐⭐ | 并行计算课程，涵盖GPU/CUDA编程基础。学习分布式训练和推理加速的底层原理 |

#### 📝 博客与在线资源

| 序号 | 名称 | 来源 | 难度 | 推荐理由 |
|------|------|------|------|----------|
| 1 | **PyTorch Performance Tuning Guide** | PyTorch Blog | ⭐⭐⭐⭐ | PyTorch性能优化官方指南，涵盖GPU利用率、内存优化、混合精度 |
| 2 | **Efficient PyTorch** | Valerio Maggio (PyTorch Conference) | ⭐⭐⭐⭐ | PyTorch效率优化教程，涵盖profiling、内存管理、向量化 |

#### 🔧 开源项目/代码库

| 序号 | 名称 | GitHub 链接 | 难度 | 推荐理由 |
|------|------|------------|------|----------|
| 1 | **PyTorch Examples** | https://github.com/pytorch/examples | ⭐⭐⭐ | 官方示例代码，包含ImageNet训练、DCGAN、Transformer等 |
| 2 | **PyTorch FSDP Tutorial** | https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html | ⭐⭐⭐⭐ | 官方FSDP教程，学习分布式训练API |
| 3 | **TensorRT** | https://github.com/NVIDIA/TensorRT | ⭐⭐⭐⭐⭐ | NVIDIA推理加速引擎，学习C++推理部署、INT8量化、层融合 |

---

### 4.6 算法设计与调优 — 学习资料

#### 📚 书籍

| 序号 | 名称 | 作者/来源 | 难度 | 推荐理由 |
|------|------|-----------|------|----------|
| 1 | **Designing Machine Learning Systems** | Chip Huyen, Cambridge 2022 | ⭐⭐⭐⭐ | ML系统设计专著，涵盖数据管线、训练基础设施、部署、监控全流程 |
| 2 | **Machine Learning Engineering** | Andriy Burkov, 2020 | ⭐⭐⭐ | ML工程实践指南，涵盖超参数调优、模型评估、生产部署 |

#### 🎓 大学课程

| 序号 | 名称 | 来源 | 难度 | 推荐理由 |
|------|------|------|------|----------|
| 1 | **Full Stack Deep Learning** | Full Stack Deep Learning | ⭐⭐⭐⭐ | 涵盖ML项目全生命周期：数据管理、训练基础设施、部署、监控。链接：https://fullstackdeeplearning.com/ |
| 2 | **CS336: Language Modeling from Scratch** | Stanford | ⭐⭐⭐⭐⭐ | 作业要求从零实现和优化语言模型，是算法设计与调优的最佳实践课程 |

#### 📝 博客与在线资源

| 序号 | 名称 | 来源 | 难度 | 推荐理由 |
|------|------|------|------|----------|
| 1 | **A Recipe for Training Neural Networks** | Andrej Karpathy | ⭐⭐⭐⭐ | Karpathy的神经网络训练方法论，涵盖调参、数据策略、训练监控。链接：https://karpathy.github.io/2019/04/25/recipe/ |
| 2 | **How to Train Your Mixture of Experts** | Hugging Face Blog | ⭐⭐⭐⭐ | MoE训练指南，学习混合专家模型训练 |
| 3 | **LoRA: Low-Rank Adaptation Explained** | Sebastian Raschka | ⭐⭐⭐⭐ | LoRA/QLoRA原理详解和代码实践 |

#### 🔧 开源项目/代码库

| 序号 | 名称 | GitHub 链接 | 难度 | 推荐理由 |
|------|------|------------|------|----------|
| 1 | **Weights & Biases** | https://github.com/wandb/wandb | ⭐⭐⭐ | 实验追踪和超参数调优平台，学习实验管理和可视化 |
| 2 | **Optuna** | https://github.com/optuna/optuna | ⭐⭐⭐ | 自动化超参数优化框架，支持贝叶斯优化、网格搜索等 |
| 3 | **Ax (Adaptive Experimentation)** | https://github.com/facebook/Ax | ⭐⭐⭐⭐ | Meta的实验优化平台，支持贝叶斯优化和bandit算法 |

---

### 4.7 系统设计与工程开发 — 学习资料

#### 📚 书籍

| 序号 | 名称 | 作者/来源 | 难度 | 推荐理由 |
|------|------|-----------|------|----------|
| 1 | **Designing Data-Intensive Applications** | Martin Kleppmann, O'Reilly 2017 | ⭐⭐⭐⭐ | 分布式系统设计圣经，涵盖数据存储、复制、分区、一致性、事务等。系统设计面试和工程实践必读 |
| 2 | **Machine Learning System Design Interview** | Ali Aminian, Alex Xu, 2023 | ⭐⭐⭐ | ML系统设计面试指南，涵盖推荐系统/搜索/广告等ML系统架构设计 |
| 3 | **Building Machine Learning Powered Applications** | Emmanuel Ameisen, O'Reilly 2020 | ⭐⭐⭐ | ML应用开发全流程，从原型到生产 |

#### 🎓 大学课程

| 序号 | 名称 | 来源 | 难度 | 推荐理由 |
|------|------|------|------|----------|
| 1 | **MIT 6.824: Distributed Systems** | MIT | ⭐⭐⭐⭐⭐ | 分布式系统经典课程，涵盖Raft/Paxos一致性协议、MapReduce、分布式事务。学习分布式训练系统设计的理论基础 |
| 2 | **Full Stack Deep Learning** | FSDL | ⭐⭐⭐⭐ | ML系统全栈课程，涵盖基础设施、部署、监控 |
| 3 | **CS149: Parallel Computing** | Stanford | ⭐⭐⭐⭐⭐ | 并行计算课程，理解GPU/多核/分布式计算原理 |

#### 📝 博客与在线资源

| 序号 | 名称 | 来源 | 难度 | 推荐理由 |
|------|------|------|------|----------|
| 1 | **Serving LLMs at Scale** | Anyscale Blog | ⭐⭐⭐⭐ | 大规模LLM推理服务架构设计，涵盖vLLM/TGI/推理优化 |
| 2 | **Continuous Batching for LLM Inference** | AnyScale Blog | ⭐⭐⭐⭐ | 连续批处理技术详解，理解vLLM推理优化核心 |
| 3 | **ML Production Architecture** | Chip Huyen Blog | ⭐⭐⭐⭐ | ML生产系统架构设计，涵盖数据管线、模型服务、监控 |

#### 🔧 开源项目/代码库

| 序号 | 名称 | GitHub 链接 | 难度 | 推荐理由 |
|------|------|------------|------|----------|
| 1 | **vLLM** | https://github.com/vllm-project/vllm | ⭐⭐⭐⭐⭐ | 高性能LLM推理引擎，Paged Attention + 连续批处理。学习推理系统设计的最佳项目 |
| 2 | **Triton Inference Server** | https://github.com/triton-inference-server/server | ⭐⭐⭐⭐ | NVIDIA模型推理服务器，学习多模型管理、动态批处理、GPU调度 |
| 3 | **MLflow** | https://github.com/mlflow/mlflow | ⭐⭐⭐ | ML生命周期管理平台，学习模型版本管理、实验追踪、模型注册 |
| 4 | **Ray** | https://github.com/ray-project/ray | ⭐⭐⭐⭐⭐ | 分布式计算框架，学习分布式训练/推理/服务的统一架构 |
| 5 | **Kubeflow** | https://github.com/kubeflow/kubeflow | ⭐⭐⭐⭐ | Kubernetes上的ML平台，学习ML工作流编排和MLOps |

---

## 五、学习路径建议（阶段划分与推荐顺序）

### 📌 阶段一：基础筑基期（第1-3个月）

**目标**：建立机器学习与深度学习理论基础，掌握Python/PyTorch编程

| 周次 | 学习内容 | 推荐资料 | 产出 |
|------|----------|----------|------|
| 1-2 | ML基础：监督/无监督学习、线性回归/逻辑回归、评估指标 | CS229前半部分 + PRML第1-4章 | 能推导线性回归和逻辑回归 |
| 3-4 | 神经网络基础：前向/反向传播、激活函数、损失函数 | Karpathy "Neural Networks Zero to Hero"前3讲 + 花书第6章 | 从零实现MLP并训练 |
| 5-6 | 优化算法：SGD/Adam/学习率调度、正则化 | 花书第8章 + PyTorch官方教程 | 理解各优化器差异，能调学习率 |
| 7-8 | CNN基础：卷积操作、池化、经典架构 | CS231n前半部分 + 动手学深度学习CV章节 | 用PyTorch实现ResNet并在CIFAR-10训练 |
| 9-10 | RNN/LSTM基础：序列建模、Seq2Seq | CS224N前半部分 + 花书第10章 | 实现LSTM和Seq2Seq |
| 11-12 | Transformer架构：Self-Attention、Multi-Head | The Illustrated Transformer + The Annotated Transformer | 从零实现Transformer |

### 📌 阶段二：领域深入期（第4-8个月）

**目标**：深入NLP、CV、语音三大领域，达到2-精通级水平

| 月份 | 学习内容 | 推荐资料 | 产出 |
|------|----------|----------|------|
| 第4月 | NLP深入：Word2Vec、BERT/GPT、预训练微调 | CS224N完整课程 + Hugging Face NLP Course | 微调BERT完成NER和情感分析任务 |
| 第5月 | NLP进阶：大模型（LLaMA）、LoRA/QLoRA微调、解码策略 | CS336课程 + nanoGPT + PEFT库 | 用LoRA微调LLaMA完成特定任务 |
| 第6月 | CV深入：目标检测（YOLO/Faster R-CNN）、分割（U-Net/Mask R-CNN） | CS231n后半部分 + Detectron2/MMDetection | 训练YOLO完成目标检测 |
| 第7月 | CV进阶：ViT、CLIP、MAE、Diffusion Model | EECS 498 + MAE/CLIP/Stable Diffusion代码 | 复现ViT或CLIP |
| 第8月 | 语音处理：音频特征、CTC、RNN-T、TTS | DLHLP课程 + ESPnet/WeNet + Whisper | 训练ASR模型或微调Whisper |

### 📌 阶段三：高级进阶期（第9-14个月）

**目标**：达到3-专家级水平，具备算法改进和系统设计能力

| 月份 | 学习内容 | 推荐资料 | 产出 |
|------|----------|----------|------|
| 第9月 | 分布式训练：DDP/FSDP/ZeRO/3D并行 | DeepSpeed + Megatron-LM + PyTorch FSDP教程 | 使用DeepSpeed训练大模型 |
| 第10月 | 推理优化：Flash Attention、KV Cache、量化、vLLM | FlashAttention代码 + vLLM代码 | 部署优化后的推理服务 |
| 第11月 | 算法改进：注意力变体、损失函数设计、新型训练范式 | 论文阅读 + 代码复现 | 设计并实现一个算法改进 |
| 第12月 | 系统设计：模型服务化、MLOps、分布式系统 | Designing Data-Intensive Applications + Full Stack DL | 设计完整的ML系统架构 |
| 第13月 | 跨领域融合：多模态（CLIP/BLIP）、AudioLLM | CLIP/BLIP/Whisper论文+代码 | 理解跨模态统一表征 |
| 第14月 | 前沿研究：顶会论文阅读、新范式追踪 | NeurIPS/ICML/ICLR/CVPR/ACL最新论文 | 复现一篇前沿论文 |

### 📌 阶段四：实战打磨期（第15-18个月）

**目标**：综合能力提升，达到专家级水平

| 月份 | 学习内容 | 推荐资料 | 产出 |
|------|----------|----------|------|
| 第15月 | C++/CUDA推理加速 | CUDA编程圣经 + TensorRT | 用C++/TensorRT部署模型 |
| 第16月 | 大规模训练实战 | llm.c + Megatron-DeepSpeed | 从零训练或微调大模型 |
| 第17月 | 端到端项目：从数据到部署 | 综合所有资料 | 完成一个完整的AI系统项目 |
| 第18月 | 综合复习：知识点查漏补缺 | 本文档所有知识点 | 达到3-专家级能力标准 |

---

### 📊 学习优先级矩阵

```
                    高优先级（核心必学）
                         │
    ┌────────────────────┼────────────────────┐
    │  • ML核心理论       │  • Transformer架构   │
    │  • PyTorch编程      │  • BERT/GPT/LLaMA   │
    │  • CNN/RNN/LSTM    │  • 分布式训练         │
    │  • 深度学习优化      │  • Flash Attention   │
    低基础                │  • 推理优化(vLLM)     │
                         │
低难度 ──────────────────┼──────────────────── 高难度
                         │
    │  • NLP基础任务      │  • Diffusion Model   │
    │  • 目标检测/分割    │  • 语音端到端模型     │
    │  • ASR/TTS基础     │  • 多模态融合         │
    │  • 评估指标        │  • 系统架构设计       │
    高应用                │  • CUDA编程          │
                         │
                    低优先级（进阶拓展）
```

---

### ⏰ 每日学习建议

| 时间段 | 活动 | 时长 |
|--------|------|------|
| 上午 | 理论学习（看书/看课程视频） | 2-3小时 |
| 下午 | 代码实践（复现算法/做作业） | 3-4小时 |
| 晚上 | 论文阅读/技术博客/笔记整理 | 1-2小时 |

---

## 附录：关键论文清单

### 机器学习核心理论
1. **Attention Is All You Need** (Vaswani et al., 2017) — Transformer原始论文
2. **BERT: Pre-training of Deep Bidirectional Transformers** (Devlin et al., 2018)
3. **Language Models are Few-Shot Learners** (GPT-3) (Brown et al., 2020)
4. **LLaMA: Open and Efficient Foundation Language Models** (Touvron et al., 2023)
5. **FlashAttention: Fast and Memory-Efficient Exact Attention** (Dao et al., 2022)
6. **LoRA: Low-Rank Adaptation of Large Language Models** (Hu et al., 2021)
7. **Training Language Models to Follow Instructions with Human Feedback** (RLHF) (Ouyang et al., 2022)
8. **Direct Preference Optimization** (DPO) (Rafailov et al., 2023)
9. **ZeRO: Memory Optimizations Toward Training Trillion Parameter Models** (Rajbhandari et al., 2019)

### NLP
10. **Word2Vec** (Mikolov et al., 2013)
11. **Sequence to Sequence Learning with Neural Networks** (Sutskever et al., 2014)
12. **Neural Machine Translation by Jointly Learning to Align and Translate** (Bahdanau et al., 2014)
13. **GPT-1: Improving Language Understanding by Generative Pre-Training** (Radford et al., 2018)
14. **T5: Exploring the Limits of Transfer Learning** (Raffel et al., 2019)
15. **RoFormer: Enhanced Transformer with Rotary Position Embedding** (Su et al., 2021)

### 计算机视觉
16. **Deep Residual Learning for Image Recognition** (ResNet) (He et al., 2016)
17. **You Only Look Once: Unified, Real-Time Object Detection** (YOLO) (Redmon et al., 2016)
18. **An Image is Worth 16x16 Words: A New State of the Art for Image Classification** (ViT) (Dosovitskiy et al., 2020)
19. **Swin Transformer: Hierarchical Vision Transformer using Shifted Windows** (Liu et al., 2021)
20. **Learning Transferable Visual Models from Natural Language Supervision** (CLIP) (Radford et al., 2021)
21. **Masked Autoencoders Are Scalable Vision Learners** (MAE) (He et al., 2021)
22. **Denoising Diffusion Probabilistic Models** (DDPM) (Ho et al., 2020)
23. **High-Resolution Image Synthesis with Latent Diffusion Models** (Stable Diffusion) (Rombach et al., 2022)
24. **U-Net: Convolutional Networks for Biomedical Image Segmentation** (Ronneberger et al., 2015)
25. **Mask R-CNN** (He et al., 2017)

### 语音处理
26. **Connectionist Temporal Classification: Labelling Unsegmented Sequence Data** (Graves et al., 2006)
27. **Listen, Attend and Spell** (LAS) (Chan et al., 2015)
28. **Wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations** (Baevski et al., 2020)
29. **HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction** (Hsu et al., 2021)
30. **Whisper: Robust Speech Recognition via Large-Scale Weak Supervision** (Radford et al., 2022)
31. **Tacotron: Towards End-to-End Speech Synthesis** (Wang et al., 2017)
32. **FastSpeech: Fast, Robust and Controllable Text to Speech** (Ren et al., 2019)
33. **Conformer: Convolution-augmented Transformer for Speech Recognition** (Gulati et al., 2020)

---

> **文档总结**：本学习计划覆盖了AI模型-AI算法岗位的全部7个知识领域，包含 **40+本推荐书籍**、**15+门大学课程**、**30+篇技术博客**、**30+个开源项目**、**33篇关键论文**，按照4个阶段18个月的学习路径进行规划。完成本计划后，候选人将具备从数学理论到工程实践的全面能力，达到岗位3-专家级能力标准。
