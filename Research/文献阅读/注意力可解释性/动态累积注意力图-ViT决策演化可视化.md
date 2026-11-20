> [!abstract] 一句话总结
> 在 ViT 每个 block 的自注意力内部，把计算 [class] token 时的"矩阵乘"改写成"逐元素乘"，从而保住原本被压缩掉的空间信息，再乘以由分类分数分解（监督 ViT）或维度重要性权重（自监督 ViT）得到的通道重要性系数，得到每个 block 的一张注意力图；把这些图从第一个 block 逐块累加，就第一次看到了决策注意力在 ViT 内部从形成到定型的完整演化过程——即本文提出的动态累积注意力图（Dynamic Accumulated Attention Map, DAAM）。

> [!warning] 缩写撞名提醒
> 本文方法简称 **DAAM（Dynamic Accumulated Attention Map）**，与 vault 内 [[DAAM-扩散模型cross-attention词像素归因|解释 Stable Diffusion 的那篇 DAAM]]（Diffusion Attentive Attribution Maps）**缩写撞名但完全无关**：后者聚合扩散模型 U-Net 的 cross-attention，回答"提示词里的词影响了生成图的哪些像素"；本文拆解判别式 ViT 的自注意力，回答"分类决策看的是输入图的哪些区域、以及这些区域如何逐块浮现"。下文所有 DAAM 均指本文方法。

## 问题背景

- **要解决什么问题**：ViT 已广泛用于自动驾驶、无人机语义分割、医学图像分析、高光谱图像处理等场景，而这些场景要求决策可信、可解释。视觉解释方法（visual explanation methods）的任务是产出解释图，标出支撑决策的图像区域；原文要回答的更深一层问题是：**判别性注意力在 ViT 内部是如何逐块演化、最终形成的**。
- **为何已有手段不够**，原文给出三条：
  - 面向 ViT 的解释方法（Transformer-LRP、GAE、IIA、基于扰动的方法）只能为**最后一个 block** 产出解释图，无法揭示网络内部注意力的演化过程；
  - 面向 CNN 的 CAM 类方法依赖卷积网络天然保留的特征图（空间语义信息），而 ViT 中真正用于决策的 [class] token 本身**不保存**空间语义信息，因此 CAM 类方法在 ViT 上原理性失效；
  - 自监督 ViT（DINO、XCiT、EsViT）用 memory bank 加 KNN 做预测，没有分类分数可用，现有方法无法解释这类决策。
- **注意力演化的价值**（原文动机）：注意力的形成过程能定性分析插入模块（如 cross-covariance attention（XCA）模块、LayerScale 模块、Token-to-Token（T2T）模块）对"形成对象注意力"的作用，从而指导更好的 ViT 设计；逐块比较注意力图还能识别贡献最大的 block。
- **中心问题**：能否把 ViT 内部"从输入 block 到输出 block"的注意力流（attention flow）可视化出来？

## 一、引言与动机：从"一张结果图"到"一条演化流"

原文把可解释性问题定义为"模型为了做出某个决策，学到了图像上的哪些区域"。已有解释方法能给出支撑决策的注意区域，但都给的是**终点**而非**过程**；原文认为过程本身携带两类额外信息：对象特征是如何被逐块学会的，以及每个中间 block 各自贡献了什么。

为此原文提出 DAAM，思路分四步：把分解模块（decomposition module）插入每个 ViT block 的自注意力中，在计算 [class] token 的同时构造并保存空间语义特征；用分类分数分解出通道重要性系数（channel-wise importance coefficients）；对自监督 ViT 改用维度重要性权重（dimension-wise importance weight）求系数；把空间特征与系数线性组合成每块注意力图，再逐块累积成注意力流。原文声明的四点贡献即：首次揭示注意力演化的累积式解释技术、解锁自注意力机制的分解模块、面向自监督 ViT 的维度重要性权重、以及在 7 个以全连接层（FC layer）为分类器的 ViT 与 2 个自监督 ViT 上的定性与定量验证。代码开源于 ly9802/DynamicAccumulatedAttentionMap。

**为什么需要这一步：** 若只关心"模型看了哪里"，最后一块的解释图已够用；但要回答"模型如何学会看那里"以及"某个内部模块是否加速了注意力的形成"，就必须让每个中间 block 都能产出解释图——这正是后文分解模块要解决的缺口。

## 二、预备知识：ViT 如何把一张图变成一个决策（原文 §3.1）

先用通俗语言过一遍 ViT 的前向过程：图像被切成固定大小的小方块（patch），每块经卷积变成一个向量（token），再额外加一个可学习的"汇总员"向量——[class] token（记为 $x_0$），它最终代表整张图去参与分类；所有 token 排成序列、加上可学习位置编码后送入一串串联的 transformer block，每个 block 内含多头自注意力（MHSA）模块与前馈网络（FFN）。形式化地，给定 patch 尺寸 $P$，图像 $I \in \mathbb{R}^{C \times H \times W}$ 被切成 $\frac{H \times W}{P^2}$ 个 patch 并映射为 token $\in \mathbb{R}^{D \times 1 \times 1}$，初始 token 序列为

$$M^0 = [x_0^0;\ x_1^0;\ \dots;\ x_p^0;\ \dots;\ x_N^0] + E_{\mathrm{pos}} \tag{1}$$

其中 $p > 0$ 时 $x_p^0$ 为第 $p$ 个 patch 的 token，$x_0^0$ 为 [class] token，$N = \frac{H \times W}{P^2}$ 为 patch 数，$E_{\mathrm{pos}}$ 为可学习位置编码，$D$ 为 token 维度。在第 $b$ 个 block（$1 \le b \le B$，$B$ 为 block 总数）中，$M^{b-1}$ 经 Layer Normalization 后投影出查询矩阵 $Q^b$、键矩阵 $K^b$、值矩阵 $V^b$（均 $\in \mathbb{R}^{(N+1) \times D}$），自注意力矩阵为

$$A^b = \mathrm{Softmax}\!\left(\frac{Q^b (K^b)^{\top}}{\sqrt{D}}\right) = [a_0^b;\ a_1^b;\ \dots;\ a_N^b] \tag{2}$$

其中第一行 $a_0^b \in \mathbb{R}^{1 \times (N+1)}$ 称为 class attention（类注意力），刻画 [class] token $x_0^b$ 与全部 patch token 的交互关系。原文说明多头机制不影响所提方法，故该节以单头形式书写。注意力矩阵与值矩阵相乘并经线性层与残差连接：

$$A^b V^b = \mathrm{Softmax}\!\left(\frac{Q^b (K^b)^{\top}}{\sqrt{D}}\right) V^b \tag{3}$$

$$\mathrm{Out}_{\mathrm{MHSA}}^b = \mathrm{Linear}(A^b V^b) + M^{b-1} \tag{4}$$

$$M^b = \mathrm{FFN}(\mathrm{LN}(\mathrm{Out}_{\mathrm{MHSA}}^b)) + \mathrm{Out}_{\mathrm{MHSA}}^b \tag{5}$$

其中 LN 为层归一化。最终 block 输出的 [class] token $x_0^B$ 送入 FC 层分类器预测类别。（解读注：式 (2) 即 [[缩放点积注意力 (Scaled Dot-Product Attention)|缩放点积注意力]] 的标准形式，$a_0^b$ 就是 [[自注意力机制 (Self-Attention)|自注意力]] 中"汇总员向每个 patch 打了多少分"的那一行权重；原文为简洁省略了[[多头注意力机制|多头]]拆分。）

**为什么需要这一步：** 后文所有改造都发生在"式 (3) 的乘法"这一个位置上，先看清 [class] token 是在哪一步、以什么方式丢失空间信息的，才能理解分解模块为何恰好插在那里。

## 三、分解模块：把被压扁的空间信息救回来（原文 §3.2 前半）

**问题所在。** 只有 [class] token $x_0^B$ 承担决策角色，因此要解释"为何预测为某类"，就必须可视化 $x_0^B$ 用到了哪些图像区域。把式 (1)–(5) 串起来，$x_0^B$ 可写成

$$x_0^B = \mathrm{FFN}(\mathrm{LN}(\mathrm{Linear}(T_B) + x_0^{B-1})) + \mathrm{Linear}(T_B) + x_0^{B-1} \tag{6}$$

其中 $T_B = a_0^B V_B \in \mathbb{R}^{1 \times D}$。麻烦在于：$T_B$ 是 class attention 与值矩阵的**矩阵乘**，$N+1$ 个 patch 的信息被求和压进一个 $D$ 维向量，空间语义信息在生成 [class] token 的过程中没有被保留——这正是自注意力模块成为黑箱、CAM 类方法失效的根源。

**原文的改造。** 为保住空间信息，原文把矩阵乘替换为**逐元素乘**：把 $V_B$ 看成 $D$ 个 $(N+1)$ 维向量，而 $a_0^B$ 本身就是一个 $(N+1)$ 维向量，让 $a_0^B$ 分别与这 $D$ 个向量逐元素相乘，得到矩阵

$$S^B = a_0^B \otimes V_B, \qquad T_B = \sum_{i=1}^{N+1} S^B \tag{7}$$

其中 $S^B \in \mathbb{R}^{(N+1) \times D}$，$\otimes$ 表示逐元素乘法，故 $S^B$ 与 $V_B$ 同尺寸，且按行（即对 $N+1$ 个位置）求和就还原出原来的 $T_B$。

符号逐项解释：

- $a_0^B$：第 $B$ 个 block 的 class attention，$(N+1)$ 维，是 [class] token 对"自己 + 各 patch"的注意力权重。
- $V_B$：第 $B$ 个 block 的值矩阵，$(N+1) \times D$，每行是一个位置（第 0 行为 [class] token 自身）的 $D$ 维内容。
- $S^B$：分解模块产出的空间语义特征矩阵；第 $i$ 行第 $d$ 列等于"位置 $i$ 的注意力权重 × 位置 $i$ 在第 $d$ 通道的内容"。
- $T_B$：$S^B$ 的行和，即未改造前的 [class] token 输入项，保证改造不改变原网络的数值。

直观解释（不改变数学含义）：式 (7) 相当于把原本"先加总再入账"的一笔总额，改记为"逐笔来源明细"——$S^B$ 中 $N+1$ 行 $D$ 列的分布，就是 [class] token 做分类时所利用的空间特征信息。

**为什么需要这一步：** 有了 $S^B$，每个 block 才第一次拥有"带空间坐标的特征图"；但它还只是"特征"而非"重要性"——哪些通道对决策真正有用，需要下一节的系数来加权。

## 四、通道重要性系数：用分类分数给每个通道打分（原文 §3.2 中，监督 ViT）

对以 FC 层为分类器的 ViT，决策依据是分类分数。给定类别 $c$ 的分数 $Y^c = X_B^0 W^c + \mathrm{bias}$（$W^c$ 为 FC 层中类别 $c$ 对应的权重向量，bias 为偏置）。对 $Y^c$ 做分解，即得分配给 $T_B$ 每个元素的通道重要性系数 $C_B^c \in \mathbb{R}^{1 \times D}$：

$$C_B^c = \frac{\partial (Y^c - \mathrm{bias})}{\partial X_B^0}\, \frac{\partial X_B^0}{\partial T_B} = W^c\, \frac{\partial X_B^0}{\partial T_B} \tag{8}$$

其中 $\frac{\partial X_B^0}{\partial T_B}$ 可由式 (6) 求得。为排除对分类分数的负贡献，$C_B$ 先过一个 ReLU 函数，再与 $S_B$ 逐元素相乘；把所有通道信息求和得到一个 $(N+1)$ 维向量，去掉第一个元素（[class] token 自身那一行），把剩余 $N$ 个元素重塑为 $\frac{H}{P} \times \frac{W}{P}$ 的图，即第 $B$ 个 block 的注意力图：

$$L_B^c = \mathrm{Reshape}\!\left(\Big(\sum_{d=0}^{D-1} \mathrm{ReLU}(C_B^c) \otimes S_B\Big)[1:]\right) \tag{9}$$

其中 $L_B^c \in \mathbb{R}^{\frac{H}{P} \times \frac{W}{P}}$，$[1:]$ 表示截去首元素。

符号逐项解释：

- $Y^c$：FC 层对类别 $c$ 输出的分类分数（标量）。
- $W^c$：FC 层权重矩阵中类别 $c$ 对应的那一行，$D$ 维。
- $C_B^c$：通道重要性系数，$1 \times D$，第 $d$ 个分量表示"第 $d$ 个通道的特征对类别 $c$ 分数的边际贡献"。
- $\mathrm{ReLU}(\cdot)$：只保留正贡献通道，抑制对分类分数起负作用的部分。
- $[1:]$ 与 $\mathrm{Reshape}$：丢弃 [class] 行、把 $N$ 个 patch 位置的值摆回二维网格。

直观解释（不改变数学含义）：式 (9) 是"空间明细 × 通道权重"再按通道汇总——$S_B$ 告诉每个位置在每个通道上贡献了多少特征，$\mathrm{ReLU}(C_B^c)$ 告诉这些通道对类别 $c$ 有多重要，两者相乘求和即得"类别 $c$ 视角下每个 patch 位置的重要性"。

**为什么需要这一步：** $S_B$ 对所有类别一视同仁，而解释必须针对"模型预测的那个类"；重要性系数正是把类别信息注入空间特征的桥梁。

## 五、逐块递归与累积：从单块图到注意力流（原文 §3.2 末）

式 (9) 只暴露了第 $B$ 块学到的空间特征，式 (6) 中 $x_0^{B-1}$ 里藏的信息仍未打开。原文把 $B$ 递归替换为 $B-1, B-2, \dots, 1$，得到

$$x_0^B = \sum_{i=1}^{B} \mathrm{FFN}(\mathrm{LN}(\mathrm{Linear}(T_i) + X_{i-1}^0)) + \sum_{i=1}^{B} \mathrm{Linear}(T_i) \tag{10}$$

并重复执行式 (6)、(7)、(9)、(8)、(10) 的替换操作，依次得到 $L_{B-1}^c, L_{B-2}^c, \dots, L_1^c$。由式 (10) 可知，$x_0^B$ 做预测时用到的是**所有 block** 学到的空间特征，因此针对类别 $c$ 的完整解释图应为各块之和：

$$L_{\mathrm{DAAM}}^c = \sum_{b=1}^{B} L_b^c \tag{11}$$

而要看"演化到第 $\bar{B}$ 块时模型已经注意到哪里"（$1 \le \bar{B} < B$），只取前缀和：

$$L_{\bar{B},\mathrm{DAAM}}^c = \sum_{b=1}^{\bar{B}} L_b^c \tag{12}$$

按原文 Fig. 1 的流程，从第一块累积到第 $b$ 块后，先做全局 max-min 归一化，再做上采样；上采样采用双线性插值。让 $\bar{B}$ 从 1 扫到 $B$，就得到一条逐帧变化的注意力流（attention flow）。

直观解释（不改变数学含义）：式 (12) 是"截至第 $\bar{B}$ 块为止，决策注意力的累积账本"；式 (11) 是账本的最终页，式 (12) 的每一页连起来就是决策的形成过程。

**为什么需要这一步：** 单块图只能回答"这一块学到了什么"，前缀累积才能回答"注意力是何时、从何处长出来的"——这正是原文相对只能解释末块的已有方法的根本差别。

## 六、维度重要性权重：没有分类分数时怎么办（原文 §3.3，自监督 ViT）

**为什么需要这一步：** DINO、XCiT 等自监督 ViT 不用 FC 层，而是用 memory bank（记忆库）机制做预测：把 [class] token 与库中特征向量做 KNN 检索，度量是余弦相似度。没有 $Y^c$ 就没有式 (8) 的分解对象，因此必须另造一套"重要性系数"。

原文假设 $x_0^B$ 与记忆库中 $K$ 个特征向量 $z_k$（$1 \le k \le K$）相似，二者同为 $D$ 维：$x_0^B = [\chi_1, \chi_2, \dots, \chi_d, \dots, \chi_D]$，$z_k = [\nu_1^k, \nu_2^k, \dots, \nu_d^k, \dots, \nu_D^k]$。决策相似度即 $\cos(x_0^B, z_k)$。对相似度的每个维度求重要性权重：

$$w_d = \frac{1}{K} \sum_{k=1}^{K} \frac{\chi_d \times \nu_d^k}{\lVert \cos(x_0^B, z_k) \rVert \cdot \lVert x_0^B \rVert \cdot \lVert z_k \rVert}, \quad d = 1, 2, \dots, D \tag{13}$$

其中 $\lVert \cdot \rVert$ 为 $L^2$ 范数。向量 $W = [w_1, w_2, \dots, w_d, \dots, w_D]$ 即可替换式 (8) 中的 $W^c$；与式 (9)、(11) 组合，得到自监督 ViT 的 DAAM：

$$L_{\mathrm{DAAM}} = \sum_{b=1}^{B} \mathrm{Reshape}\!\left(\Big(\sum_{d=1}^{D} \mathrm{ReLU}(w_d) \otimes S_b\Big)[1:]\right) \tag{14}$$

符号逐项解释：

- $z_k$：memory bank 中与 $x_0^B$ 最相似的 $K$ 个特征向量之一（KNN 的 $K$ 个邻居）。
- $\chi_d, \nu_d^k$：$x_0^B$ 与 $z_k$ 在第 $d$ 维上的分量；分子 $\chi_d \times \nu_d^k$ 是余弦相似度分子在第 $d$ 维上的分项。
- 分母 $\lVert \cos(x_0^B, z_k) \rVert \cdot \lVert x_0^B \rVert \cdot \lVert z_k \rVert$：把该分项归一化回"对相似度的贡献"口径。
- $w_d$：第 $d$ 维对决策相似度的平均重要性，充当自监督情形下的通道重要性系数。
- 式 (14) 无类别上标 $c$：自监督模型没有类别分数，解释的是"与记忆库邻居匹配"这一决策本身。

直观解释（不改变数学含义）：式 (13) 把"余弦相似度"这个标量按维度拆账再对 $K$ 个邻居取平均，得到"每一维对'认亲'贡献了多少"，从而在没有分类分数的模型上重建出式 (8) 的角色。

## 七、实验设置（原文 §4.1–4.2）

- **数据集**：ImageNet（ILSVRC2012），约 130 万训练图、验证集 50,000 图、1,000 个语义类别；定量与定性评估均在验证集上进行，验证集提供每张图的 bounding box 标注，用于评估定位能力。
- **模型（共 9 个）**：7 个以 FC 层为分类器的监督 ViT——DeiT-small-patch16、DeiT-tiny-patch16、CaiT-S24、T2T-ViT-14、T2T-ViT-24、ViT-small、ViT-base；2 个自监督 ViT——DINO-ViT-small-patch8、XCiT-m24。原文按此顺序给出报告准确率：78.98%、72.20%、83.50%、81.70%、82.60%、77.91%、83.73%、78.30%（KNN）、77.90%（KNN），对应正确预测的验证图数为 39,488、36,100、41,750、40,850、41,300、38,955、41,865、39,150、38,950。
- **实现细节**：基于 PyTorch image models 库（Timm）；验证图先缩放到 $224 \times 224$、像素缩放到 $[0,1]$，再按均值 [0.485, 0.456, 0.406]、标准差 [0.229, 0.224, 0.225] 归一化；不做额外数据增强；**定量分析只在预测正确的样本上进行**；PyTorch + Python 3.8，NVIDIA RTX 3090 GPU。

## 八、定性观察：注意力流长什么样（原文 §4.3）

- **单模型的演化细节**（Fig. 2 第一行，DeiT-small-patch16）：第 1 块时注意力尚未落在蜘蛛上；到第 6 块注意力逐渐转移到蜘蛛背部；自第 7 块起背部注意力增长加快；自第 10 块起注意区域扩大到两条后腿；第 12 块为最终输出。由此既能看到末块决策所用特征，也能看到每个 block 对特征的贡献。
- **同图跨模型对比**（Fig. 3）：DeiT-small-patch16、DINO-ViT-small-patch8、DeiT-tiny-patch16、ViT-small、ViT-base 均含 12 个 block，五者的 12 张累积图并排对比显示不同模型"工作方式不同"，内部 DAAM 各异。
- **块数的影响**（Fig. 4）：T2T-ViT-14 与 T2T-ViT-24 分别含 14、24 个 block，可用于分析 block 数量对注意力演化的影响。
- **插入模块的诊断**（Fig. 5）：XCiT-m24-p8 在 2 个 ViT block 之前堆了 24 个 XCA block，CaiT-S24 含 24 个 LayerScale block 后接 2 个 ViT block；由于 [class] token 是在 2 个 ViT block 之前才插入（而非 24 个 XCA/LayerScale 模块之前），故这两个模型只可视化 2 个 block 的注意力流。对比可见：在同一张图上，24 个 XCA 模块比 24 个 LayerScale 模块能让 ViT block **更快**聚焦到目标对象。原文据此把 DAAM 定位为"分析任意内部组件对注意力形成影响"的定性诊断工具。
- **局限伏笔**：DAAM 能为任何"存在 [class] token"的中间 block 产出解释图，但**暂时无法**解释 [class] token 尚不存在的中间块（如上述 24 个前置 XCA/LayerScale 模块）内部的注意力流。

## 九、Sanity check：热图确实依赖模型参数（原文 §4.4）

Sanity check 检验解释图对模型参数的敏感度。原文对 DeiT-small-patch16 的 MHSA 模块中线性层权重做**级联随机化**（cascading randomization）：从顶层 block 开始，依次随机化到第 3、6、9、12 块，观察末块注意力图（Fig. 6）。结果显示随参数随机化注意力图被破坏，故 DAAM 对 ViT 模型参数敏感，通过 sanity check。

## 十、与现有解释方法的对比（原文 §4.5–4.7）

### 10.1 末块定性对比

原文强调 DAAM 是第一个能解释 ViT 全部中间块的工具，而现有方法只能解释末块输出，因此对比只能放在末块进行。在预训练 DeiT-small-patch16 上、对同一批正确分类图像，与 RISE、IIA、GAE、Transformer-LRP 四种方法对比（Fig. 7，均用各自公开代码）。

### 10.2 定位能力定量对比（Table 1、Table 2）

沿用 IIA（原文引用 [29]）的实验设置，取 DAAM 注意力流的**最后一帧**与 Transformer-LRP、GAE、IIA2、IIA3 对比。指标方向以原文表注为准：AUC Pos Perturb、AUC Del、ADP 在 predicted 与 target 两种标签下均**越低越好**；AUC Neg Perturb、AUC Ins、PIC 均**越高越好**。（解读注：perturbation 类指标大致衡量"按热图重要性顺序逐块遮盖或保留像素时，类别置信度变化曲线下的面积"，PIC/ADP 为点选/平均下降类定位指标；原文未展开定义，口径以 IIA 代码设置为准。）

**Table 1（ViT-small，数值引自原文）**

| 指标 | 标签 | T-LRP | GAE | IIA2 | IIA3 | Ours |
| --- | --- | --- | --- | --- | --- | --- |
| AUC Neg Perturb | Predicted | 53.29% | 52.81% | 55.76% | 56.39% | **57.36%** |
| AUC Neg Perturb | Target | 53.93% | 53.58% | 58.71% | **59.31%** | 58.67% |
| AUC Pos Perturb | Predicted | 14.16% | 14.75% | 13.06% | 12.15% | **12.04%** |
| AUC Pos Perturb | Target | 13.08% | 14.38% | 12.97% | **11.86%** | 11.87% |
| AUC Ins | Predicted | 45.42% | 45.21% | 46.55% | 47.68% | **48.44%** |
| AUC Ins | Target | 46.12% | 45.69% | 47.83% | 48.53% | **48.95%** |
| AUC Del | Predicted | 11.28% | 11.92% | 11.18% | **10.31%** | 10.34% |
| AUC Del | Target | 11.06% | 11.69% | 10.98% | 10.16% | **9.98%** |
| PIC | Predicted | 13.67% | 8.68% | 15.49% | 17.79% | **22.16%** |
| PIC | Target | 15.00% | 10.02% | 18.14% | 19.59% | **24.23%** |
| ADP | Predicted | 51.94% | 36.98% | 36.74% | 36.04% | **32.29%** |
| ADP | Target | 50.59% | 64.72% | 39.58% | 39.43% | **35.99%** |

**Table 2（ViT-base，数值引自原文）**

| 指标 | 标签 | T-LRP | GAE | IIA2 | IIA3 | Ours |
| --- | --- | --- | --- | --- | --- | --- |
| AUC Neg Perturb | Predicted | 54.16% | 54.61% | 56.01% | 57.68% | **57.85%** |
| AUC Neg Perturb | Target | 55.04% | 55.67% | 57.47% | 58.31% | **59.10%** |
| AUC Pos Perturb | Predicted | 17.03% | 17.32% | 15.19% | 14.96% | **14.09%** |
| AUC Pos Perturb | Target | 16.04% | 16.72% | 15.81% | 15.02% | **14.92%** |
| AUC Ins | Predicted | 48.58% | 48.96% | 49.31% | 50.71% | **51.06%** |
| AUC Ins | Target | 49.19% | 49.65% | 50.49% | 51.26% | **51.71%** |
| AUC Del | Predicted | 14.20% | 14.37% | 12.89% | **12.25%** | 12.90% |
| AUC Del | Target | 13.77% | 13.99% | 13.12% | **12.38%** | 13.21% |
| PIC | Predicted | 13.37% | 23.65% | 26.18% | 30.41% | **32.84%** |
| PIC | Target | 14.97% | 25.53% | 28.97% | 31.75% | **35.23%** |
| ADP | Predicted | 54.02% | 37.84% | 33.93% | 34.05% | **32.73%** |
| ADP | Target | 56.68% | 36.09% | 31.08% | 32.64% | **28.96%** |

原文的结论表述：DAAM 在 AUC Neg Perturb（predicted）、AUC Pos Perturb（predicted）、ADP、PIC、AUC Ins、AUC Del（target）上优于全部对比方法；在 AUC Neg Perturb（target）、AUC Pos Perturb（target）、AUC Del（predicted）上优于 T-LRP 与 GAE、略逊于 IIA；总体说明 DAAM 在类别判别与定位两方面产出最准确的解释图。

### 10.3 与 CAM 类方法对比（Table 3、Fig. 8、Fig. 9）

CAM 类方法本为 CNN 设计，但社区中常被借用来解释 ViT（如 pytorch-grad-cam 的 ViT 教程）。原文在 DeiT-small-patch16、CaiT-S24-224、T2T-ViT-t-14（准确率 78.98%、83.50%、81.70%，对应 39,488、41,750、40,850 张正确预测图）上，用式 (11) 的 $L_{\mathrm{DAAM}}^c$ 与 Grad-CAM、Grad-CAM++、XGrad-CAM、Layer-CAM、Ablation-CAM、Score-CAM 六种方法对比。定位度量取平均 IoU 与定位准确率（IoU $\ge 0.5$ 记为正确）：

$$\mathrm{IoU} = \frac{\sum_{(i,j) \in (\mathrm{bbox}_e \cap \mathrm{bbox}_g)} 1}{\sum_{(i,j) \in (\mathrm{bbox}_e \cup \mathrm{bbox}_g)} 1} \tag{15}$$

其中 $\mathrm{bbox}_e$ 为估计框、$\mathrm{bbox}_g$ 为真值框；估计框由显著图以最大强度为阈值二值化后取最大连通区域的紧框得到，阈值取 0.1；计算前所有解释图先缩放回与原图相同的尺寸。

**Table 3（数值引自原文，越高越好）**

| 方法 | DeiT IoU | DeiT Acc | CaiT IoU | CaiT Acc | T2T-ViT IoU | T2T-ViT Acc |
| --- | --- | --- | --- | --- | --- | --- |
| Grad-CAM | 53.72% | 56.40% | 41.11% | 39.84% | 55.35% | 59.69% |
| Grad-CAM++ | 44.24% | 41.47% | 35.54% | 31.29% | 52.57% | 54.92% |
| XGrad-CAM | 35.31% | 26.71% | 37.70% | 30.73% | 39.45% | 32.60% |
| Layer-CAM | 18.09% | 8.73% | 22.88% | 15.85% | 29.35% | 19.15% |
| Ablation-CAM | 52.33% | 52.80% | 38.82% | 35.94% | 48.51% | 46.54% |
| Score-CAM | 51.85% | 53.86% | 38.05% | 36.01% | 52.90% | 54.73% |
| Ours | **56.45%** | **61.58%** | **42.36%** | **40.53%** | **56.08%** | **61.25%** |

定性方面（Fig. 8、Fig. 9）：六种 CAM 方法在 DeiT-small-patch16 的 12 个 block 上逐块出图时**观察不到任何演化模式**，而 DAAM 第一行的 12 张累积图清晰展示决策注意力的演化过程。

## 十一、用注意力流诊断错误分类（原文 §4.8）

DAAM 还可作为分析"模型为何判错"的视觉手段。原文不重训模型，直接用预训练 DeiT-small-patch16 对六张被误分类的测试图出图（Fig. 10）：注意力分别落在 Greater Swiss Mountain Dog、harp、screwdriver、Swiss Mountain Dog、bell pepper、cello 上，而真值标签依次为 hatchet、hoopskirt、hammer、hairdryer、broccoli、harp——注意区域与**错误预测**一致，说明误判源于模型确实看了别的对象。

进一步，把用于分解的分数从错误类 $c_3$ 的 $Y^{c_3}$ 手动换成真值类 $c_5$ 的 $Y^{c_5}$（ImageNet2012 共 1000 类，分数向量记为 $[Y^0, Y^1, \dots, Y^{999}]$），再出图（Fig. 11）：注意力随即从"狗"移动到"hatchet"等真值对象上。这一对照表明 DAAM 能定位"给定标签下模型应当看的区域"，可作为误判分析工具。

## 十二、结论、局限与 vault 内关联

**结论。** 原文提出 DAAM，利用承担决策角色的 [class] token，在自注意力矩阵与值矩阵上提取并保存空间语义特征，用分类分数分解（监督）或维度重要性权重（自监督）计算重要性系数，再经逐块累积揭示动态注意力流；大量定量与定性实验验证了其优越的定位能力与解释工具有效性，且同时适用于以 FC 层为分类器的 ViT 与自监督 ViT。

**原文声明的局限。**

- 方法当前**依赖 [class] token**：只能解释存在 [class] token 的 block，无法解释与可视化 [class] token 尚不可用的中间块内部注意力流（如 CaiT 的 LayerScale 块、XCiT 的 XCA 块）；原文将"利用整个注意力矩阵来保存空间信息"列为未来工作，以拓宽适用架构范围。
- 原文计划放松对 [class] token 的依赖以增强对多样场景的适应性，并计划与 iterative integration 类技术结合，把方法推广到 CNN 基模型，成为 model-agnostic 的解释工具。
- 原文展望的方向（非已验证结论）：用 DAAM 指导新分类模型设计、推进弱监督技术、服务模型优化与对抗攻击防御，以及面向医学图像分析、自动驾驶等实际应用。

**与 vault 内笔记的关系。** 本文的 DAAM 属于判别模型侧的"注意力内部拆解"路线：它不依赖梯度回传，而是改写自注意力内部的乘法以抢救空间信息，这与生成模型侧聚合 cross-attention 的 [[DAAM-扩散模型cross-attention词像素归因|扩散模型 DAAM]] 形成鲜明对照——后者把多尺度 cross-attention 上采样求和得到词级归因图，本文把逐块自注意力分解图前缀求和得到决策演化流，二者"聚合对象"与"求和方向"完全不同，阅读时切勿因缩写相同而混用。在出图口径上，本文累积后先做全局 max-min 归一化再双线性上采样，与 [[注意力热力图产出规范#三、绘图约定|绘图约定]] 中"归一化口径必须显式声明"的要求同源：累积图的亮度可比性完全取决于归一化发生在累积之后还是之前。本文也是 [[2026-11-19-注意力可视化新文献复现规划|注意力可视化复现规划]] 中登记待复现的一行（判据为"逐块累积图展示决策区域形成过程，监督与自监督 ViT 均适用"），复现时需注意原文定量分析只统计预测正确样本、IoU 估计框阈值取 0.1 这两处口径。此外，DAAM 提供的"模型在看哪里"的逐块证据，与 [[劫持攻击]] 笔记中借助注意力占比分析攻击面的思路可互相印证：前者给出决策区域如何形成，后者关心这些区域能否被外部注入所劫持。
