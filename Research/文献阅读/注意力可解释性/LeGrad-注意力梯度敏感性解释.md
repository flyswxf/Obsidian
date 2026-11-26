> [!abstract] 一句话总结
> LeGrad 把"目标类别（或文本提示）激活值对每一层注意力图的梯度"本身当作解释信号：逐层求梯度、用 ReLU 丢弃负值、在注意力头与行上平均得到该层的 token 级热图，再对所有层取平均并做 min-max 归一化；由于逐层信号口径统一为梯度，跨层相加无需额外归一化，因而能直接扩展到 ViT-BigG/14 这类 2.5B 参数的大模型与 SigLIP 的注意力池化结构。

## 问题背景

- Vision Transformer（ViT）已是视觉标准架构，但为卷积网络或前馈网络设计的解释方法套不上去：Grad-CAM 依赖卷积层的空间特征图，逐层相关性传播（LRP）依赖为每种模块定制的传播规则；而 ViT 没有空间特征图，patch 之间由自注意力复杂耦合。
- 已有 ViT 专用手段仍不够：Chefer 等把 LRP 推广到 transformer，计算重且换架构就要重写每个模块的规则；CheferCAM 用梯度给注意力头加权，使解释与类别绑定；rollout 与 attention flow 只把各层注意力图线性串联，忽略头与后续层的非线性交互，也不区分正负贡献，可能给出误导解释。
- 中心问题：能否只用一种信号——注意力图对预测的敏感性（梯度）——构造一个概念简单、跨层可加、跨聚合策略可用的 ViT 解释方法？原文（Bousselham, Boggust, Chaybouti, Strobelt, Kuehne, ICCV 2025, arXiv:2404.03214）的回答即 LeGrad。

## 一、动机与预备概念：ViT 里有什么可解释

先给三个通俗概念，再给术语。**patch token**：ViT 把图像切成 $n$ 个小块，每块线性嵌入成一个 $d$ 维向量，加位置编码后排成序列，即"把图变成一句话"。**class token（[CLS]）**：序列开头额外插入的一个可学习向量，训练中被要求汇总全图信息，最后由它代表整张图做分类。**注意力图（attention map）**：某一层自注意力中"谁在看谁"的分数矩阵，是信息在 patch 间流动的通道，因此是解释的天然切入点。

原文的形式化（式 1）：输入序列 $Z^0 = \{z_0^0, z_1^0, \dots, z_n^0\} \in \mathbb{R}^{(n+1)\times d}$，其中 $z_0^0 = z_{[CLS]}^0 \in \mathbb{R}^d$；每层先多头自注意力（MSA）后 MLP，均带残差连接：

$$\hat{Z}^l = \mathrm{MSA}^l(Z^{l-1}) + Z^{l-1}, \qquad Z^l = \mathrm{MLP}^l(\hat{Z}^l) + \hat{Z}^l \tag{1}$$

$L$ 层之后要得到"整图向量"，原文对比两种特征聚合方式。其一是 class token 法：$\bar{z}_{[CLS]} = z_0^L$，依赖训练把 patch 信息汇入 [CLS]。其二是注意力池化（attentional pooler，SigLIP 采用）：用一个可学习查询 $q_{\mathrm{pool}} \in \mathbb{R}^d$ 与末层 patch token 做多头注意力：

$$\bar{z}_{\mathrm{AttnPool}} = \mathrm{softmax}\!\left(\frac{q_{\mathrm{pool}} \cdot (W_K Z^L)^{\top}}{\sqrt{d}}\right)(W_V Z^L) \tag{2}$$

其中 $W_K, W_V \in \mathbb{R}^{d\times d}$ 为可学习投影矩阵。

**为什么需要这一步：** 解释方法必须先回答"以哪个向量为预测来源、解释覆盖哪些层"。原文强调 ViT 的特征形成是逐层迭代的，只盯末层输出的解释会漏掉中间层贡献；先固定聚合方式与层间记号，后面的逐层信号才有定义的地基。LeGrad 的目标正是融合每一层的信息，既支持逐层细看、也支持合并总览。

## 二、核心思想：单层注意力图的梯度就是解释信号

**梯度敏感性**的通俗说法：稍微改动注意力图里的某个元素，看目标分数变化多快；变化快说明这条注意力连接对当前预测影响大。LeGrad 的关键选择是：不拿梯度去加权注意力图（CheferCAM 的做法），而是把梯度本身当作热图原料。

记第 $l$ 层输出 token 为 $Z^l = \{z_0^l, z_1^l, \dots, z_n^l\} \in \mathbb{R}^{(n+1)\times d}$，层内 token 平均为 $\bar{z}^l = \frac{1}{n+1}\sum_{i=0}^{n} z_i^l$。再设映射模型 $C \in \mathbb{R}^{d \times C}$，把 token 维度映射到 $C$ 个 logit：监督分类时 $C$ 是训练好的分类头、$C$ 为类别数；零样本场景下 $C$ 由若干提示词的文本嵌入构成、$C$ 为提示词数。把层内平均特征送入 $C$ 得到该层预测：

$$\bar{y}^l = \bar{z}^l \cdot C \in \mathbb{R}^C \tag{3}$$

对视觉语言模型，$\bar{y}^l$ 即文本嵌入与图像嵌入点积的结果向量。取目标类别/提示词 $\hat{c}$ 对应的激活 $s^l = \bar{y}^l_{[\hat{c}]}$ 作为求导起点。

**为什么需要这一步：** 多数解释方法只对末层输出求一次梯度，信号只有一条；把每一层都变成可求导的"子预测"（用该层自己的平均 token，即中间 token 激活），才能拿到逐层信号并在后面自由聚合。原文特别指出，大多数解释方法只用模型最终输出，而利用中间表示是有益的（见其 4.7 节消融）。

单层流程（原文 3.2 节与图 3）：对第 $l$ 层注意力图 $A^l \in \mathbb{R}^{h \times n \times n}$（$h$ 为头数）求激活的梯度：

$$\nabla A^l = \frac{\partial s}{\partial A^l} \in \mathbb{R}^{h \times (n+1) \times (n+1)} \tag{4}$$

（原文记 $A^l \in \mathbb{R}^{h\times n\times n}$，而式 (4) 的梯度维度为 $(n+1)\times(n+1)$，多出的行列对应 [CLS]。）用 ReLU（记 $(\cdot)^+$）丢弃负梯度，防止负梯度影响正激活，再对头与行求平均：

$$\hat{E}^l(s) = \frac{1}{h \cdot (n+1)} \sum_{h} \sum_{i} \left(\nabla A^l_{h,i,:}\right)^+ \in \mathbb{R}^{n+1} \tag{5}$$

符号逐项解释：

- $h$：自注意力头数；$n$：patch 数；$i$：注意力图的行（query token）索引，$\nabla A^l_{h,i,:}$ 是第 $h$ 头、第 $i$ 行对所有 key token 的梯度向量。
- $(\cdot)^+$：ReLU，负梯度置零。
- $\frac{1}{h\,(n+1)}\sum_h \sum_i$：跨头、跨行平均，把三维梯度压成长 $n+1$ 的 token 级向量。
- $\hat{E}^l(s)$：第 $l$ 层每个 token 的解释强度。

去掉 [CLS] 列（下标 $1:$）、reshape 成二维并做 min-max 归一化（把数值线性拉伸到 $[0,1]$），得单层热图：

$$E^l(s) = \mathrm{norm}(\mathrm{reshape}(\hat{E}^l(s)_{1:})) \in \mathbb{R}^{W \times H} \tag{6}$$

直观解释（不改变数学含义）：式 (5) 相当于逐个 patch 提问"你被关注的程度若微调一点，目标分数会变多少"，把答案按头与行平均后摊回图像网格；负变化视为无贡献。

## 三、跨层聚合：中间 token 激活 + 逐层平均

**为什么需要这一步：** 单层热图只反映某一阶段的敏感性，而原文消融显示定位信号分散在多层；若每层用不同口径的信号（如激活值），各层量纲不同、直接混用会互相淹没。LeGrad 的解法是让每层用本层激活 $s^l$ 独立产图（口径统一为梯度），再简单平均——原文在引言中指出，逐层梯度信号相比其他信号的优点正是跨层相加无需额外归一化。

对每层用中间 token $Z^l$ 计算激活 $s^l$ 并产图：

$$\hat{E}^l(s^l) = \frac{1}{h \cdot (n+1)} \sum_{h} \sum_{i} \left(\nabla A^l_{h,i,:}\right)^+ \in \mathbb{R}^{n+1} \tag{7}$$

对所有层取平均（同样去掉 [CLS] 列）：

$$\bar{E} = \frac{1}{L} \sum_{l} \hat{E}^l(s^l)_{1:} \tag{8}$$

最后 reshape 回原图尺寸并 min-max 归一化：

$$E = \mathrm{norm}(\mathrm{reshape}(\bar{E})) \in \mathbb{R}^{W \times H} \tag{9}$$

符号逐项解释：$L$ 为层数；$\hat{E}^l(s^l)_{1:}$ 为第 $l$ 层去掉 [CLS] 后的 patch 级解释向量；$\mathrm{norm}$ 为 min-max 归一化；$W, H$ 为原图宽高。直观解释：把"每一层各自认为重要的 patch"投票平均，再拉伸到 $[0,1]$ 画成一张热图。原文称这一步"询问每个 patch token 在该阶段对预测的影响"。

**注意力池化的适配。** 对 SigLIP 这类用注意力池化的 ViT，把池化模块 $\mathrm{Attn}_{\mathrm{pool}}$ 作用于每层中间表示 $Z^l$ 得到池化查询 $q^l \in \mathbb{R}^d$，激活改为 $s^l = q^l \cdot C_{:,c} \in \mathbb{R}$；求导对象也从自注意力图换成池化注意力图 $A_{\mathrm{pool}} \in \mathbb{R}^{h \times 1 \times n}$，即对每层 $\nabla A^l = \partial s^l / \partial A_{\mathrm{pool}}^l$。同一套公式由此覆盖 class token 与注意力池化两种聚合策略。

## 四、与 Grad-CAM、rollout、Chefer 等方法的关系

- 与 CheferCAM / AttentionCAM：它们用梯度**加权**注意力图（梯度乘注意力再聚合），LeGrad 只用梯度本身。原文附录把 GradCAM 适配到 ViT（对 token 维度求梯度作权重），并实验发现给 GradCAM 套上 LeGrad 的逐层聚合（式 8）只带来微弱提升（OpenImagesV7 p-mIoU 由 8.72 到 9.51，表 7），归因于 ViT 各层激活值量纲不同；LeGrad 只用梯度，天然避开该问题。
- 与 rollout / raw attention：rollout 用矩阵乘法把各层注意力图串成重要性流，忽略头与后续层的非线性交互、也不区分正负贡献；raw attention 只取末层 [CLS] 行。原文定性比较指出 raw attention 常把高强度打在少数背景像素上（与"个别 token 不成比例地吸收注意力权重"的已知现象一致），依赖它的 CheferCAM 带同类伪影；而 LeGrad 中那些对不同提示词都均匀响应的噪声区域梯度接近零，会被自动滤掉。
- 与 Chefer 的 LRP 推广：LRP 路线要为每个模块写传播规则，计算重、换架构不灵活；LeGrad 只需要"能对注意力图求梯度"这一个接口。
- 与 TextSpan：TextSpan 不用梯度，靠中间特征沿文本方向的相似度聚合产图；LeGrad 属梯度路线，但信号定义更简单。
- 速度（表 3，ViT-B/16，1,000 张图平均，fps）：GradCAM 108、AttentionCAM 103、LeGrad 96、CheferCAM 21、LRP 4.0、TextSpan 3.8。原文解释：LeGrad 虽用多层，但靠逐层求和而非 CheferCAM 的矩阵乘法，速度只比单层方法略降。

（解读注：四条路线可记成一条光谱——"注意力本身（raw attention / rollout）→ 梯度×注意力（GradCAM / AttentionCAM / CheferCAM）→ 梯度本身（LeGrad）→ 不用梯度（TextSpan）"，LeGrad 站在信号最单一、跨层口径最统一的一端。）

## 五、实验设置与结果

### 5.1 物体分割（空间保真度）

**设置。** 任务：以真值类别激活（分类模型）或提示词 "A photo of a [class]"（视觉语言模型）产热图，阈值 0.5 二值化做前景/背景分割。数据集：ImageNet-Segmentation，4,276 张带分割标注的图。指标：mIoU（预测掩码与真值掩码的重叠度）、像素准确率、mAP。 backbone 为 OpenCLIP ViT-B/16（Laion2B 权重）。

**结果（表 1，数值引自原文）。**

| 方法 | Pixel Acc.↑ | mIoU↑ | mAP↑ |
| --- | --- | --- | --- |
| LRP | 52.81 | 33.57 | 54.37 |
| Partial-LRP | 61.49 | 40.71 | 72.29 |
| rollout | 60.63 | 40.64 | 74.47 |
| Raw attention | 65.67 | 43.83 | 76.05 |
| GradCAM | 70.27 | 44.50 | 70.30 |
| CheferCAM | 69.21 | 47.47 | 78.29 |
| TextSpan | 73.01 | 40.26 | 81.4 |
| LeGrad | 77.52 | 58.66 | 82.49 |

正文口径为 mIoU 58.7%，同时超过梯度路线的 CheferCAM 与非梯度路线的 TextSpan，三项指标均为全场最优。

### 5.2 开放词汇定位

**为什么需要这一步：** ImageNet 类别有限，分割实验无法说明方法在任意文本描述下是否仍定位准确；开放词汇设置把评测推到训练类别之外。

**设置。** 任务：对图中实际出现的每个类别产热图、0.5 二值化、评定位精度。数据集/指标：OpenImagesV7 验证集的点标注，36,702 张图、5,827 个唯一类别标签，每图含正负点标注；指标为点标注口径的 mIoU（p-mIoU，原文图 4 注记为 point-mIoU）。对 GradCAM 做了逐层搜索并取表现最好的层。

**结果（表 2，p-mIoU↑，OpenCLIP 三种尺寸，数值引自原文）。**

| 方法 | B/16 | L/14 | H/14 |
| --- | --- | --- | --- |
| rollout | 8.75 | 6.85 | 5.82 |
| Raw attention | 0.94 | 1.60 | 0.85 |
| GradCAM | 8.72 | 2.80 | 2.46 |
| AttentionCAM | 5.87 | 4.74 | 1.20 |
| CheferCAM | 5.87 | 2.51 | 9.49 |
| TextSpan | 9.44 | 21.73 | 23.74 |
| LeGrad | 48.38 | 47.69 | 46.51 |

LeGrad 相对第二名有 2×–5× 的提升，原文视其为细粒度识别能力的证据；摘要口径为 OpenCLIP-ViT-B/16 上 48.4 p-mIoU。注意其他方法在类别多样性面前普遍崩溃（raw attention 甚至低于 2），而 LeGrad 在三种尺寸上稳定在 46–48。

### 5.3 音频定位（跨模态通用性）

**为什么需要这一步：** 若方法只适用于图文模型，通用性存疑；ImageBind 的图像编码器是 vanilla ViT，且对齐了音频模态，是检验"换模态提示是否仍可用"的天然测试台。

**设置。** 模型：ImageBind。任务：给定音频提示，产热图定位图中对应区域。数据集：ADE20KSoundPrompted 与 ADE20KSpeechPrompted。指标：mAP 与 mIoU。

**结果（表 5，数值引自原文）。**

| 方法 | Speech mAP↑ | Speech mIoU↑ | Sound mAP↑ | Sound mIoU↑ |
| --- | --- | --- | --- | --- |
| DAVENet | 32.2 | 26.3 | 16.8 | 17.0 |
| CAVMAE | 27.2 | 19.9 | 26.0 | 20.5 |
| ImageBind | 20.2 | 19.7 | 18.3 | 18.1 |
| ImageBind + LeGrad | 23.3 | 21.8 | 48.0 | 38.9 |
| DenseAV* | 48.7 | 36.8 | 32.7 | 24.2 |

声音定位上 ImageBind + LeGrad 大幅领先并刷新该基准 SoTA（摘要口径：较此前 SoTA +14 mIoU，超过 DAVENet、CAVMAE 与 DenseAV）。语音定位提升不明显，原文归因于 ImageBind 训练时只用了声音、未用语音数据，底座在语音上本身表现差，LeGrad 无法进一步补救。

### 5.4 扰动评测（忠实性）

**为什么需要这一步：** 分割与定位只说明热图"像不像物体"，扰动评测直接检验热图是否忠实标出了模型真正依赖的区域。

**设置。** 任务：正、负两种扰动——按解释图相关性从高到低（正）或从低到高（负）逐步遮挡图像区域，观察模型精度变化。指标：像素擦除比例 0%–90% 曲线下面积（AUC），分别对预测类别与真值类别报告；类别特定的方法在真值口径下预期更好。数据集：ImageNet-val，50K 张图、1,000 类。

**结果（表 6 节选，AUC；Negative 越高越好、Positive 越低越好，数值引自原文）。**

| 模型 | 方法 | Neg Pred.↑ | Neg Targ.↑ | Pos Pred.↓ | Pos Targ.↓ |
| --- | --- | --- | --- | --- | --- |
| ViT-B/16 | TextSpan | 50.92 | 52.81 | 15.10 | 14.26 |
| ViT-B/16 | LeGrad | 50.24 | 52.27 | 15.06 | 13.97 |
| ViT-H/14 | TextSpan | 60.14 | 61.91 | 20.07 | 19.14 |
| ViT-H/14 | LeGrad | 60.02 | 61.72 | 19.30 | 18.26 |
| ViT-BigG/14 | TextSpan | 63.32 | 64.95 | 21.96 | 21.06 |
| ViT-BigG/14 | LeGrad | 62.62 | 64.67 | 21.33 | 20.28 |

原文表述：LeGrad 在正扰动上与 TextSpan 相当、负扰动略优；对其余方法（rollout、raw attention、GradCAM、AttentionCAM、CheferCAM）在各模型尺寸与预测/真值两种类别口径上全面占优，说明其能识别并保留对分类关键的图像区域。（解读注：按表 6 数值，ViT-B/16 上 LeGrad 对 TextSpan 的优势体现在正扰动 AUC 更低，负扰动两者相差不到 0.7 点，"相当"更贴合数值。）附录表 8 另给出在 ImageNet 上监督训练的 ViT-B/16 的同类比较：LeGrad 负扰动真值口径 56.43、正扰动真值口径 14.13，均为最优。

### 5.5 SigLIP 适配与速度

表 4（SigLIP-B/16）：LeGrad 在 ImageNet-val 负扰动 Pred./Targ. 为 50.08 / 51.67、正扰动为 18.48 / 17.55，OpenImagesV7 p-mIoU 25.40，而次优 GradCAM 仅 6.97（正文口径 7.0），说明方法对注意力池化机制的鲁棒适配。附录 7.5 记录了一个工程细节：SigLIP 对所有解释方法都在背景区出现与输入无关的高激活，原文用空提示 "a photo of" 的热图做抑制（该图超过阈值 0.8 的位置置零）。速度方面见第四节表 3。

### 5.6 消融与分析

- 层数累积（图 4）：负扰动上 ViT-B/16 用较少层反而最优，ViT-L/14 与 ViT-H/14 则层数越多越好；开放词汇分割上所有模型都随层数增加而提升、超过一定层数后进入平台期。原文解释：模型越大，信息汇入 [CLS] 的过程越分散，需要更完整的逐层分析。
- 逐层热图定性（图 5）：提示词的定位信号不局限于单层；小模型信号集中在末层，大模型的低层也有贡献。
- 梯度分布（图 7）：在 PascalVOC 上对 OpenCLIP(Laion400M)、OpenAI CLIP、MetaCLIP 三种 ViT-L/14 权重统计逐层梯度强度（掩码内平均、跨层 min-max 归一化）：贡献最大的层多靠网络末端，MetaCLIP 在倒数第二层，Laion400M 的中间层影响更明显；原文称这种分布如同模型的"指纹"。附录进一步发现 'person' 类在各尺寸与权重下都对中间层更敏感。
- 组件消融（表 9，目标类别 AUC）：去掉 ReLU 或去掉中间 token（只用末层特征求梯度）都会降点。ViT-L/14：两者皆无 47.81 / 20.80，仅 ReLU 49.32 / 19.95，仅全层 52.01 / 16.80，完整配置 54.48 / 15.23；ViT-H/14 完整配置 61.72 / 18.26。
- sanity check（附录表 10，FunnyBirds Co-12 十二条性质）：LeGrad 在 CSDC 0.90、PC 0.83、DC 0.92、D 1.、SD 0.77、TS 0.97，优于梯度类方法、与 LRP 相当。

## 六、结论、局限与 vault 内关联

**结论。** 原文提出 LeGrad：以逐层注意力图梯度为解释信号、跨层平均聚合，兼容 class token 与注意力池化两种特征聚合策略；在物体分割、开放词汇定位、扰动评测与音频定位四类任务上验证了空间保真度与通用性，并能直接用于 2.5B 参数的 ViT-BigG/14。原文把方法定位为概念简单、易实现、易迁移的 ViT 透明化手段。

**原文声明的局限与边界。** 原文未设独立 Limitations 章节，以下边界均出自正文自述：

- 语音定位收益有限：ImageBind 未用语音数据训练，底座在语音上表现差时，LeGrad 无法进一步补救（4.3 节）。
- 层数配置与模型规模相关：ViT-B/16 在负扰动上用较少层反而最优，说明"全层平均"并非对所有规模都是最优配置；且层数超过一定数量后性能进入平台期（4.7 节）。
- SigLIP 的背景区存在与输入无关的高激活，需要空提示抑制这一额外工程步骤（附录 7.5）。
- 数据伦理声明：ImageNet 与 OpenImagesV7 含网络来源、可能未经本人同意的图像；所用视觉语言模型训练于可能含敏感内容的 LAION-2B（附录 11）。

**与 vault 内笔记的关系。** LeGrad 属"把注意力当作解释原料"、并对注意力图求梯度、用敏感性替代原始分数的路线，可避开 raw attention 的背景伪影。在 CLIP 可解释性谱系上，LeGrad 属事后归因，与改造图像编码器本身的 [[CLIP-AFT-对抗微调增强视觉可解释性|CLIP-AFT]] 互补；其提示词条件热图也可用于检查注入提示引发的响应区域，与 [[劫持攻击]] 中按 token 注意力占比分析攻击面的思路互证。LeGrad 已列入 [[注意力可视化|复现规划]] 的待复现清单，判据即"注意力梯度聚合图在分割重叠与扰动保真上优于 rollout"。
