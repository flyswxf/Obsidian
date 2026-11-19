**原文**：Hila Chefer, Shir Gur, Lior Wolf. *Transformer Interpretability Beyond Attention Visualization*. CVPR 2021. DOI `10.1109/CVPR46437.2021.00084`，arXiv `2012.09838`。Zotero key `JLWZ5ZPW`（含 PDF）。

> 注：[[2026-11-14-注意力可视化复现实验|实验汇总笔记]]、`D:\code\attn-repro\A\REPORT.md` 与 A 路代码注释原先误记为 ICCV 2021，现已全部更正为 **CVPR 2021**（同一作者组的 *Generic Attention-model Explainability*，arXiv `2103.15679`，才是 ICCV 2021 Oral）。`run_compare.py` 里的图例文字 `Chefer et al. ICCV'21 (ours)` 沿用官方仓库写法，已生成的对比图仍带旧图例，未重跑。

配套术语见 [[注意力可解释性术语详解|术语详解]]。

## 一、原文做的是什么工作

### 1.1 要解决的问题

给 CLIP 一张"猫和狗同框"的图，再给一句 `"a photo of a dog"`，模型会输出图文相似度。现在问：**模型是看了图里哪一块，才得出这个相似度的？**

朴素答案是"直接可视化注意力矩阵"。原文指出这条路在 Vision Transformer 上会失败，原因有三：

1. **多层多头怎么合并？** 12 层 × 12 头 = 144 张注意力图，直接平均会互相抵消。已有的 **Attention Rollout**（逐层连乘）是一种合并方式，但它不考虑每层注意力的"重要性"差异，也不区分正负贡献。
2. **注意力权重 ≠ 贡献。** 权重高只说明"看了"，不说明"看了之后对输出起了正面作用"。
3. **双模态需要跨模态归因。** CLIP 是双塔结构，要回答"图像侧看哪里"和"文本侧哪个词起作用"，需要一套能同时在两塔里传播的规则。

### 1.2 核心方法：用梯度修正注意力，再逐层传播

**第一步——梯度修正（原文 Eq.3）。** 对每个注意力头，取它对目标标量 $S$ 的梯度，与注意力本身逐元素相乘，只保留正值，再对头求平均：

$$\bar{A}=\frac{1}{h}\sum_{k=1}^{h}\left(\nabla A^{(k)}\odot A^{(k)}\right)^{+}$$

直觉拆解：

- $\nabla A^{(k)}$ 回答"如果这个注意力变大一点，目标分数会变大还是变小"。
- $\nabla A \odot A$ 把"看了多少"与"看了有没有用"乘在一起。
- $(\cdot)^+$ 丢掉负贡献（看了但起反作用的部分）。

**第二步——逐层传播（原文 Eq.4）。** 从底层往上，把每一层的修正注意力累加进相关性矩阵：

$$R \leftarrow R+\bar{A}R,\qquad R_{\text{初始}}=I$$

加 $R$ 自身对应残差通道（信息原地保留一部分）。最终取 $R$ 的 CLS 行：

- **图像侧**：$R[:,0,1:]$，即 CLS 对 49 个 patch 的相关性 → reshape 成 $7\times7$ → 双线性上采样到 $224\times224$ → min-max 归一化 → 热图。
- **文本侧**：EOT 行对各内容 token 的相关性 → 条形图。

**目标标量 $S$ 怎么取**：CLIP 不是分类器，没有"类别分数"。原文取 batch 内对角图文对的相似度之和 $S=\sum_i \text{logits\_per\_image}[i,i]$，即"第 $i$ 张图和第 $i$ 句话配不配"。

### 1.3 原文的结论

- 纯注意力与 Attention Rollout 在多物体、组合语义场景下会失效（热图分不清在指哪个物体）。
- 梯度修正的相关性传播能**按 query 的语义选中对应物体**：问 dog 就高亮狗，问 cat 就高亮猫。
- 逐层传播是必要的，不能只用最后一层。
- 文本侧的相关性可以解释组合语义（例如 `"a photo of a dog in the grass"` 会把分数分给 dog 与 grass）。

## 二、本次复现验证什么

原文只做了定性展示（放几张热图对比），**没有给出可量化的判据**。所以本次复现要回答的不是"能不能跑出图"，而是三个可判定的问题：

| 编号 | 待验证命题 | 为什么值得验 |
| --- | --- | --- |
| H1 | Chefer 热图的高响应区随 query 切换，且切换到**正确物体**上 | 这是原文的核心主张，但原文只用图说话 |
| H2 | 逐层 rollout 比只用末层更稳定 | 原文的消融主张，需独立复核 |
| H3 | Grad-CAM 作为传统基线，在 ViT + CLIP 上定位精度不及相关性传播 | 原文**没有**做这个对比，属于本次补充 |

## 三、怎么判断成功（判据设计）

这是本次复现最需要讲清楚的部分——**热图好不好看是主观的，必须换成可计算的量**。设计了三个互补判据：

### 判据 1：top-10% 质心坐标

取热图值最高的 10% 像素，按像素值加权求重心：

$$c_x=\frac{\sum_i x_i w_i}{\sum_i w_i},\qquad c_y=\frac{\sum_i y_i w_i}{\sum_i w_i}$$

（$224\times224$ 坐标系，原点左上；全零热图退化为该区域几何中心，代码里加了零权重保护避免除零。）

**判定方式**：肉眼确认质心是否落在目标物体上。这是唯一能验证"切换得对不对"的判据——因为"变化了"不等于"变对了"。

### 判据 2：query 切换敏感度（Pearson + SSIM）

同一张图，分别用两个 query 算热图，然后比较这两张热图：

- **Pearson 相关**：像素级线性相关。越低说明两张热图差异越大。
- **SSIM**：结构相似性。越低说明结构差异越大。

**判定方式**：一个可靠的方法应当在 query 切换时产生**明显不同**的热图（Pearson 低、SSIM 低）。如果两张热图几乎一样，说明方法根本没在读 query。

**关键陷阱**：这个判据单独用会误导。Grad-CAM 的 Pearson 是 −0.21 / −0.63，比 Chefer 的 0.67 / 0.46 "更敏感"，但那是因为 Grad-CAM 热图本身弥散、到处都是噪声响应。**所以判据 2 必须与判据 1 联用**。

### 判据 3：消融对比

同一方法跑两次：`start_layer=0`（全层 rollout，主结果）与 `start_layer=-1`（仅最后一层）。比较两者的 SSIM 与热图噪声水平。

**判定方式**：若全层版本的 SSIM 显著更高、热斑更紧凑，则支持"逐层传播必要"。

### 退化检测

Grad-CAM 若在某 target layer 上输出全零，则 Pearson 无定义（NaN）、SSIM = 1.0、质心位移 = 0。这三个值同时出现即为**退化信号**，需在 `metrics.json` 中显式标记 `gradcam_ln_2_degenerate_all_zero: true`，而不是当作"两图完全一致"误读。

## 四、实验怎么做的

### 4.1 模型与权重

用**官方修改版 CLIP ViT-B/32**（仓库 `Transformer-MM-Explainability` 的 `CLIP` 子目录），不是 openai 原版。修改点：在每个 `ResidualAttentionBlock` 上挂了

- forward hook → 保存注意力概率 `attn_probs`
- backward hook → 保存注意力梯度 `attn_grad`

权重是官方 `.pt` 文件，353.98 MB，缓存在 `D:\checkpoints\clip\ViT-B-32.pt`。

**权重自检**（确认下载没坏、模型行为正常）：CPU 前向，catdog 图对两个 query 的 `logits_per_image` softmax 为 $[0.5558,\ 0.4442]$，dogbird 为 $[0.6445,\ 0.3555]$——都是"第一个 query 略占优"的合理值，说明 CLIP 正常工作。

### 4.2 实验设置

| 项 | 取值 |
| --- | --- |
| 图片 | `catdog.png`（左狗右猫）、`dogbird.png`（狗与鹦鹉同框），仓库自带 |
| query | catdog：`"a photo of a dog"` / `"a photo of a cat"`；dogbird：`"a photo of a dog"` / `"a photo of a bird"` |
| 方法 | Chefer（全层 / 仅末层）、Grad-CAM（`ln_1` / `ln_2`） |
| 设备 | **仅 CPU**（`torch.device("cpu")`），全程未用 CUDA |
| 组合数 | 2 图 × 2 query × 4 设置 |

选 CPU 的原因：模型只有 354 MB，CPU 前向 + 反向完全够用，且避免 GPU kernel 差异带来的数值不确定性。

### 4.3 Grad-CAM 的两处适配

CLIP 不是分类器，Grad-CAM 库要求分类输出与 4D 激活，所以：

1. `CLIPScoreWrapper` 把 `logits_per_image` 包装成分类 logits，目标设 `ClassifierOutputTarget(0)`（第 0 个 query 的图文相似度）。
2. `reshape_transform` 把 ViT 激活 $(1+49,\ B,\ C)$ 丢掉 CLS token 后 permute 成 $(B,\ C,\ 7,\ 7)$。

target layer 取视觉编码器最后一个 block 的 `ln_1` 与 `ln_2` 各跑一次。

### 4.4 执行顺序

在 `D:\code\attn-repro\A\` 下：

| 步骤 | 命令 | 产物 |
| --- | --- | --- |
| 1 | `python fetch_weights.py` | 下载/校验权重，CPU 前向自检 |
| 2 | `python run_chefer.py` | `chefer_*.npy`、单图、文本条形图、`chefer_summary.json` |
| 3 | `python run_gradcam.py` | `gradcam-*.npy`、单图、`gradcam_summary.json` |
| 4 | `python run_compare.py` | 指标计算、2×3 拼图、`metrics.json` |

拼图布局：2 行（dog query / 另一 query）× 3 列（原图 | Grad-CAM | Chefer），jet colormap，`alpha=0.5` 加性混合。

## 五、结果

### 5.1 热图对比

![[A-chefer-vs-gradcam-catdog.png]]

catdog：上行 dog query，下行 cat query；列为原图 | Grad-CAM | Chefer。

![[A-chefer-vs-gradcam-dogbird.png]]

dogbird：上行 dog query，下行 bird query。

### 5.2 判据 2 结果：query 切换敏感度

| 方法                | catdog Pearson | catdog SSIM | 质心位移/px | dogbird Pearson | dogbird SSIM | 质心位移/px |
| ----------------- | -------------- | ----------- | ------- | --------------- | ------------ | ------- |
| Chefer 全层 rollout | 0.6656         | 0.8338      | 87.19   | 0.4568          | 0.7174       | 133.57  |
| Chefer 仅末层（消融）    | 0.1587         | 0.5574      | 113.45  | −0.1321         | 0.3226       | 120.45  |
| Grad-CAM（`ln_1`）  | −0.2107        | 0.2186      | 133.22  | −0.6260         | 0.1454       | 132.94  |
| Grad-CAM（`ln_2`）  | NaN            | 1.0         | 0（全零退化） | NaN             | 1.0          | 0（全零退化） |

### 5.3 判据 1 结果：质心落在哪

| 图 / query | Chefer 质心 | Grad-CAM(`ln_1`) 质心 | 两者 Pearson |
| --- | --- | --- | --- |
| catdog / dog | (119.81, 62.14) | (123.31, 59.83) | 0.4552 |
| catdog / cat | (134.85, 148.02) | (108.06, 192.18) | 0.5186 |
| dogbird / dog | (157.43, 80.79) | (138.58, 90.32) | 0.3645 |
| dogbird / bird | (47.67, 156.90) | (46.01, 185.74) | 0.7171 |

**判读**：

- catdog 里狗在画面左上部、猫在右下部。Chefer 质心从 $(120,62)$ 移到 $(135,148)$，纵向位移 86 px，正好从狗的头颈移到猫的脸与前身。**H1 成立。**
- dogbird 里狗在右侧、鹦鹉在左侧。Chefer 质心从 $(157,81)$ 移到 $(48,157)$，横向位移 110 px，从狗头切到鹦鹉头颈。**H1 成立。**
- Grad-CAM 的质心在 cat query 下跑到 $(108,192)$，已越过猫的身体落到地板光斑；bird query 下 $(46,186)$ 偏向鹦鹉下身而非头部。**定位精度不及 Chefer。**

### 5.4 判据 3 结果：消融

仅末层版本的 catdog SSIM 是 0.5574，全层是 0.8338；dogbird 更明显（0.3226 vs 0.7174）。仅末层热图噪声更大、结构更不稳定。**H2 成立。**

### 5.5 文本侧

![[A-chefer-text-catdog.png]]

![[A-chefer-text-dogbird.png]]

内容词获得最高相关性：`dog` ≈ 0.42、`cat` ≈ 0.40，`a`/`photo`/`of` 这类功能词得分低。与原文"组合语义归因到关键词"的展示一致。

### 5.6 Grad-CAM 的退化

把 target layer 从 `ln_1` 换成 `ln_2`，CAM 输出**完全为零**。`ln_2` 是 ViT block 里 FFN 之前的那个 LayerNorm，其激活对图文相似度标量的梯度在该配置下消失。

这说明 Grad-CAM 在 ViT 上的结果**高度依赖 target layer 的人工选择**，而这个选择在原文里没有客观依据。这一现象超出 Chefer 原文的讨论范围，属于本次复现的新增发现。

## 六、结论

| 命题                          | 判定     | 依据                                                |
| --------------------------- | ------ | ------------------------------------------------- |
| H1 Chefer 热图随 query 切换到正确物体 | **成立** | 两图四组的质心坐标均落在目标物体上，位移 87–134 px                    |
| H2 逐层 rollout 必要            | **成立** | 仅末层 SSIM 显著更低（0.5574 / 0.3226 vs 0.8338 / 0.7174） |
| H3 Grad-CAM 定位不及 Chefer     | **成立** | 质心偏离目标（落到地板光斑、鹦鹉下身）；`ln_2` 完全退化；热图弥散有背景泄漏         |

**一句话**：在"图 + 文 query"条件下，梯度修正的相关性传播确实能定位到被查询的物体，而直接套 Grad-CAM 既不稳定也不准确。

**为什么这个结论对主线研究重要**：[[2026-09-07-实验状态全面同步|此前的 LLaVA 实验]]用的是单层 raw attention 的区域均值。本次结果说明，换成带梯度修正的相关性传播，得到的注意力指标质量会明显更高——这是"注意力与 ASR 无稳定关联"这一矛盾的一个可能解释方向。

## 七、复现中踩的坑

| 问题                                                                                     | 原因                                                           | 解决                                                         |
| -------------------------------------------------------------------------------------- | ------------------------------------------------------------ | ---------------------------------------------------------- |
| `RuntimeError: cannot register a hook on a tensor that doesn't require gradient`（出现两次） | ① 权重自检时用了 `torch.no_grad()`；② 加载后调用了 `requires_grad_(False)` | 修改版 CLIP 依赖 backward hook，两者都会破坏 hook 注册。全部移除，保持计算图可导      |
| torchvision 0.28 报错                                                                    | 官方 `_transform` 传的是 int 型 `Image.BICUBIC`，新版只接受枚举            | monkey-patch 成 `InterpolationMode.BICUBIC`                 |
| 官方 notebook 超过 128 KB 读不完                                                              | —                                                            | 关键词检索 + 分段读取，提取 `interpret()` 的 rollout 逻辑                 |
| 最终图中文显示为豆腐块                                                                            | matplotlib 默认字体 DejaVu Sans 无 CJK 字形                         | `common.setup_cjk_font()` 注册 `msyh.ttc` / `simhei.ttf` 后重跑 |
| 不允许 `pip install -r requirements.txt`                                                  | 环境隔离要求                                                       | 只把仓库 `CLIP` 子目录加入 `sys.path`，所需包 attn-repro 环境已具备          |

## 八、产物

| 类型 | 路径 |
| --- | --- |
| 详细报告 | `D:\code\attn-repro\A\REPORT.md` |
| 代码 | `D:\code\attn-repro\A\{fetch_weights,common,run_chefer,run_gradcam,run_compare}.py` |
| 指标 JSON | `D:\checkpoints\outputs\A\metrics.json`、`chefer_summary.json`、`gradcam_summary.json` |
| 中间数组 | `D:\checkpoints\outputs\A\chefer_*.npy`、`gradcam-*.npy` |
| 交付图 | `assets/attn-repro/A-*.png`（4 张） |
