---
状态: 第 1 轮已完成
日期: 2026-11-25
研究线: "[[注意力可视化|注意力可视化复现]]"
基础: "[[LeGrad-注意力梯度敏感性解释]]"
---

## 第 1 轮（2026-11-24 至 2026-11-25）：本地全量复现与口径核查

### 0 摘要

本轮在本地单机完成 [[2026-11-24-LeGrad在ImageNet-S上的复现与可视化|规划]] 的 S0–S3：环境就位、单图 dry-run、三联图可视化、ImageNet-S validation 全量 12,403 张指标复算 #实测 `D:\datasets\imagenets\results\eval_s224_l-2.jsonl`。主结果：224 输入下 mIoU 46.06／PixAcc 80.73／mAP 71.90，448 输入下 45.51／81.20／74.64 #实测 同上及 `eval_s448_l-2.jsonl`；原文表 1 为 58.66／77.52／82.49（n=4,276）#文献 Zotero key 8S8R9TW4。定性定位行为与原文一致（目标物体被正确点亮、prompt 可切换定位对象）#实测 `D:\Obsidian\Research\组会与实验\附件\图片\LeGrad-ImageNetS\`。对分辨率、二值化阈值、归一化方式、指标聚合、层聚合深度五项口径逐一探针后，均未支持"实现口径差异"的解释 #实测 `D:\datasets\imagenets\results\probe_sweeps.json`；差距的残余解释列入第 9 节未知项。

### 1 目标与范围

- 目标：在 ImageNet-Segmentation 上复现 LeGrad，复算原文表 1 的 mIoU／像素准确率／mAP，并产出"原图＋热图＋真值掩码"三联图 #文献 Zotero key 8S8R9TW4，§4.1。
- 范围：S0–S3 执行；S4 基线对照在规划中即标注为不做 #代码事实 `D:\Obsidian\Research\组会与实验\实验规划\2026-11-24-LeGrad在ImageNet-S上的复现与可视化.md` 变量矩阵节。
- 与规划的两处偏离（均为执行期决定，不回填规划）：
  1. 算力改本地单机（RTX 5060 Laptop，8 GB）而非 Slurm 集群。原因：该 GPU 计算能力为 (12,0)（sm_120），torch 2.7+cu128 起才含对应 kernel，集群工具链未验证 #实测 `probe_sweeps.json` → `environment`。
  2. 评测集取 ImageNet-S validation 公开掩码全量 12,403 张，而非规划中的 4,276 张。原因：4,276 的子集口径无法从公开元数据重建（见 3 与 5.4）#实测 `probe_sweeps.json` → `subset_probes`。

### 2 被复现方法与评测协议

LeGrad 对第 $l$ 层以文本-图像相似度 $s_l$ 对该层注意力图 $A_l$ 求梯度，去负、按头与查询维平均、去除 [CLS] 列后 reshape 并归一化，再跨层累加 #文献 Zotero key 8S8R9TW4，式 (5)–(8)；对应实现见 `d:\code\LeGrad\legrad\wrapper.py` 的 `compute_legrad_clip`（第 144–186 行）与 `legrad\utils.py` 的 `min_max`（第 498–503 行） #代码事实 

$$\hat{E}^{(l)}=\mathrm{ReLU}\!\Big(\frac{\partial s_l}{\partial A_l}\Big),\qquad E=\mathrm{norm}\!\Big(\sum_{l}\mathrm{reshape}\big(\overline{\hat{E}^{(l)}}\big)\Big)$$

评测协议按原文 §4.1：提示模板 "a photo of a [class]"；热图按 0.5 二值化得前景／背景分割；指标为 mIoU、像素准确率、mAP #文献 Zotero key 8S8R9TW4，§4.1。单图指标定义（$P$ 预测前景、$G$ 真值前景、$V$ 有效像素，即排除标注 id=1000 的忽略区）#代码事实 `d:\code\LeGrad\imagenets_eval.py` 第 162–178 行：

$$\mathrm{IoU}=\frac{|P\cap G|}{|P\cup G|},\qquad \mathrm{PixAcc}=\frac{\sum_{i\in V}\mathbb{1}[p_i=g_i]}{|V|},\qquad AP=\sum_k (R_k-R_{k-1})\,P_k$$

其中 AP 不依赖阈值：将热图分数降序排列后扫遍截断点，取精度-召回曲线下面积 #代码事实 `imagenets_eval.py` 第 108–124 行。前缀 m 表示对全部图像取平均 #代码事实 `imagenets_eval.py` 第 197–202 行。两处实现口径需注明：热图由 224（或 448）双线性上采样至掩码原图尺寸后再比较；多目标掩码中非目标物体计入背景 #代码事实 `imagenets_eval.py` 第 163、171 行。

### 3 数据与完整性核查

| 项                  | 取值                                                                                  | 依据                                                                     |
| ------------------ | ----------------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| 图像＋掩码来源            | HF 复刻 `braceletboy/imagenet-s` validation，4 个 parquet 分片共 1.71 GB                   | #实测 `D:\datasets\imagenets\parquet\`                                |
| 官方元数据              | `LUSSeg/ImageNet-S` 的 mapping／names／categories（经 jsDelivr 获取）                       | #实测 `D:\datasets\imagenets\meta\`                                   |
| 行数／可用行数            | 12,419／12,403（16 张掩码不含目标类，跳过）                                                       | #实测 `probe_sweeps.json` → `dataset_integrity`                       |
| 掩码编码               | $\mathrm{id}=R+256G$；忽略区 1000；其它物体 0                                                | #实测 `probe_sweeps.json` → `dataset_integrity`（编码经解码后与 names 映射比对校验） |
| parquet `label` 字段 | 与掩码 id 不一致（例：首行 label=915 而掩码 id=851），弃用                                            | #实测 `probe_sweeps.json` → `dataset_integrity`                       |
| names 与 mapping 差异 | 12,419 中 1,652 张不同；names 为 ImageNet-S 修正标签、mapping 为 ImageNet-1k 原始标签；目标类以 names 为准 | #实测 同上                                                              |
| 单类／多类掩码            | 11,280／1,123                                                                        | #实测 同上                                                              |

类别名映射链路为：names 的 wnid → ImageNet-S 类 id（categories 排序下标 +1）→ ImageNet-1k 类名表 → 提示文本 #代码事实 `imagenets_eval.py` 第 36–48 行。

### 4 实验设置

| 项 | 取值 |
|---|---|
| 环境 | conda `legrad`：py 3.10.22、torch 2.7.1+cu128、torchvision 0.22.1+cu128、open_clip_torch 2.23.0、timm 0.9.12 #实测 `probe_sweeps.json` → `environment` |
| 模型 | OpenCLIP ViT-B/16，权重 `laion2b_s34b_b88k`，class token 聚合 #代码事实 `D:\Obsidian\Research\组会与实验\实验规划\2026-11-24-LeGrad在ImageNet-S上的复现与可视化.md` 约束节 |
| 固定超参 | `layer_index=-2`、ReLU 去负、min-max 归一化、阈值 0.5 #代码事实 同上；实现见 `d:\code\LeGrad\legrad\wrapper.py` 第 177–185 行 |
| 自变量 | 输入分辨率 {224, 448}；探针阶段另扫阈值与层数 |
| 评测代码 | `d:\code\LeGrad\imagenets_eval.py`（prepare／eval／report／viz／thresh 五阶段，支持断点续跑） |

### 5 结果

#### 5.1 定性（S1、S2）

单图 dry-run 中 "cat" 与 "remote control" 两个提示分别点亮猫与遥控器，互不串扰，热图形态与原文 teaser 一致 #实测 `D:\Obsidian\Research\组会与实验\附件\图片\LeGrad-ImageNetS\`：![[legrad-qual-cat.png]] ![[legrad-qual-remote.png]]
全量结果按 IoU 分位抽样的三联图显示：

高 IoU 样本前景覆盖完整（IoU 0.77，Komondor）
![[legrad-viz-iou077-komondor.png]] 
中位样本定位到目标主体但边界偏紧（IoU 0.48，冻原狼）
![[legrad-viz-iou048-wolf.png]]

低 IoU 样本集中于"铺满全图的 stuff 类"（IoU 0.10，window screen），此类真值前景占比接近 1，而逐图 min-max 归一化后的热图难以给出近全图前景 #实测 同目录
![[legrad-viz-iou010-windowscreen.png]]

![[legrad-viz-iou062.png]]

![[legrad-viz-iou031.png]]

#### 5.2 全量定量（S3）

| 配置 | n | mIoU | PixAcc | mAP |
|---|---|---|---|---|
| 224，layer -2 | 12,403 | 46.06 | 80.73 | 71.90 |
| 224，单类掩码子集 | 11,280 | 47.06 | 80.26 | 73.27 |
| 224，多类掩码子集 | 1,123 | 36.03 | 85.42 | 58.17 |
| 448，layer -2 | 12,403 | 45.51 | 81.20 | 74.64 |
| 448，单类掩码子集 | 11,280 | 46.38 | 80.65 | 75.95 |

#实测 `D:\datasets\imagenets\results\eval_s224_l-2.jsonl`、`D:\datasets\imagenets\results\eval_s448_l-2.jsonl`

#### 5.3 与原文对照

| 来源 | n | mIoU | PixAcc | mAP |
|---|---|---|---|---|
| 本轮 224 | 12,403 | 46.06 | 80.73 | 71.90 |
| 本轮 448 | 12,403 | 45.51 | 81.20 | 74.64 |
| 原文表 1 | 4,276 | 58.66 | 77.52 | 82.49 |

#文献 原文表 1，Zotero key 8S8R9TW4。mIoU 差距约 12–13 点，超过规划设定的 5 点止损线，按规划要求区分"实现错误"与"结论不成立"，排查过程见 5.4 与第 6 节 #代码事实 `D:\Obsidian\Research\组会与实验\实验规划\2026-11-24-LeGrad在ImageNet-S上的复现与可视化.md` 判读与止损节。

#### 5.4 差距排查探针

以下探针样本量为 800–1,500，不具统计显著性，仅作方向性证据 #实测 `probe_sweeps.json`。

1. **分辨率**：448 与 224 的 mIoU 相差 0.55 点（45.51 vs 46.06），mAP 提升 2.74 点；分辨率不构成 mIoU 差距的解释 #实测 5.2 表。
2. **二值化阈值**：扫 0.1–0.6，mIoU 峰值 48.05（阈值 0.40），该点 PixAcc 78.36 与原文 77.52 接近，但 mIoU 仍低约 10.6 点；在扫描范围内未出现能同时复现原文 (mIoU, PixAcc) 组合的阈值 #实测 `probe_sweeps.json` → `threshold_sweep`。
3. **归一化方式**：min-max 与仅除最大值在 0.5 阈值下结果差 0.04 点，不构成解释 #实测 同上。
4. **指标聚合**：按图平均 46.06、按类平均 46.13、图像中位数 47.75、类中位数均值 46.81，四种口径均远低于 58.66 #实测 `probe_sweeps.json` → `aggregation_variants`。
5. **层聚合深度**：layer -1／-4／-6／-12 在阈值 0.5 下 mIoU 为 37.60／46.87／45.16／39.98，-2 与 -4 最优，与代码默认值一致；更深或更浅均更差 #实测 `probe_sweeps.json` → `layer_sweep`。
6. **子集口径**：真值前景面积分档 mIoU 为 33.39（<0.1）至 52.22（0.3–0.4）；恰好 4,276 张的面积截断为 0.1507（含并列值），该子集 mIoU 仅 37.67；S50∪S300 validation 为 4,114 张；单类掩码为 11,280 张。三种候选口径均不等于原文 4,276 或其指标 #实测 `probe_sweeps.json` → `gt_area_bands`、`subset_probes`。

### 6 讨论

本轮 PixAcc 高于原文（81.20 vs 77.52）而 mIoU 低于原文（45.51 vs 58.66）的交叉形态，提示本轮二值化前景整体偏小或偏稀疏：背景像素占多数且判对率高时 PixAcc 不敏感，而前景漏检直接压低 IoU #实测 5.3 表。但 5.4 的阈值扫描显示，把阈值放宽到 mIoU 峰值（0.40）仍无法接近原文 mIoU，故"阈值口径"这一最直接的候选解释未获支持 #实测 `probe_sweeps.json` → `threshold_sweep`。

stuff 类失败模式（5.1 第三张）说明该基准含相当比例的全图级真值前景，而 LeGrad 的逐图 min-max 归一化在结构上不利于产生近全图前景；面积分档数据同时显示极小目标（<0.1）也是低分区 #实测 `probe_sweeps.json` → `gt_area_bands`，极小目标样张见 `d:\code\LeGrad\out\imagenets_samples\viz_iou0.07_*.png` 等四张 #实测 同目录。这两类样本拉低全量均值，但按类平均口径并未改善（46.13），故聚合方式同样不构成解释 #实测 同上 → `aggregation_variants`。

威胁效度：评测集为社区复刻版而非官方 release 掩码，完整性核查（3 节）显示复刻版掩码与官方 names 映射自洽，但无法排除官方 release 与复刻版之间存在个别掩码差异 #实测 `probe_sweeps.json` → `dataset_integrity`。原文未公开其 4,276 张子集的筛选条件与评测代码，故本轮与原文数字的对比属于"不同评测集上的对比"，不宜直接解读为方法失效 #文献 Zotero key 8S8R9TW4，PDF `C:\Users\fengl\Zotero\storage\DJN2RJBW\2404.03214v2.pdf`，§4.1 仅给出图像数而未给出筛选条件。

### 7 结论

- LeGrad 的定性行为在本地完整复现：提示可切换定位对象，高 IoU 样本前景覆盖完整 #实测 `D:\Obsidian\Research\组会与实验\附件\图片\LeGrad-ImageNetS\`。
- ImageNet-S validation 全量 12,403 张的基线数字为 mIoU 46.06／PixAcc 80.73／mAP 71.90（224）与 45.51／81.20／74.64（448）#实测 `D:\datasets\imagenets\results\eval_s224_l-2.jsonl`、`D:\datasets\imagenets\results\eval_s448_l-2.jsonl`。
- 在分辨率、阈值、归一化、聚合、层数五个口径上，公开实现的 mIoU 上限约为 47–48，未支持"实现口径差异"可解释与原文的 12 点差距 #实测 `probe_sweeps.json`。
- 原文 4,276 张子集的筛选口径无法从公开元数据重建，本轮全量数字应作为自有基准记录，与原文数字的对比需注明评测集不同 #实测 `probe_sweeps.json` → `subset_probes`。
- 规划 S3 止损条款已触发并按约执行：差异已标注，且"实现错误"候选解释经探针排除，未将差距归因于方法本身 #实测 5.4。

### 8 产物与复现指引

| 产物 | 路径 |
|---|---|
| 评测脚本（五阶段） | `d:\code\LeGrad\imagenets_eval.py` |
| 单图定性脚本 | `d:\code\LeGrad\repro.py` |
| 数据集（图像／掩码／manifest） | `D:\datasets\imagenets\` |
| 全量逐图结果 | `D:\datasets\imagenets\results\eval_s224_l-2.jsonl`、`eval_s448_l-2.jsonl` |
| 探针与口径核查数字 | `D:\datasets\imagenets\results\probe_sweeps.json` |
| 三联图样张 | `D:\Obsidian\Research\组会与实验\附件\图片\LeGrad-ImageNetS\` |

复现命令（环境 `legrad`）：`python imagenets_eval.py prepare` → `python imagenets_eval.py eval` → `python imagenets_eval.py report --tag s224_l-2`；样张用 `python imagenets_eval.py viz --device cpu`，极小目标样张加 `--frac-max 0.1`。

### 9 未知项与后续轮次

- #猜想 原文 4,276 张子集可能来自其内部数据版本或人工筛选（如仅保留可分割物体类），该假设本轮无法验证。
- #猜想 原文实现可能在热图归一化或上采样方式上与公开仓存在未释放差异，导致前景区域更大。
- 后续轮次候选：补 rollout／GradCAM／CheferCAM／TextSpan 基线做相对对比（规划 S4）；SigLIP 与 CoCa 路径验证（含空提示校正开关）；按类别粒度的失败模式分析。
