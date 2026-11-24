- 状态：待执行｜日期：2026-11-24｜研究线：[[2026-11-19-注意力可视化新文献复现规划|注意力可视化复现]]｜基础：[[LeGrad-注意力梯度敏感性解释]] + `WalBouss/LeGrad`

**目标**

在 ImageNet-Segmentation 上复现 LeGrad，复算原论文表 1 的 mIoU／像素准确率／mAP 口径，并产出可视化热图（原图＋热图＋真值掩码三联图）。

**约束**

- 固定基线：OpenCLIP ViT-B/16（Laion2B 权重）、class token 聚合、目标类激活取自文本提示 "A photo of a [class]"（零样本口径，与原文表 1 一致）。
- 固定超参：全层平均、ReLU 去负梯度、min-max 归一化、阈值 0.5 二值化；除"层数"外不改动。
- 只能二选一之处：激活来源取零样本文本提示（主）或分类头（备），不同时混用，结果需标注所用口径。
- 算力：GPU 任务走 Slurm，先小样本 dry-run 再全量；使用（59.78.189.140）服务器
- 数据：ImageNet-Segmentation 4,276 张带分割标注图；若服务器无外网，权重与安装包需本地 `scp` 搬运后离线安装。

**待验证假设**

- H1 `[文献]` LeGrad 在 ImageNet-S 上 mIoU 58.66、像素准确率 77.52、mAP 82.49，三项均最优（[[LeGrad-注意力梯度敏感性解释|原文表 1]]，Zotero key 8S8R9TW4）。
- H2 `[文献]` LeGrad 优于 rollout、raw attention、GradCAM、CheferCAM、LRP、TextSpan（同上表）。
- H3 `[代码事实]` 官方仓 `WalBouss/LeGrad` 提供 `legrad-torch` 包；是否自带 ImageNet-S 分割评测入口与提示模板待确认（见未知项）。
- H4 `[猜想]` 去掉 ReLU 或只用末层特征求梯度会降低 mIoU（依据原文表 9 组件消融）。

**变量与最小矩阵**

- 自变量：方法（LeGrad／对照）、层数（全层／单层）、样本规模（3 张 smoke → 20 张出图 → 全量 4,276）。

| 阶段  | 内容         | 条件×样本                                                        | 前置  |
| --- | ---------- | ------------------------------------------------------------ | --- |
| S0  | 环境与权重就位    | `pip install legrad-torch`，下载 OpenCLIP ViT-B/16 与 ImageNet-S | —   |
| S1  | 单图 dry-run | LeGrad × 3 图，核对热图形态与量级                                       | S0  |
| S2  | 可视化出图      | LeGrad × 20 图（原图／热图／掩码三联图）                                   | S1  |
| S3  | 全量指标复算     | LeGrad × 4,276 图                                             | S2  |
| S4  | 对照复算（不做）   | rollout／GradCAM／CheferCAM × 4,276 图                          | S3  |

**判读与止损**

- 主指标 mIoU（0.5 二值化）；拆分口径按方法与层数。
- S1 若热图形态与原文示例不符（热点弥漫／全背景／单向偏置），判为实现错误，先排查 [CLS] 列去除与归一化，不进入全量。
- S3 若 mIoU 低于原文 5 点以上，须标注差异并区分"实现错误"与"结论不成立"，不得直接判方法失效。
- 止损：优先参考legrad自身代码仓

**未知项**

- 官方仓是否自带 ImageNet-S 分割评测脚本与 "A photo of a [class]" 提示模板。
- ImageNet-S 类别下标与 OpenCLIP 文本提示的映射对齐方式。
- 组内服务器实际可用显存能否支撑 ViT-B/16 批量前向。
