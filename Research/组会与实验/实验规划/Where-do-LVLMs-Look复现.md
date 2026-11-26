---
状态: 待执行
日期: 2026-11-26
研究线: "[[注意力可视化]]"
基础: "[[注意力可视化-待填论文详细摘要]]、官方仓 `D:\code\LVLM_Interpretation`"
---

**目标**

用官方仓按原文设置复现 iGOS++ 在 LLaVA-1.5-7b 等 LVLM 上的定位优势（删除分最低、插入分最高），结果与论文同量级，并做成可复用工具仓 `D:\code\LVLM_Interpretation`；跑通后输出报告与工具部署三文档。

**约束**

- 忠实原文：模型、数据集、指标、基线一律照原文，不自创对照 #文献 Zotero key VAI8N77N。
- 用官方实现，不重写内核；官方仓只含 iGOS++（`methods.py`），基线须由「[[注意力可视化基线工具]]」提供 #代码事实 `D:\code\LVLM_Interpretation\methods.py`。
- 数据取 HF `xiaoying0505/LVLM_Interpretation`，按原文字段过滤（保留 MMVP 35%、MMStar 47%、CV-Bench 34%，去选项与指令转开放式）#文献 Zotero key VAI8N77N。
- 指标为删除分（越低越好）与插入分（越高越好，AUC），按原图与基线图分数归一化 #文献 Zotero key VAI8N77N。
- 本地 8 GB 显存装不下 7B，全流程走 centos7-lab Slurm；集群遵守 `cluster-safety`，同步按 `git-cluster-sync`。

**待验证假设**

- H1 `#文献` 该方法在所有模型／数据集上取得最低删除分、最高插入分（LLaVA-1.5-7b：MMVP 0.366／0.811、MMStar 0.292／0.953、CV-Bench 0.402／0.965、LLaVA-Bench 0.358／0.864）#文献 Zotero key VAI8N77N。
- H2 `#文献` 消融：单掩码配模糊基线最佳，分离掩码略优但单样本 19.1s vs 7.4s #文献 Zotero key VAI8N77N。
- H3 `#猜想` 官方仓在集群可直接跑通 LLaVA-1.5，只需补集群入口与批处理。

**变量与最小矩阵**

自变量：解释方法（iGOS++ 对 Grad-CAM [39]、T-MM [9]、IIA [4]）。

| 阶段 | 内容 | 条件×样本 | 前置 |
| --- | --- | --- | --- |
| S0 | 按官方 `install.sh` 建环境并单图跑通 | LLaVA-1.5-7b × 3 图 | 集群环境 + 权重 + 基线工具 |
| S1 | 过滤子集上算删除／插入分（复现表 1） | 同模型 × 过滤子集 | S0 |
| S2 | 消融：token 选择／基线图／单掩码（表 3） | 同模型 × 子集 | S1 |
| S3 | 换 LLaVA-OV／Cambrian 复现（次优先） | 视显存 | S1 |

**判读与止损**

- 主判据：LLaVA-1.5-7b 的删除／插入分复现论文表 1 的量级与最优排序，即 pass。
- 基线 T-MM／IIA 因 Cambrian 用非 transformer 编码器而不适用，该项不做 #文献 Zotero key VAI8N77N。
- fail 归因「实现错误」或「结论不成立」；前者修实现重跑，连续两次未修通 → 暂停转他行。
- S0 因环境／依赖／显存报错未通过不算 fail，先修通再接 S1／S2。

**未知项**

- 官方 `install.sh` 的环境依赖与集群 glibc 2.17 的兼容性未验证。
- 基线 Grad-CAM／T-MM／IIA 无官方脚本，依赖「[[注意力可视化基线工具]]」对齐，其可信度待测。
- 数据集在 HF 的完整性与许可、集群下载路径未确认。
