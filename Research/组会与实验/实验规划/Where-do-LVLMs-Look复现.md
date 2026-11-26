---
状态: 待执行
日期: 2026-11-26
研究线: "[[注意力可视化]]"
基础: "[[注意力可视化审核-生成式VLM新方法]]、[[注意力可视化-待填论文详细摘要]]"
---

**目标**

在 centos7-lab 跑通官方仓 `bytedance/LVLM_Interpretation`（iGOS++ 扰动显著性），复现其在 LLaVA-1.5-7B 上相对 Grad-CAM／T-MM／IIA 的定位优势，并做成可复用工具仓（`D:\code\LVLM_Interpretation`）。

**约束**

- 本地 8 GB 显存装不下 LLaVA-1.5-7B fp16，全流程走 centos7-lab Slurm，登录节点只做同步与轻量检查 #实测。
- 用官方仓既有实现，不重写内核；被解释模型固定为 LLaVA-1.5-7B（`llava-hf/llava-1.5-7b-hf`）。
- 方法需可微标量目标（关键 token 累计 log 概率）与对图像掩码的优化，`attn_implementation` 无关（读 logits 非注意力），冻结参数、不重训 #代码事实 `D:\code\LVLM_Interpretation\utils.py`、`methods_helper.py`。
- 该口径为扰动／梯度显著性，非注意力权重；尺度为掩码优化分辨率（默认 32×32），与 `cross_attention_raw` 的 24×24 不同源，只作互补对照。
- 集群任务遵守 `cluster-safety`，代码同步用 bare 仓库拓扑并登记到 `git-cluster-sync` 映射表。

**待验证假设**

- H1 `[文献]` 该方法在所有模型／数据集上取得最低删除分与最高插入分（如 LLaVA-1.5 MMVP 0.366／0.811）#文献 Zotero key VAI8N77N。
- H2 `[猜想]` 官方仓在集群环境可直接跑通 LLaVA-1.5，无须改内核，只需补集群入口与批处理。
- H3 `[猜想]` 关键 token 选择（LLR 阈值）与 `--size` 掩码分辨率是主要口径旋钮，调整可影响结果稳定性。

**变量与最小矩阵**

自变量：解释方法（iGOS++ 对 Grad-CAM 对 T-MM／IIA）。

| 阶段 | 内容 | 条件×样本 | 前置 |
| --- | --- | --- | --- |
| S0 | 按官方 `install.sh` 建环境并单图跑通 | LLaVA-1.5-7b × 3 图 | 集群环境 + 权重缓存 |
| S1 | 在 MMVP／MMStar 子集算删除／插入 AUC | 同模型 × ≥30 题 | S0 |
| S2 | 迁移到 patch 攻击图，导出掩码并重采样到 24×24 与 `cross_attention_raw` 对照 | 攻击样例 × 8 图 | S0 |

**判读与止损**

- 主指标：删除分（越低越好）与插入分（越高越好，AUC），须优于同模型 Grad-CAM／T-MM／IIA 基线，达标记「结果优秀」。
- 拆分口径：单样本耗时、掩码分辨率、与 `cross_attention_raw` 的空间相关性。
- fail 须归因到「实现错误」或「结论不成立」；前者修实现后重跑，连续两次未修通 → 暂停该线转做他行。
- S0 因环境／依赖／显存报错未通过不算 fail，先修通再接 S1／S2。

**未知项**

- 官方环境依赖（PyTorch 版本、`qwen_vl_utils` 等）与集群 glibc 2.17 的兼容性未验证，`install.sh` 能否直接复用待测。
- 基准数据集（MMVP／MMStar／CV-Bench／LLaVA-Bench）与 Grad-CAM／T-MM／IIA 基线脚本的可获取性未确认；若基线缺脚本须自建，开工前确定。
- 与 `cross_attention_raw` 逐格对照的重采样与出图、选层体检口径尚未确定。
