---
状态: 完成
日期: 2026-11-26
研究线: "[[注意力可视化]]"
基础: 4 篇论文全文 + 官方代码仓 + Zotero 入库记录
---

## 背景与目的

围绕「原生面向生成式 VLM 的注意力归因」方向检索，按 [[zotero]] 规则（近 3 年、CCF-A/最新 arXiv、必附 PDF）入库，并新增到 [[注意力可视化]] 规划表。

本报告对入选的 4 篇核心论文，沿用同一三条判据：

1. **同类性**：是否面向生成式、产出文本的多模态模型（decoder-only VLM，如 LLaVA／Qwen-VL），而非 CLIP 这类对比／分类模型。
2. **迁移便利性与通用性**：能否迁移 LLaVA 并通用到各类生成文本多模态模型；须指明改造点。
3. **契合度**：与实验两类注意力信号的接近程度。

迁移目标实验信号 #代码事实 `D:\code\Physical-Attention-Attack\src\models\llava_model.py`：`cross_attention_raw`（第 129–151 行）为逐生成步取最后一层注意力中「当前 query token → 图像 patch」分量，头平均、跨生成步平均，$576=24\times24$；`vision_attention_raw`（第 176–183 行）为视觉塔 CLS→patch 自注意力。

## 检索与筛选

两轮检索（WebSearch）覆盖「generative VLM／LVLM attention explainability／attribution」「decoder-only VLM attention rollout」「visual token attribution MLLM」等方向，得候选：GLIMPSE、DEX-AR、Where do LVLMs Look、LVLM-Interpret、MM-SHAP decoder（ICLR 2025）、Grad-ELLM、OTaT。经与用户确认，**入选 4 篇核心论文**（不收录次要候选）。

## 方法逐一审查

### DEX-AR（arXiv:2603.06302，ECCV 2026）

- **面向模型／任务**：自回归 VLM，论文与代码自述「explainability method for autoregressive Vision-Language Models (VLMs) such as LLaVA」；支持模型清单为 LLaVA-1.5-7B（`llava-hf/llava-1.5-7b-hf`）与 BakLLaVA——**与实验模型完全一致**。

- **方法要点**：逐生成 token 计算，(1) 中间层 logits 对注意力图的层梯度（logit lens）；(2) 动态头过滤，头部按「对图像 vs 对文本的关注差」加权 $w=(S_{\text{img}}-S_{\text{text}})^{+}$（式 5）；(3) 以视觉相关性权重 $\delta^t$ 聚合为句级热图（式 6）。

- **代码支持**：官方仓 `WalBouss/DEX-AR`（已克隆至 `D:\code\DEX-AR`）。#代码事实 `D:\code\DEX-AR\dexar\wrapper.py`：`self.num_img_tokens = 576  # 24x24 patch grid`（第 57 行）；`idx_img_tok_start/end`（第 142–143 行）；主循环逐 token 取 `grad[:, :, -1, idx_img_tok_start:idx_img_tok_end]`（第 199 行）即「当前 query token 末位 → 图像列」；输出 `per_token_heatmaps [T,24,24]` 与 `sentence_heatmap [24,24]`（第 239、264 行）。

- **迁移到 LLaVA 生成式 VLM 的可行性：可直接迁移（首选）。** 其网格 $24\times24$、`num_img_tokens=576`、逐 token 切片区间与实验 `llava_model.py` 的 `token_attn[image_start_idx : image_start_idx + num_image_tokens].reshape(24,24)` **完全同构**；本质是 `cross_attention_raw` 的「梯度加权 + 多层 + 头过滤」加强版，可直接接入 `attention_metrics.py` 的 bbox 指标。改造点：(i) 需 `attn_implementation="eager"` 且开启反传（实验为纯前向读注意力，成本与口径不同）；(ii) 依赖 HF LLaVA 内部路径 `language_model.model.layers / lm_head / norm`。

- **对各类生成文本多模态模型的通用性**：高。LLaVA 系列近乎直接可用；换其他 HF decoder VLM（Qwen-VL 等）须改内部路径与 image token 数，方法内核（注意力层梯度）模型无关。

- **判定**：同类性满足；判据 2 可直接迁移；判据 3 与 `cross_attention_raw` 同源同网格。记「可直接迁移」。

### GLIMPSE（arXiv:2506.18985，2025）

- **面向模型／任务**：自由形式 LVLM 回答。论文自述「the first attention–gradient framework capable of explaining entire free-form LVLM responses」「lightweight, model-agnostic」。

- **方法要点**：融合 gradient-weighted attention、adaptive layer propagation、relevance-weighted token aggregation，产出 response-level 热图，兼顾视觉与文本显著性；VQA-HAT 上人类注视对齐为 SOTA（rank correlation +46%、NSS +71%）。

- **代码支持**：**无官方仓**（检索到的同名 `glimpse` 仓为 ICLR 2025「LLM-generated text detection」，与本论文无关），须按论文自实现。

- **迁移到 LLaVA 的可行性：需改造（自实现）。** response-level「跨生成步聚合」与实验「跨生成步平均」同思路，且为注意力-梯度口径，与 `cross_attention_raw` 兼容；但无代码支撑，复现与验证成本高。

- **通用性**：方法自述 model-agnostic，理论可迁移至各类生成式 VLM。

- **判定**：同类性满足；判据 2 无代码、须自实现；判据 3 概念契合。记「需改造（无代码）」。

### Where do LVLMs Look（arXiv:2503.13891，2025，ByteDance）

- **面向模型／任务**：任意结构 LVLM 的开放式回答。论文自述「the first heatmap visualization method that applies to any LVLM structure and generate a global interpretation for open-ended responses」。

- **方法要点**：基于 iGOS++ 的扰动／梯度显著性优化（非注意力本体），面向需视觉信息作答的基准做全局解释。

- **代码支持**：官方仓 `bytedance/LVLM_Interpretation`（已克隆至 `D:\code\LVLM_Interpretation`）。入口 `main.py --method iGOS+ --model llava`，支持 LLaVA-1.5／LLaVA-OneVision／Cambrian／Mini-Gemini；环境遵循 LLaVA 官方仓 `install.sh`。

- **迁移到 LLaVA 的可行性：需改造（口径互补）。** 产出同为 $24\times24$ 网格级空间显著性、面向 open-ended response，可与实验指标对接；但口径是「扰动／梯度显著性」而非注意力权重，属**互补对照**而非同源。改造点：环境与实验 HF transformers 4.49 需对齐，且其优化式口径与 `cross_attention_raw` 不可直接等同。

- **通用性**：中-高（原生多模型支持）。

- **判定**：同类性满足；判据 2 需环境改造；判据 3 为互补口径。记「需改造（互补口径）」。

### LVLM-Interpret（arXiv:2404.03118，CVPRW 2024，IntelLabs）

- **面向模型／任务**：LLaVA 等 transformer-LLM 前端的 LVLM。提供 raw attention、relevancy maps、causal interpretation 三类解释。

- **方法要点**：把 relevance propagation 适配到 LVLM——relevancy 分数沿 LLM 与 ViT 反传，图像 patch 对应分数 reshape 成原图网格。

- **代码支持**：官方仓 `IntelLabs/lvlm-interpret`（已克隆至 `D:\code\lvlm-interpret`），**仓库已归档（archived，Intel 不再维护）**。Gradio 交互工具，`python app.py --model_name_or_path llava-hf/llava-1.5-7b-hf`。

- **迁移到 LLaVA 的可行性：需改造。** relevancy map reshape 到 patch 网格正对 `cross_attention_raw`；但工具为交互式 Gradio，抽取口径需读源码，归档状态带来维护风险。

- **通用性**：中（面向 transformer-LLM 前端，理论通用）。

- **判定**：同类性满足；判据 2 需读源码迁移且工具归档；判据 3 同源。记「需改造（工具归档）」。

## 汇总判定

| 论文（venue/年份） | Zotero key | 面向模型类 | 迁移结论 |
| --- | --- | --- | --- |
| DEX-AR（arXiv 2026／ECCV 2026） | 6U2FFLUA | 自回归 VLM（LLaVA-1.5-7B、BakLLaVA） | **可直接迁移（首选主线）**：同模型同网格同逐 token 口径 |
| GLIMPSE（arXiv 2025） | ATGW7RFS | 自由形式 LVLM | 需改造（无官方仓，须自实现） |
| Where do LVLMs Look（arXiv 2025） | VAI8N77N | 任意结构 LVLM 开放回答 | 需改造（iGOS++ 显著性，互补对照） |
| LVLM-Interpret（CVPRW 2024） | YXRLTYZC | LLaVA 等 | 需改造（relevancy map 同源；工具已归档） |

**首要结论**：4 篇**均原生面向生成式／自回归 VLM**，与实验同类性全部满足。其中 **DEX-AR 最契合**：直接包裹与实验同一 HF LLaVA-1.5-7B，$576=24\times24$ 网格与逐 token 切片口径与 `cross_attention_raw` 同构，迁移成本最低。

## 入库与代码仓记录

- Zotero「注意力可视化」collection（`N5QFDSQS`）新增 4 条，均附 PDF 全文：#文献 Zotero key 6U2FFLUA、ATGW7RFS、VAI8N77N、YXRLTYZC。
- 官方代码仓克隆：`D:\code\DEX-AR`、`D:\code\LVLM_Interpretation`、`D:\code\lvlm-interpret`；GLIMPSE 无官方仓。

## 对规划表的更新

[[注意力可视化]] 规划表新增上述 4 行（结果列均「待填」）。

## 未知项

- DEX-AR 为梯度-注意力方法，需 eager attention + 反传，与实验纯前向 `cross_attention_raw` 的口径／开销差异须先做单图 dry-run 量化。
- GLIMPSE 无代码，若纳入需自实现，成本待评估。
- 4 篇与实验 `attention_metrics.py` 的 bbox 指标对接均未实测。
