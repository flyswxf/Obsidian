DEX-AR 是一套面向自回归视觉语言模型（如 LLaVA）的可解释性方法。给定一张图与一段目标文本，它输出热图，标出模型逐 token 生成这段文本时依赖了图像哪些区域。方法内核三步：对中间层 logits（logit lens）关于注意力图求梯度；按「关注图像列 vs 文本列」之差动态过滤注意力头 $w_{l,t,i}=(S_{\text{img}}-S_{\text{text}})^{+}$；再按视觉相关性权重 $\delta_t$ 聚合成句级热图。本地仓库在原实现之外补齐了集群复现入口与 PascalVOC 批量清单生成，用于评估其迁移价值。

## 工具组成

- **`dexar`（本地包，包名 `dexar_torch`）**：`DexarWrapper` / `DexarResult`、`compute_dexar` 主逻辑，以及 `visualize` / `visualize_multi` 出图。
- **LLaVA-1.5-7B**：被解释的自回归 VLM，`llava-hf/llava-1.5-7b-hf`，fp16 + `eager` 注意力。
- **`repro_dexar.py`**：自建复现入口，`dry-run`（单图固定句）与 `batch`（manifest 批量，可带 bbox）。
- **`tools/make_voc_manifest.py`**：自建，从 PascalVOC2012 标注生成批量 manifest，bbox 归一化到 $[0,1]$。
- **`playground.py`**：官方最小示例（单图、固定目标句）。
- **`jobs/repro.sbatch`**：自建 Slurm 提交脚本。

## 关键函数

| 函数 / 对象 | 位置 | 作用 |
| --- | --- | --- |
| `DexarWrapper.from_pretrained` | `dexar/wrapper.py` | 加载 LLaVA（`eager` + fp16 + `output_attentions`/`hidden_states`），按模型选 processor revision |
| `DexarWrapper.__init__` | `dexar/wrapper.py` | 冻结全部参数，仅 `language_model` 开梯度；缓存 `num_layers`、`lm_head`、`norm`；`num_img_tokens=576` |
| `compute_dexar` | `dexar/wrapper.py` | 主入口：逐 token 前向 + logit lens 梯度 + 头过滤 + $\delta_t$ 聚合，返回 `DexarResult` |
| `min_max` | `dexar/utils.py` | min-max 归一化到 $[0,1]$ |
| `topk_norm` | `dexar/utils.py` | 对 top-k 绝对值求均值，作注意力头打分 |
| `visualize` | `dexar/utils.py` | 单张热图 JET 叠加原图，可 `save_path` |
| `visualize_multi` | `dexar/utils.py` | 逐 token 热图网格，附每 token 单图 |
| `focus_metrics` | `repro_dexar.py` | 聚焦度代理：`top10_mass`、`entropy_norm`、`peak` |
| `bbox_metrics` | `repro_dexar.py` | 定位指标：`peak_in_bbox`、`mass_in_bbox`（bbox 归一化） |
| `profile_forward` | `repro_dexar.py` | 纯前向计时，量化 DEX-AR 相对开销 |
| `generate_target` | `repro_dexar.py` | 让模型贪心生成回答，作为被解释的目标句 |
| `make_voc_manifest.main` | `tools/make_voc_manifest.py` | PascalVOC2012 → 批量 manifest |

## `compute_dexar` 的四个步骤

```text
1. processor(text=prompt, images=image) → input_ids / pixel_values；目标句编成 target_ids
2. 对 target_ids 逐 token（teacher forcing）：
   前向一次，对 starting_depth..L-1 每层用 logit lens 取该 token 的中间层 logit，
   对第 l 层注意力图求 autograd.grad，切出图像列梯度与文本列梯度
3. 动态头过滤 w=(S_img-S_text)^+ 加权图像梯度 → 逐 token 热图；δ_t 聚合成句级热图
4. 逐 token 与句级热图均 min-max 归一化后返回
```

`layer_index` 决定累加起始层：`layer_index=0` 即全层（官方 `playground.py`），默认 `-10` 只取末尾 10 层。

环境版本与权重缓存见[[部署]]，从单图到批量的运行顺序见[[工作流]]；本轮复现结论见[[DEX-AR复现实验报告]]，迁移判断见[[注意力可视化审核-生成式VLM新方法]]。
