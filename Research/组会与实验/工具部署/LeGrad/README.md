LeGrad 是一套面向 ViT / 视觉语言模型的可解释性方法。它不改动 OpenCLIP 原模型，而是以 wrapper 形式给中间层挂 hook，对「文本 embedding 与逐层中间特征的相似度」关于注意力图求梯度，正梯度跨层累加后得到一张与任意文本 prompt 绑定的定位热图。本地仓库在原实现之外补齐了 ImageNet-S 全量评测入口，用于复现原文表 1。

## 工具组成

- **`legrad`（本地包）**：`LeWrapper` 挂钩与热图计算、`LePreprocess` 任意分辨率预处理、`visualize` 热图叠加出图。
- **OpenCLIP**：视觉-语言 backbone，默认 `ViT-B-16` + `laion2b_s34b_b88k` 权重。
- **ImageNet-S**：12,419 张带像素级标注的 ImageNet 验证图，919 类，用于分割指标评测。
- **`repro.py`**：单图 + 自定义 prompt → 热图 PNG，最快的肉眼验证入口。
- **`imagenets_eval.py`**：`prepare / eval / report / viz / thresh` 五阶段全量评测，支持断点续跑。
- **`playground.py` / `app.py`**：官方最小示例与 Gradio demo（联网取 COCO 图）。

## 关键函数

| 函数 | 位置 | 作用 |
| --- | --- | --- |
| `LeWrapper.__init__` | `legrad/wrapper.py` | 复制原模型全部属性，再调用 `_activate_hooks`，原模型功能不受影响 |
| `_activate_hooks` | `legrad/wrapper.py` | 按 `visual` 类型分派：`VisionTransformer` → `clip` / `coca`，`TimmModel` → `timm_siglip`；算出 `starting_depth`、`patch_size` |
| `compute_legrad` | `legrad/wrapper.py` | 统一入口，按 `model_type` 分派到下面三条分支 |
| `compute_legrad_clip` | `legrad/wrapper.py` | CLIP 分支，本次复现走的就是这条 |
| `compute_legrad_siglip` | `legrad/wrapper.py` | SigLIP 分支，额外用空 prompt `"a photo of a"` 算修正图并按 `correction_threshold` 抑制 |
| `compute_legrad_coca` | `legrad/wrapper.py` | CoCa 分支，中间特征过 `attn_pool` 后再求梯度 |
| `LePreprocess` | `legrad/wrapper.py` | 把原预处理的首个 `Resize` 换成指定 `image_size`，让 ViT 吃 448 等非 224 输入 |
| `hooked_resblock_forward` | `legrad/utils.py` | 把每层 `feat_post_attn` / `feat_post_mlp` 缓存到 block 上 |
| `hooked_torch_multi_head_attention_forward` | `legrad/utils.py` | 复刻 PyTorch MHA 前向并把 `attention_maps` 挂到 `attn` 上；关键改动是**不再对 head 求平均** |
| `hooked_attentional_pooler_timm_forward` | `legrad/utils.py` | timm/SigLIP 分支用，缓存 `attn_probs` |
| `vit_dynamic_size_forward` | `legrad/utils.py` | 输入尺寸与位置编码不一致时用 `resample_abs_pos_embed` 做 bicubic 重采样 |
| `min_max` | `legrad/utils.py` | 逐 `(batch, prompt)` 把热图归一化到 `[0,1]` |
| `visualize` | `legrad/utils.py` | JET 色图叠加原图并 `savefig` |
| `list_pretrained` | `legrad/utils.py` | 列出 LeGrad 支持的 OpenCLIP 权重（过滤掉 CNN 系） |

## `compute_legrad_clip` 的六步

```text
1. encode_image 前向一次，hook 缓存每层 feat_post_mlp 与 attention_maps
2. 对 starting_depth..L-1 每一层：patch 特征沿 token 求均值 → ln_post → @proj → normalize
   得到「假设在这一层就退出」的图像 embedding
3. sim = text_embedding @ image_embedding，对 attention_maps 求 grad
4. grad.clamp(min=0) → mean(heads) → mean(query 行) → 去掉第 0 列（CLS）
5. rearrange 成 w×h，F.interpolate(scale_factor=patch_size) 上采样，跨层累加
6. min_max 归一化 → [B, num_prompt, H, W]
```

`layer_index=-2` 时 `starting_depth = 12 - 2 = 10`，即只累加最后两层；`--layer-index -12` 才是全层累加。

## 相关笔记

- 环境版本、依赖、数据集落盘位置与兼容性坑：[[部署]]
- 从单图冒烟到全量评测的运行顺序与结果：[[工作流]]
- 文献笔记：[[LeGrad-注意力梯度敏感性解释]]（Zotero key `8S8R9TW4`）
- 实验计划：[[2026-11-24-LeGrad在ImageNet-S上的复现与可视化]]
