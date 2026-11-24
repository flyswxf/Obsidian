> [!info] 目标
> 把 `wrapper.py` 里 `compute_legrad_clip` 的 43 行代码逐行讲清楚。默认已经读过 [[01-张量形状与常用API]]、[[03-梯度与autograd]]、[[05-玩具例子手算LeGrad]]。
> 举例规模：**ViT-B/16、输入 448×448、3 个 prompt**，即 $P=3$、patch 网格 $28\times28$、$T=785$、$h=12$、$d=768$、$d_t=512$、`patch_size=16`。

## 1. 完整调用链

```python
import open_clip, torch
from legrad import LeWrapper, LePreprocess, visualize

# ① 建原始模型
model, _, preprocess = open_clip.create_model_and_transforms(
    model_name='ViT-B-16', pretrained='laion2b_s34b_b88k', device=device)
tokenizer = open_clip.get_tokenizer(model_name='ViT-B-16')
model.eval()                                   # 必须！否则注意力图带 dropout 随机性

# ② 装上 LeGrad（就地改原模型：装 hook + 关前 10 层梯度）
model = LeWrapper(model, layer_index=-2)
preprocess = LePreprocess(preprocess=preprocess, image_size=448)

# ③ 准备输入
image = preprocess(pil_image).unsqueeze(0).to(device)          # (1, 3, 448, 448)
text = tokenizer(['a photo of a cat.', 'a photo of a dog.']).to(device)
text_embedding = model.encode_text(text, normalize=True)       # (2, 512)

# ④ 算热图
with torch.enable_grad():
    heatmaps = model.compute_legrad(text_embedding=text_embedding, image=image)
                                                               # (1, 2, 448, 448)
# ⑤ 叠图显示
visualize(image=image, heatmaps=heatmaps, text_prompts=['cat', 'dog'])
```

几个容易踩的点：

- **`model.eval()` 与 `torch.enable_grad()` 是两件独立的事**。`eval()` 只关掉 dropout/BN 的训练行为，**不会**关掉梯度；反过来 `torch.no_grad()` 会让 `autograd.grad` 直接报错。有些地方（如推理脚本）习惯性地套了 `no_grad`，所以要显式 `enable_grad`。
- **`LeWrapper` 不是新建一个模型**，它把原模型的属性搬过来、`self.visual` 与原 `model.visual` 是同一个对象，hook 是就地打的（详见 [[04-hook机制]] 第 5.3 节）。
- **`compute_legrad` 是分发器**，按 `model_type` 转到三个实现之一：

```python
def compute_legrad(self, text_embedding, image=None, apply_correction=True):
    if 'clip' in self.model_type:
        return self.compute_legrad_clip(text_embedding, image)
    elif 'siglip' in self.model_type:
        return self.compute_legrad_siglip(text_embedding, image, apply_correction=apply_correction)
    elif 'coca' in self.model_type:
        return self.compute_legrad_coca(text_embedding, image)
```

`model_type` 在 `_activate_hooks` 里判定：`visual` 是 `VisionTransformer` 且 `attn_pool is None` → `'clip'`；有 `attn_pool` → `'coca'`；`visual` 是 `TimmModel` → `'timm_siglip'`。

`LePreprocess` 做的事：把 open_clip 默认的"按短边 Resize + CenterCrop"换成"直接 Resize 成 $448\times448$ 方形"，保留原 pipeline 的最后三步（RGB 转换、ToTensor、Normalize）。

```python
self.transform = Compose([
    Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
    preprocess.transforms[-3], preprocess.transforms[-2], preprocess.transforms[-1],
])
```

配合 `vit_dynamic_size_forward` 里的位置编码重采样，就能用高于 224 的分辨率跑，把热图网格从 $14\times14$ 提到 $28\times28$。副作用见 [[07-三条分支与复现坑位]]。

## 2. 逐行注释版源码

`wrapper.py` L144-186 全文，右侧是解释：

```python
def compute_legrad_clip(self, text_embedding, image=None):
    num_prompts = text_embedding.shape[0]                    # P = 3

    # ---------- ① 一次前向，填满所有 hook ----------
    if image is not None:
        image = image.repeat(num_prompts, 1, 1, 1)           # (1,3,448,448) → (3,3,448,448)
        _ = self.encode_image(image)                         # 返回值不要，只要 hook 里的副产物

    blocks_list = list(dict(self.visual.transformer.resblocks.named_children()).values())
                                                             # 12 个 resblock 的列表

    # ---------- ② 把每层的中间特征压成图像向量 ----------
    image_features_list = []
    for layer in range(self.starting_depth, len(self.visual.transformer.resblocks)):
        intermediate_feat = self.visual.transformer.resblocks[layer].feat_post_mlp
                                                             # Z^l: (785, 3, 768)  LND 布局
        intermediate_feat = self.visual.ln_post(intermediate_feat.mean(dim=0)) @ self.visual.proj
                                                             # 式(3): 平均→(3,768)→LN→@proj→(3,512)
        intermediate_feat = F.normalize(intermediate_feat, dim=-1)   # 每行单位化 → 点积=余弦
        image_features_list.append(intermediate_feat)
                                                             # 先全部收进列表，防止后续前向覆盖 hook

    num_tokens = blocks_list[-1].feat_post_mlp.shape[0] - 1   # 785 - 1 = 784（减掉 [CLS]）
    w = h = int(math.sqrt(num_tokens))                        # w = h = 28

    # ---------- ③④⑤ 逐层求梯度、聚合、累加 ----------
    accum_expl_map = 0
    for layer, (blk, img_feat) in enumerate(zip(blocks_list[self.starting_depth:],
                                                image_features_list)):
        self.visual.zero_grad()                              # 空操作（见 03 第 7.3 节）

        sim = text_embedding @ img_feat.transpose(-1, -2)     # 式(3): (3,512)@(512,3) → (3,3)
        one_hot = F.one_hot(torch.arange(0, num_prompts)).float().requires_grad_(True).to(...)
                                                             # 3×3 单位阵
        one_hot = torch.sum(one_hot * sim)                    # trace(sim) = Σ_p s_p，标量

        attn_map = blocks_list[self.starting_depth + layer].attn.attention_maps
                                                             # A^l: (36, 785, 785) = (P·h, T, T)

        grad = torch.autograd.grad(one_hot, [attn_map], retain_graph=True, create_graph=True)[0]
                                                             # 式(4): ∇A^l, (36, 785, 785)
        grad = rearrange(grad, '(b h) n m -> b h n m', b=num_prompts)   # → (3, 12, 785, 785)
        grad = torch.clamp(grad, min=0.)                      # 式(5): (·)^+

        image_relevance = grad.mean(dim=1).mean(dim=1)[:, 1:] # 式(5)(6): → (3,785) → (3,784)
        expl_map = rearrange(image_relevance, 'b (w h) -> 1 b w h', w=w, h=h)   # (1,3,28,28)
        expl_map = F.interpolate(expl_map, scale_factor=self.patch_size, mode='bilinear')
                                                             # (1,3,448,448)
        accum_expl_map += expl_map                            # 式(8): 逐层累加

    accum_expl_map = min_max(accum_expl_map)                  # 式(9): 逐 prompt 归一化
    return accum_expl_map                                     # (1, 3, 448, 448)
```

## 3. 五步拆解

### 步骤 ①：把一张图复制 $P$ 份

```python
image = image.repeat(num_prompts, 1, 1, 1)   # (1,3,448,448) → (3,3,448,448)
_ = self.encode_image(image)
```

三份**完全相同**的图像副本，第 $p$ 份专门配第 $p$ 个 prompt。返回值 `_` 被丢弃——要的只是 hook 顺手存在各层模块上的 `feat_post_mlp` 和 `attention_maps`。

这一行是 CLIP 分支独有的（CoCa / SigLIP 分支不做，见 [[07-三条分支与复现坑位]]），也是它能一次出 $P$ 张热图的前提。

对应玩具版：`Z0b = np.stack([Z0]*P, axis=1)`。

### 步骤 ②：把每层的 785 个 token 压成一个图像向量

```python
intermediate_feat = resblocks[layer].feat_post_mlp              # (785, 3, 768)
intermediate_feat = self.visual.ln_post(intermediate_feat.mean(dim=0)) @ self.visual.proj
intermediate_feat = F.normalize(intermediate_feat, dim=-1)      # (3, 512)
```

| 操作 | 形状 | 论文 |
| --- | --- | --- |
| `feat_post_mlp` | $(785, 3, 768)$ | $Z^l$，式(1) 的输出 |
| `.mean(dim=0)` | $(3, 768)$ | $\bar z^l = \frac{1}{n+1}\sum_i z_i^l$，式(3) |
| `ln_post(·)` | $(3, 768)$ | — |
| `@ proj` | $(3, 512)$ | 映射 $C$ |
| `F.normalize` | $(3, 512)$ | 让点积等于余弦相似度 |

`mean(dim=0)` 之所以是"对 token 平均"，是因为 open_clip 用 **LND 布局**，token 维在第 0 位（见 [[02-ViT内部结构]] 第 4 节）。这个平均**包含 `[CLS]`**，与式(3) 的 $\frac{1}{n+1}\sum_{i=0}^{n}$ 一致。

与原模型推理路径的差别：CLIP 正常是"先 `ln_post` 再取 `[CLS]`"，这里是"先平均再 `ln_post`"。LayerNorm 逐 token 独立归一化，$\mathrm{LN}(\bar z) \ne \overline{\mathrm{LN}(z)}$，两者数值不等价——LeGrad 用的正是式(3) 的定义，因为浅层的 `[CLS]` 还没汇总好信息，而"平均所有 token"对任何层都成立。

**为什么要先把所有层的特征收进 `image_features_list`**：hook 存的是实例属性，下一次前向就会覆盖。虽然这里只前向一次，但先收集再使用是防御性写法，也方便和 `blocks_list` 用 `zip` 配对。

### 步骤 ③：从 $3\times3$ 相似度矩阵里挑出对角线

```python
sim = text_embedding @ img_feat.transpose(-1, -2)     # (3, 3)
one_hot = F.one_hot(torch.arange(0, num_prompts)).float()
one_hot = torch.sum(one_hot * sim)                     # 标量
```

`sim[i,j]` = 第 $i$ 个 prompt 与第 $j$ 份图像副本的余弦相似度：

```text
              图像副本0  图像副本1  图像副本2
prompt 0 (cat) [  ★        ✗        ✗   ]
prompt 1 (dog) [  ✗        ★        ✗   ]
prompt 2 (car) [  ✗        ✗        ★   ]

★ = 对角线，唯一有意义的组合    ✗ = 交叉项，无意义
```

三份副本的图像内容相同，所以 `sim` 的三列其实数值一样；但因为**梯度只沿对角线流动**（第 $p$ 份副本的注意力图只出现在 $s_p$ 里），取对角线是正确且必要的做法。

`F.one_hot(torch.arange(3))` 造出 $3\times3$ 单位阵，逐元素乘再求和 = 取对角线之和：

$$s = \sum_p s_p = \mathrm{trace}(\text{sim})$$

对应玩具版 Part C 里验证过的结论：一次求导即可拿到 $P$ 份互不污染的梯度，误差 $10^{-11}$ 量级。

`requires_grad_(True)` 挂在单位阵上没有作用（从没对它求过导）。

### 步骤 ④：求梯度 → ReLU → 两次平均 → 丢 CLS

```python
attn_map = blocks_list[self.starting_depth + layer].attn.attention_maps   # (36, 785, 785)
grad = torch.autograd.grad(one_hot, [attn_map], retain_graph=True, create_graph=True)[0]
grad = rearrange(grad, '(b h) n m -> b h n m', b=num_prompts)             # (3, 12, 785, 785)
grad = torch.clamp(grad, min=0.)
image_relevance = grad.mean(dim=1).mean(dim=1)[:, 1:]                     # (3, 784)
```

| 代码 | 论文 | 形状 | 备注 |
| --- | --- | --- | --- |
| `attn.attention_maps` | $A^l$ | $(36,785,785)$ | batch 与 head 被 PyTorch 压在同一维 |
| `autograd.grad(...)` | 式(4) $\nabla A^l$ | $(36,785,785)$ | `retain_graph=True` **必需**；`create_graph=True` **多余** |
| `rearrange('(b h) n m -> b h n m')` | — | $(3,12,785,785)$ | 拆开 batch 与 head，否则没法只对 head 平均 |
| `clamp(min=0.)` | 式(5) $(\cdot)^+$ | 不变 | 丢负梯度 |
| `.mean(dim=1)` 第 1 次 | $\frac{1}{h}\sum_h$ | $(3,785,785)$ | 消 head |
| `.mean(dim=1)` 第 2 次 | $\frac{1}{n+1}\sum_i$ | $(3,785)$ | 消 query 行 |
| `[:, 1:]` | 下标 $1:$ | $(3,784)$ | 丢 `[CLS]` **列** |

**这一行 `grad.mean(dim=1).mean(dim=1)` 是全篇最容易看错的**：两次都是 `dim=1`，但第一次消的是 head，第二次消的是 query 行——因为第一次平均之后形状从 $(3,12,785,785)$ 变成了 $(3,785,785)$，原来的第 2 维升到了第 1 维。玩具版 [[05-玩具例子手算LeGrad]] 第 5 节有专门对照表。

**剩下的一维是"列"（key token）**，所以 $\hat E^l_j$ 的含义是"token $j$ **被关注**的程度拧一下，目标分数变多少"。正因为剩下的是 key 轴（= 图像位置），才能 reshape 成网格；`[:, 1:]` 丢的也是 `[CLS]` 作为 key 的那一列。

### 步骤 ⑤：reshape → 上采样 → 累加 → 归一化

```python
expl_map = rearrange(image_relevance, 'b (w h) -> 1 b w h', w=w, h=h)   # (3,784) → (1,3,28,28)
expl_map = F.interpolate(expl_map, scale_factor=self.patch_size, mode='bilinear')  # (1,3,448,448)
accum_expl_map += expl_map
...
accum_expl_map = min_max(accum_expl_map)     # (1,3,448,448)
```

- `rearrange` 里最前面那个 `1` 是新插的**通道维**占位，因为 `F.interpolate` 要求 4 维输入 $(N,C,H,W)$
- `scale_factor=16` 把 $28\times28$ 放大到 $448\times448$。热图的**真实信息量只有 784 个数字**，放大只是插值补出来的
- `accum_expl_map += expl_map` 对应式(8) 的 $\frac{1}{L}\sum_l$，代码用**求和**代替平均；常数 $1/L$ 会被后面的 min-max 归一化吃掉，所以等价
- 代码是"每层先插值到 448 再累加"，论文是"先累加再 reshape"。双线性插值是线性算子、与求和可交换，两种顺序结果等价，只是前者多占显存

`utils.min_max`（L498-503）：

```python
def min_max(logits):
    B, num_prompt = logits.shape[:2]                                  # B=1, num_prompt=3
    logits_min = logits.reshape(B, num_prompt, -1).min(dim=-1, keepdim=True)[0].unsqueeze(-1)
    logits_max = logits.reshape(B, num_prompt, -1).max(dim=-1, keepdim=True)[0].unsqueeze(-1)
    return (logits - logits_min) / (logits_max - logits_min)
```

把 $(1,3,448,448)$ 拉成 $(1,3,448\cdot448)$ 后按最后一维求 min/max，得到 $(1,3,1,1)$，再靠广播减回去。效果是**每个 prompt 的热图各自独立归一化到 $[0,1]$**。

两个后果：
1. 跨 prompt 比较热图数值没有意义（每张图的最大值都是 1）
2. 若某 prompt 的热图恰好全相等，分母为 0 → `nan`。代码没有加分母保护

最终返回 $(1, P, H, W)$，`visualize` 会 `squeeze(0)` 成 $(P,H,W)$ 后逐个 prompt 叠图。

## 4. 形状流水总表

```text
image           (1, 3, 448, 448)
  repeat        (3, 3, 448, 448)      ← P 份副本
  encode_image  ↓ hook 存下每层：
    feat_post_mlp     (785, 3, 768)   ← Z^l，LND
    attention_maps    (36, 785, 785)  ← A^l，(P·h, T, T)

每层循环：
    mean(dim=0)        (785,3,768) → (3, 768)         式(3) z̄^l
    ln_post @ proj     (3,768)     → (3, 512)         映射 C
    normalize          (3,512)     → (3, 512)
    text @ img.T       (3,512)×(512,3) → (3, 3)       式(3) ȳ^l
    Σ(one_hot * sim)   (3,3)       → 标量             s = Σ_p s_p
    autograd.grad      标量        → (36, 785, 785)   式(4)
    rearrange          (36,·,·)    → (3, 12, 785, 785)
    clamp(min=0)       不变                            式(5)
    mean(dim=1)        (3,12,785,785) → (3, 785, 785)  消 head
    mean(dim=1)        (3,785,785)    → (3, 785)       消 query 行
    [:, 1:]            (3,785)        → (3, 784)       丢 CLS 列
    rearrange          (3,784)        → (1, 3, 28, 28) 摆回网格
    interpolate ×16    (1,3,28,28)    → (1, 3, 448, 448)
    accum +=           (1,3,448,448)                   式(8)

循环外：
    min_max            (1,3,448,448) → (1, 3, 448, 448) 式(9)
```

## 5. 极简等价实现

与 `compute_legrad_clip` 数学等价，去掉 hook 安装、模型分支、einops，只用朴素索引：

```python
import torch
import torch.nn.functional as F


def layer_heatmap(blk, text_emb, ln_post, proj, patch):
    """单层热图，对应式 (3)~(6)。blk 为已装 hook 的 resblock。"""
    Z = blk.feat_post_mlp                          # (T, P, d)，LND 布局
    P, T = text_emb.shape[0], Z.shape[0]
    n = T - 1                                      # patch 数

    z_bar = Z.mean(dim=0)                          # (P, d)       式(3)：对 T 个 token 求平均
    img = F.normalize(ln_post(z_bar) @ proj, -1)    # (P, d_text)  映射 C
    s = (text_emb * img).sum()                     # 标量         Σ_p s_p，即对角线之和

    A = blk.attn.attention_maps                    # (P*h, T, T)  式(4) 的求导对象
    g = torch.autograd.grad(s, A, retain_graph=True)[0]
    g = g.view(P, -1, T, T).clamp(min=0.)          # (P, h, T, T) 式(5)：拆 head + ReLU

    E = g.mean(dim=1).mean(dim=1)[:, 1:]           # (P, n)       式(5)：先 head 后 query，丢 CLS 列
    w = int(n ** 0.5)
    E = E.view(P, 1, w, w)                         # 式(6)：reshape 成 2D 网格
    return F.interpolate(E, scale_factor=patch, mode='bilinear')


def legrad(model, image, text_emb, layer_index=-2, patch=16):
    P = text_emb.shape[0]
    _ = model.encode_image(image.repeat(P, 1, 1, 1))    # 一次前向，填满所有 hook
    blocks = list(model.visual.transformer.resblocks)
    acc = 0
    for blk in blocks[len(blocks) + layer_index:]:      # 式(8)：逐层累加
        acc = acc + layer_heatmap(blk, text_emb, model.visual.ln_post,
                                  model.visual.proj, patch)
    lo = acc.amin(dim=(-2, -1), keepdim=True)           # 式(9)：逐 prompt min-max
    return (acc - lo) / (acc.amax(dim=(-2, -1), keepdim=True) - lo)
    # → (1, P, H, W)
```

三处相对真实代码的改动，都不影响结果：

1. `create_graph=True` 去掉（全程没有二阶求导）
2. `zero_grad()` 去掉（`autograd.grad` 不写 `.grad`）
3. `rearrange` 换成 `view`，`min_max` 换成 `amin`/`amax`

`(text_emb * img).sum()` 等价于 `torch.sum(one_hot * sim)`：逐元素乘再全求和，就是 $\sum_p \sum_d \text{text}[p,d]\cdot\text{img}[p,d] = \sum_p \cos_p$。

## 6. 与玩具例子的对应

| 玩具版（[[05-玩具例子手算LeGrad]]） | 真实代码 | 规模差异 |
| --- | --- | --- |
| `T=5, n=4, d=4, dt=3, h=1~2` | `T=785, n=784, d=768, dt=512, h=12` | 只是数字变大 |
| `forward()` 里的手写前向 | `encode_image()` + hook | 真实版靠 hook 截获 |
| `for` 双重循环有限差分 | `torch.autograd.grad` | 700 万次前向 → 1 次反向 |
| `np.stack([Z0]*P, axis=1)` | `image.repeat(P,1,1,1)` | 完全对应 |
| `sum(forward(...) for p)` | `torch.sum(one_hot * sim)` | 完全对应 |
| `.mean(axis=0).mean(axis=0)[1:]` | `.mean(dim=1).mean(dim=1)[:, 1:]` | 真实版多一个 batch 维，所以是 `dim=1` |
| `minmax(accum)` | `min_max(accum_expl_map)` | 真实版按 `(B, num_prompt)` 分别做 |

下一篇：[[07-三条分支与复现坑位]]
