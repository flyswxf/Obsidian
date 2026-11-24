> [!info] 目标
> 搞清楚 LeGrad 操作的那些张量在物理上是什么：一张图怎么变成 197 个向量，注意力矩阵每一格代表什么，`[CLS]` 为什么在第 0 位，以及 open_clip 与 timm 的存储顺序为什么不同。

## 1. 图像 → patch → token

以 ViT-B/16、输入 $224\times224$ 为例：

**第一步：切块。** `16` 就是 patch 边长，$224/16 = 14$，横 14 块、竖 14 块，共 $14\times14 = 196$ 块。

```text
224×224 图像                切成 14×14 = 196 个 patch
┌──────────────────┐        ┌──┬──┬──┬──┬─  ─┬──┐
│                  │        │0 │1 │2 │3 │ ... │13│   ← 第 0 行
│                  │  ───►  ├──┼──┼──┼──┼─  ─┼──┤
│                  │        │14│15│16│17│ ... │27│   ← 第 1 行
│                  │        ├──┼──┼──┼──┼─  ─┼──┤
│                  │        │  │  │  │  │ ... │  │
└──────────────────┘        └──┴──┴──┴──┴─  ─┴──┘
                             每块 16×16 像素
```

**第二步：每块变成一个向量。** 一个 $16\times16\times3$（RGB）的块有 $16\times16\times3 = 768$ 个数字，代码里用一个卷积核 $16\times16$、步长 $16$ 的 `nn.Conv2d` 一次性完成"切块＋线性投影"，输出 768 维向量。这个向量就叫 **token**，相当于"一句话里的一个词"。

```python
# utils.py 里 vit_dynamic_size_forward 的前几行
x = self.conv1(x)                            # (1, 768, 14, 14)  每块一个 768 维向量
x = x.reshape(x.shape[0], x.shape[1], -1)    # (1, 768, 196)     14×14 拉直成 196
x = x.permute(0, 2, 1)                       # (1, 196, 768)     换轴，token 维提前
```

拉直顺序是**行优先**：第 0 个 token 是左上角，第 13 个是第 0 行最右，第 14 个是第 1 行最左……第 195 个是右下角。这个顺序很重要——最后一步 `reshape(14, 14)` 能还原成图像网格，靠的就是它。

**第三步：前面插入 `[CLS]`。**

```python
x = torch.cat([_expand_token(self.class_embedding, x.shape[0]), x], dim=1)
# (1, 1, 768) 与 (1, 196, 768) 拼接 → (1, 197, 768)
```

`class_embedding` 是一个可学习的 768 维向量，不对应图像上任何位置，纯占位。训练时被要求"汇总全图信息"，最后由它代表整张图去做分类／对比。它永远在**第 0 位**，所以代码里到处出现 `[1:]`（丢掉第 0 项）和 `- 1`（token 数减 1 得 patch 数）。

**第四步：加位置编码。** token 本身不含位置信息（第 5 块和第 100 块的处理方式完全一样），必须额外加一个"我在第几行第几列"的向量：

```python
x = x + self.positional_embedding.to(x.dtype)   # (197, 768) 逐个 token 相加
```

**第五步：进 Transformer。** 之后过 `ln_pre`，再换成 LND 布局送进 12 层：

```python
x = x.permute(1, 0, 2)   # (1, 197, 768) → (197, 1, 768)   NLD → LND
x = self.transformer(x)
```

## 2. 一层 Transformer 做了什么（式 1）

$$\hat{Z}^l = \mathrm{MSA}^l(Z^{l-1}) + Z^{l-1}, \qquad Z^l = \mathrm{MLP}^l(\hat{Z}^l) + \hat{Z}^l$$

拆成人话，每层做两件事，各自带一条"原样加回来"的残差：

```text
输入 Z^{l-1}  (197, 1, 768)
     │
     ├──► LayerNorm ──► 多头自注意力 MSA ──┐
     │                                     ├──► 相加 ──► Ẑ^l   (注意力后、MLP 前)
     └─────────────────────────────────────┘
     │
     ├──► LayerNorm ──► MLP（升维再降维）──┐
     │                                     ├──► 相加 ──► Z^l    (本层最终输出)
     └─────────────────────────────────────┘
```

- **MSA（多头自注意力）**：让 197 个 token 互相交换信息。这是"图像上下文融合"发生的地方。
- **MLP**：对每个 token 单独做一次非线性变换（token 之间不交流）。
- **残差 `+ Z^{l-1}`**：把输入原样加回来，所以每一层的输出都"包含"前面所有层的信息，这也是为什么中间层的表示同样可用。

代码里 hook 存的就是这两个中间量：

```python
# utils.py L65-70
x = q_x + self.ls_1(self.attention(q_x=self.ln_1(q_x), k_x=k_x, v_x=v_x, attn_mask=attn_mask))
self.feat_post_attn = x          # ← Ẑ^l
x = x + self.ls_2(self.mlp(self.ln_2(x)))
self.feat_post_mlp = x           # ← Z^l    LeGrad 实际用的是这个
```

（`ls_1` / `ls_2` 是 LayerScale，一个逐通道的缩放系数，ViT-B/16 里通常是 `nn.Identity()`，可以忽略。）

## 3. 注意力矩阵：$T\times T$ 的"谁在看谁"表

### 3.1 Q / K / V 的比喻

自注意力把每个 token 投影成三样东西，用图书馆检索来类比：

| 记号  | 全称    | 比喻           | 作用                  |
| --- | ----- | ------------ | ------------------- |
| $Q$ | query | "我想找关于猫的书"   | 提问方，注意力的**行**       |
| $K$ | key   | 书脊上的标签"猫科动物" | 被检索方，用来匹配，注意力的**列** |
| $V$ | value | 书的实际内容       | 匹配上之后真正被取走的信息       |

计算过程：

1. 每个 token $i$ 拿自己的 $q_i$ 去和所有 token $j$ 的 $k_j$ 做点积 → 得到"匹配分数"
2. 分数除以 $\sqrt{d_{\text{head}}}$（缩放，防止数值过大）
3. 对每一行做 `softmax` → 变成加起来等于 1 的权重
4. 用这些权重对所有 $v_j$ 加权求和 → token $i$ 的新表示

$$A_{i,j} = \mathrm{softmax}_j\!\left(\frac{q_i \cdot k_j}{\sqrt{d_{\text{head}}}}\right), \qquad \text{输出}_i = \sum_j A_{i,j} v_j$$

**第 3 步的 softmax 是沿行做的**，所以 $A$ 的每一行加起来等于 1：token $i$ 把自己的"100% 注意力"分配给 197 个 token。

### 3.2 一个具体小例子

假设只有 3 个 patch（简化掉 `[CLS]`），某一层某个 head 的注意力矩阵：

```text
              被看的 token (key)
              patch0  patch1  patch2     每行之和
提问方  patch0 [ 0.70   0.20   0.10 ]  →  1.00
(query) patch1 [ 0.05   0.90   0.05 ]  →  1.00
        patch2 [ 0.30   0.30   0.40 ]  →  1.00
```

读法：
- $A_{0,0} = 0.70$：patch0 把 70% 的注意力放在自己身上
- $A_{1,0} = 0.05$：patch1 只花 5% 注意力看 patch0
- **列**加起来没有约束：第 0 列之和 $=0.70+0.05+0.30 = 1.05$，表示"patch0 总共被看了多少"

LeGrad 的第 8、9 步平均掉的是**行**（query），剩下的是**列**（key）。所以最终每个 patch 的重要性分数含义是：**"这个 patch 被大家关注的程度，若拧动一点，目标分数会变多少"**——正因为剩下的是列索引（对应图像位置），才能 reshape 成 $14\times14$ 贴回图上。

### 3.3 多头

$h=12$ 表示有 12 套独立的 $Q/K/V$ 投影，各算各的注意力矩阵。每套的维度是 $768/12 = 64$。12 张矩阵可以关注不同类型的关系（有的看颜色、有的看边缘、有的看空间邻近）。

所以第 $l$ 层的完整注意力图是 $A^l \in \mathbb{R}^{12\times197\times197}$：12 张表，每张 $197\times197$。

### 3.4 代码里存下来的形状

```python
# utils.py L344-383，PyTorch 多头注意力实现的拷贝
attn_output_weights = torch.bmm(q, k.transpose(1, 2))   # (1*12, 197, 197)
attn_output_weights = F.softmax(attn_output_weights, dim=-1)
attn_output_weights = F.dropout(attn_output_weights, p=dropout_p, training=training)
attn_output = torch.bmm(attn_output_weights, v)
...
return attn_output, attn_output_weights    # ← 原版 PyTorch 在这里会把 12 个 head 平均掉，LeGrad 删了
```

注意两点：
1. 存下来的形状是 $(P\cdot h, T, T)$——**batch 和 head 被压在同一维**（$P=1$ 时就是 $(12,197,197)$）。这是 PyTorch 为了用 `bmm` 批量计算的实现细节，所以后面必须 `rearrange('(b h) n m -> b h n m')` 拆开。
2. 存的是 **softmax 之后、dropout 之后**的张量。`model.eval()` 时 dropout 是恒等变换，所以就是纯 softmax 概率。求梯度求的也是这个概率，不是 softmax 前的原始分数。

## 4. LND vs NLC：两套布局，别搞混

| 库 | 布局 | `feat_post_mlp.shape` | token 平均要写 |
| --- | --- | --- | --- |
| open_clip（CLIP / CoCa） | **LND**：token 在前，batch 在中 | $(197, P, 768)$ | `mean(dim=0)` |
| timm（SigLIP） | **NLC**：batch 在前 | $(P, 784, C)$ | `mean(dim=1)` |

`L` = length（token 数），`N` = batch，`C`/`D` = 特征维。open_clip 沿用早期 Transformer 代码的习惯把 token 放最外层，timm 用主流的 batch-first。

这就解释了为什么三个分支里"平均 token"和"取 patch 数"的写法不一样：

```python
# compute_legrad_clip（LND）
intermediate_feat.mean(dim=0)                      # 对 token 平均
num_tokens = blocks_list[-1].feat_post_mlp.shape[0] - 1   # shape[0] 是 token 数，减 1 去 CLS

# compute_legrad_siglip（NLC）
num_tokens = blocks_list[-1].feat_post_mlp.shape[1]       # shape[1] 才是 token 数，SigLIP 无 CLS 所以不减
```

**看到 `dim=` 数字先问一句：这个张量是 LND 还是 NLC？**

## 5. 从 197 个 token 到 1 个"整图向量"

Transformer 出来的是 197 个向量，但和文本比对只需要 1 个向量。有三种做法：

### 5.1 `[CLS]` 法（原模型推理路径）

```python
x = self.ln_post(x)                 # 对全部 197 个 token 做 LayerNorm
pooled, tokens = self._global_pool(x)   # pool_type='tok' → pooled = x[:, 0]，取第 0 个
pooled = pooled @ self.proj         # (1, 768) @ (768, 512) → (1, 512)
```

顺序是"**先 LayerNorm，再取 `[CLS]`**"。

### 5.2 token 平均法（LeGrad 采用，式 3）

$$\bar{z}^l = \frac{1}{n+1}\sum_{i=0}^{n} z_i^l$$

```python
# wrapper.py L155-157
intermediate_feat = self.visual.transformer.resblocks[layer].feat_post_mlp   # (197, P, 768)
intermediate_feat = self.visual.ln_post(intermediate_feat.mean(dim=0)) @ self.visual.proj
intermediate_feat = F.normalize(intermediate_feat, dim=-1)
```

顺序是"**先对 197 个 token 平均，再 LayerNorm**"。

与 5.1 的差别不只是取值方式：LayerNorm 是逐 token 独立归一化的非线性操作，$\mathrm{LN}(\bar{z}) \ne \overline{\mathrm{LN}(z)}$，两者数值上不等价。LeGrad 之所以选平均，是因为式(3) 要让**每一层**都能独立产出一个"该层的预测"，而 `[CLS]` 在浅层还没汇总好信息；平均所有 token 是个对任何层都成立的通用做法。

另外这一步用的 `ln_post` 和 `proj` 是模型原有的（`self.visual.ln_post`、`self.visual.proj`），所以 $\bar z^l$ 被投到了和文本嵌入同一个 512 维空间——这就是论文里那个映射 $C$ 的实现。

### 5.3 注意力池化（SigLIP / CoCa，式 2）

$$\bar{z}_{\mathrm{AttnPool}} = \mathrm{softmax}\!\left(\frac{q_{\mathrm{pool}} \cdot (W_K Z^L)^{\top}}{\sqrt{d}}\right)(W_V Z^L)$$

不用 `[CLS]`，而是**另配一个可学习的 query** $q_{\mathrm{pool}}$ 去"问"所有 patch，本质上是一层只有一行 query 的注意力：

```text
q_pool (1, 768)  ── 提问 ──►  ┌────────────────────────┐
                               │  patch0 patch1 ... patchN │  ← 所有 patch 当 key/value
                               └────────────────────────┘
                                        │
                        注意力权重 (1, N)，加起来等于 1
                                        │
                              加权求和 → 整图向量 (1, 768)
```

对应的注意力图形状是 $(B, h, 1, N)$——中间那个 `1` 就是唯一的 query。所以池化模型的 LeGrad 求梯度对象从"自注意力图"换成了"池化注意力图"，其余流程完全一致：

```python
# wrapper.py L271（SigLIP 分支）
image_relevance = grad.mean(dim=1)[:, 0]   # 平均掉 head，再取第 0 个（也是唯一一个）query 行
```

这里 `[:, 0]` 选的是 **query（行）**，而 CLIP 分支的 `[:, 1:]` 丢的是 **key（列）**，两处下标含义完全不同，别混。SigLIP 没有 `[CLS]`，所以 $N$ 个位置全是 patch，不需要再丢一项。

## 6. 论文符号 ↔ 代码模块名

| 论文 | open_clip 代码 | timm（SigLIP）代码 |
| --- | --- | --- |
| $Z^0$（输入序列） | `vit_dynamic_size_forward` 里 `x` 进 `transformer` 前 | `trunk.patch_embed` 输出 |
| 第 $l$ 层模块 | `visual.transformer.resblocks[l]` | `visual.trunk.blocks[l]` |
| $\hat Z^l$ | `resblocks[l].feat_post_attn` | `blocks[l].feat_post_attn` |
| $Z^l$ | `resblocks[l].feat_post_mlp` | `blocks[l].feat_post_mlp` |
| $A^l$ | `resblocks[l].attn.attention_maps` | `blocks[l].attn.attn_probs` |
| $A_{\mathrm{pool}}$ | `visual.attn_pool.attn.attention_maps`（CoCa） | `visual.trunk.attn_pool.attn_probs` |
| 映射 $C$ | `visual.ln_post` + `visual.proj` | `trunk.attn_pool`（内含投影） |
| 文本嵌入 | `model.encode_text(tokens, normalize=True)` | 同 |
| 层数 $L$ | `len(visual.transformer.resblocks)` = 12 | `len(trunk.blocks)` |
| 头数 $h$ | `attn.num_heads` = 12 | `attn_pool.num_heads` |
| patch 边长 | `visual.patch_size[0]` = 16 | `trunk.patch_embed.patch_size[0]` |

`layer_index=-2` 会被换算成绝对深度：

```python
# wrapper.py L45-46
self.starting_depth = layer_index if layer_index >= 0 else len(self.visual.transformer.resblocks) + layer_index
# layer_index=-2, 共 12 层 → starting_depth = 10 → 只用第 10、11 层（最后两层）
```

下一篇：[[03-梯度与autograd]]
