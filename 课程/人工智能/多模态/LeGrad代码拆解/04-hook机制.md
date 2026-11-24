> [!info] 目标
> 解释 LeGrad 是怎么在**不改 open_clip 一行源码**的前提下，把每一层的注意力矩阵和中间特征"偷"出来的。这部分是全篇最工程的内容，理解了它，`utils.py` 里那几百行看起来像复制粘贴的代码就都有意义了。

## 1. 问题：`forward` 只吐出最终结果

```python
image_embedding = model.encode_image(image)   # (1, 512)
```

想要的东西全在中间：第 $l$ 层的 token 特征 $Z^l$、第 $l$ 层的注意力矩阵 $A^l$。而 open_clip 的 `ResidualAttentionBlock.forward` 只 `return x`，注意力矩阵算完就用掉了，根本没往外传。

三种解决办法：

| 方案 | 做法 | 缺点 | LeGrad 选它吗 |
| --- | --- | --- | --- |
| A. 改源码 | fork open_clip，在需要的地方加 `return` | 要维护一个分支，版本升级就得重新改 | 否 |
| B. `register_forward_hook` | PyTorch 官方机制，能拿到**模块的输入和输出** | 注意力矩阵既不是 `attn` 模块的输入也不是输出（输出是 `attn(q,k,v)[0]`，矩阵被丢在内部），**拿不到** | 否 |
| C. 替换 `forward`（monkey patch） | 把整个函数抄一遍，在中间插一行保存 | 与库版本强耦合 | **是** |

方案 B 的失效是关键：`nn.MultiheadAttention` 内部确实算出了 `attn_output_weights`，但 `need_weights=False` 时直接返回 `None`，即使返回也只是**对 head 求平均后**的结果。想要 12 个 head 各自的原始矩阵，只能进函数体里面拿。所以 LeGrad 选了 C。

## 2. Monkey patch 怎么生效

```python
# wrapper.py L86-92
for layer in range(self.starting_depth, len(self.visual.transformer.resblocks)):
    blk = self.visual.transformer.resblocks[layer]
    blk.attn.forward = types.MethodType(hooked_torch_multi_head_attention_forward, blk.attn)
    blk.forward      = types.MethodType(hooked_resblock_forward, blk)
```

两个要素：

### 2.1 `types.MethodType(func, obj)` 干什么

普通函数的第一个参数不会自动填充：

```python
def f(self, x):
    return self.name + x

class Obj:
    name = "hi"

o = Obj()
o.forward = f          # 直接赋值
o.forward("!!")        # TypeError: f() 缺少参数 self
```

`types.MethodType` 把函数**绑到某个对象上**，之后调用会自动把该对象作为第一个参数传入：

```python
import types
o.forward = types.MethodType(f, o)
o.forward("!!")        # 'hi!!'   self 自动就是 o
```

这就是为什么 `hooked_resblock_forward(self, q_x, ...)` 里能直接用 `self.ln_1`、`self.mlp`——`self` 就是那个被 patch 的 resblock 实例。

### 2.2 为什么改实例属性能覆盖类方法

`nn.Module.__call__` 内部大致是这样：

```python
def _call_impl(self, *args, **kwargs):
    forward_call = self.forward        # ← 调用时才去查 self.forward
    ...
    return forward_call(*args, **kwargs)
```

Python 查找属性时**实例字典优先于类字典**，所以 `blk.forward = ...` 之后，`blk(...)` 就会走新函数。只影响这一个实例，同类的其他层不受影响（这正是需要的：只 hook `starting_depth` 之后的层）。

## 3. 完整调用链：三层嵌套的 patch

`compute_legrad_clip` 跑一次前向时，第 $l$ 层（$l \ge$ `starting_depth`）内部发生的事：

```text
Transformer.forward
  └─► resblocks[l].__call__
        └─► resblocks[l].forward        ← 【patch 1】hooked_resblock_forward
              │  x = q_x + ls_1( self.attention(q_x=ln_1(q_x), ...) )
              │  self.feat_post_attn = x                     ★ 存下 Ẑ^l
              │  x = x + ls_2( mlp(ln_2(x)) )
              │  self.feat_post_mlp = x                      ★ 存下 Z^l
              │  return x
              │
              └─► resblocks[l].attention(...)     ← open_clip 原方法，**没有**被替换
                    │  return self.attn(q_x, k_x, v_x, need_weights=False, attn_mask=...)[0]
                    │                                                       ↑
                    │                                            注意这里传的是 False，且取 [0]
                    │
                    └─► attn.__call__
                          └─► attn.forward        ← 【patch 2】hooked_torch_multi_head_attention_forward
                                │  内部强制 need_weights=True，调 ↓
                                │
                                └─► hooked_torch_func_multi_head_attention_forward   ← 【patch 3】
                                      │  PyTorch F.multi_head_attention_forward 的完整拷贝
                                      │  attn_output_weights = softmax(q @ kᵀ)
                                      │  attn_output_weights = dropout(...)
                                      │  attn_output = bmm(attn_output_weights, v)
                                      │  return attn_output, attn_output_weights     ★ 原始 per-head 矩阵
                                      │
                                │  self.attention_maps = _attn_maps                   ★ 挂到 attn 模块上
                                │  return out, _attn_maps
```

三个 patch 各司其职，配合得非常紧：

| patch | 位置 | 干了什么 |
| --- | --- | --- |
| 1 | `resblocks[l].forward` | 存两个中间特征 `feat_post_attn` / `feat_post_mlp` |
| 2 | `attn.forward` | **把 `need_weights` 硬编码成 `True`**，并把结果挂到 `self.attention_maps` |
| 3 | 内层函数 | **删掉 PyTorch 原本的 head 平均**，返回 per-head 原始矩阵 |

### 3.1 为什么必须硬编码 `need_weights=True`

open_clip 的 `attention()` 方法传的是 `need_weights=False`（推理时不需要注意力图，省算力）。若照实传下去，内层函数会 `return attn_output, None`，`attention_maps` 就成了 `None`。

```python
# utils.py L133-145：外层 hook 无视传进来的 need_weights，一律写死 True
else:
    out, _attn_maps = hooked_torch_func_multi_head_attention_forward(
        query, key, value, ...,
        need_weights=True,          # ← 硬编码，覆盖调用方的 False
        attn_mask=attn_mask,
    )
    self.attention_maps = _attn_maps
    return out, _attn_maps
```

### 3.2 为什么返回 tuple 还能正常工作

`attention()` 结尾有个 `[0]`：

```python
return self.attn(q_x, k_x, v_x, need_weights=False, attn_mask=attn_mask)[0]
```

原版 `nn.MultiheadAttention` 本来就返回 `(attn_output, attn_output_weights)`，`[0]` 取输出。hook 版也返回 tuple，`[0]` 同样取到输出——**上层代码完全不用改**，多出来的注意力矩阵则被顺手挂在 `attn.attention_maps` 上。这是这个设计能"抄一遍就插上"的原因。

### 3.3 为什么必须删掉 head 平均

PyTorch 原版在 `need_weights=True` 时会做：

```python
# 原版行为（LeGrad 里被注释掉了，utils.py L379-383）
attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
return attn_output, attn_output_weights.sum(dim=1) / num_heads     # 12 个 head 平均成 1 个
```

LeGrad 需要的是**每个 head 各自的矩阵**，因为式(5) 要求"先对每个 head 单独 ReLU，再对 head 求平均"。提前平均会让正负梯度抵消，ReLU 就白做了（数值演示见 [[03-梯度与autograd]] 第 3 节）。所以直接：

```python
return attn_output, attn_output_weights     # 保持 (bsz*num_heads, tgt_len, src_len)
```

代价是形状里 batch 和 head 揉在一起，后续必须 `rearrange('(b h) n m -> b h n m')` 拆开。

## 4. 四个 hook 函数逐个看

### 4.1 `hooked_resblock_forward`（utils.py L55-71）— open_clip 分支

open_clip 2.23 `ResidualAttentionBlock.forward` 的**逐字拷贝** + 2 行埋点：

```python
def hooked_resblock_forward(self, q_x, k_x=None, v_x=None, attn_mask=None):
    k_x = self.ln_1_kv(k_x) if hasattr(self, "ln_1_kv") and k_x is not None else None
    v_x = self.ln_1_kv(v_x) if hasattr(self, "ln_1_kv") and v_x is not None else None

    x = q_x + self.ls_1(self.attention(q_x=self.ln_1(q_x), k_x=k_x, v_x=v_x, attn_mask=attn_mask))
    self.feat_post_attn = x          # ★ 新增：Ẑ^l，形状 (197, P, 768)  LND
    x = x + self.ls_2(self.mlp(self.ln_2(x)))
    self.feat_post_mlp = x           # ★ 新增：Z^l，形状 (197, P, 768)  LND
    return x
```

`k_x` / `v_x` / `ln_1_kv` 那两行是给 CoCa 的 cross-attention 用的（key/value 可以来自文本），普通 CLIP 里都是 `None`。

`feat_post_attn` 存了但三条分支都没用到——LeGrad 的"中间 token 激活"取的是 MLP 之后的 $Z^l$。

### 4.2 `hooked_torch_multi_head_attention_forward`（utils.py L77-145）— 外层

只做两件事：强制 `need_weights=True`，以及 `self.attention_maps = _attn_maps`。它有两条分支：

```python
if not self._qkv_same_embed_dim:
    # Q/K/V 维度不同 → 用三套独立权重（use_separate_proj_weight=True）
else:
    # Q/K/V 维度相同 → 用一套合并权重
```

- **普通 CLIP 的自注意力**：`nn.MultiheadAttention(d, h)`，Q/K/V 同维 → 走 `else`
- **CoCa 的 `AttentionalPooler`**：`nn.MultiheadAttention(d_model, n_head, kdim=context_dim, vdim=context_dim)`，key/value 来自图像特征、query 来自可学习向量，维度可能不同 → `_qkv_same_embed_dim=False`，走 `if` 分支

这就是这个函数为什么长得啰嗦：要同时兼容两种 `nn.MultiheadAttention` 配置。

### 4.3 `hooked_torch_func_multi_head_attention_forward`（utils.py L147-385）— 内层

PyTorch `torch/nn/functional.py` 里 `multi_head_attention_forward` 的完整拷贝，改动只有末尾那处 head 平均。约 240 行里绝大部分（QKV 投影、mask 处理、`bias_k`、`add_zero_attn`、`static_k/v`）LeGrad 用不到，但为了让函数能独立运行，全抄了过来。

**读这个文件的正确方式：只看 L344-385。**

```python
attn_output_weights = torch.bmm(q, k.transpose(1, 2))    # (P*h, 197, 197) 原始分数
assert list(attn_output_weights.size()) == [bsz * num_heads, tgt_len, src_len]
if attn_mask is not None: ...                            # ViT 里是 None
if key_padding_mask is not None: ...                     # ViT 里是 None
attn_output_weights = F.softmax(attn_output_weights, dim=-1)   # ← 沿最后一维（列）归一化
attn_output_weights = F.dropout(attn_output_weights, p=dropout_p, training=training)
attn_output = torch.bmm(attn_output_weights, v)          # ← 保存的张量正是参与运算的那个
...
return attn_output, attn_output_weights
```

倒数第二行很重要：保存下来的 `attn_output_weights` **就是被 `bmm` 用掉的那个张量对象**，它在计算图里，所以可以对它求梯度。如果这里存的是 `.detach()` 或 `.clone()` 出来的副本，梯度就断了。

另外它位于 `dropout` **之后**——`model.eval()` 时 dropout 是恒等变换所以无所谓，但若忘了 `eval()`，注意力图会带随机丢弃，热图每次都不一样。

> [!warning] 一处潜在隐患
> L171-183 的 `if not torch.jit.is_scripting()` 分支里引用了 `multi_head_attention_forward` 和 `pad`，这两个名字在 `utils.py` 里**没有导入**。只有当输入是 Tensor 子类（触发 `F.has_torch_function`）时才会走到，正常路径不会碰到，但一旦碰到就是 `NameError`。

### 4.4 `hooked_resblock_timm_forward` + `hooked_attentional_pooler_timm_forward`（utils.py L389-432）— SigLIP 分支

timm 的 block 结构不同（用 `norm1`/`drop_path1`/`ls1`），要单独抄一份：

```python
def hooked_resblock_timm_forward(self, x):
    x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    self.feat_post_attn = x                       # (P, N, C)  NLC 布局
    x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    self.feat_post_mlp = x                        # ★
    return x
```

SigLIP 没有 `[CLS]`，也不用每层自注意力做解释，而是把每层特征送进**注意力池化头**，对池化注意力求梯度：

```python
def hooked_attentional_pooler_timm_forward(self, x):
    B, N, C = x.shape
    q_latent = self.latent.expand(B, -1, -1)                    # (B, 1, C) 可学习 query
    q = self.q(q_latent).reshape(B, 1, num_heads, head_dim).transpose(1, 2)
    kv = self.kv(x).reshape(B, N, 2, num_heads, head_dim).permute(2, 0, 3, 1, 4)
    k, v = kv.unbind(0)
    q, k = self.q_norm(q), self.k_norm(k)
    attn = (q * self.scale) @ k.transpose(-2, -1)               # (B, heads, 1, N)
    attn = attn.softmax(dim=-1)
    x = attn @ v
    self.attn_probs = attn                                      # ★ 池化注意力图
    ...
```

`self.latent` 就是式(2) 里的 $q_{\mathrm{pool}}$；`latent_len=1` 所以中间那一维长度是 1。

### 4.5 `hooked_attention_forward`（utils.py L24-50）— 死代码

这个函数是给 open_clip 老版本 `Attention` 类（`forward(self, x, x_k, x_v, ...)`）用的，`wrapper.py` L12-18 导入了它，但**全文没有任何地方调用**。属于版本演进留下的残留，读代码时可以直接跳过。

## 5. hook 的三个隐含性质

### 5.1 存的是"最后一次前向"的结果

```python
blk.feat_post_mlp = x
blk.attn.attention_maps = attn
```

都是普通的实例属性赋值，**每次前向都会覆盖上一次**。这带来两个后果：

- `compute_legrad_*` 里必须先 `encode_image(image)`，紧接着在同一个循环里把各层的缓存读出来用（`wrapper.py` L152-158 先把所有层的 `feat_post_mlp` 收进 `image_features_list`，就是为了防止后面被覆盖）
- 同一进程里交错调用两次推理，中间结果会串

### 5.2 显存被这些属性吊住

`feat_post_mlp` 形状 $(197, P, 768)$、`attention_maps` 形状 $(P\cdot12, 197, 785)$，都挂在模块上，**只要模块活着就不会被回收**，而且它们还连着计算图（因为 `retain_graph=True`）。批量评测时每张图跑完，这些属性会被下一次前向覆盖，所以不会无限增长；但如果想彻底释放，需要显式 `del`。

### 5.3 原模型被就地修改

`LeWrapper.__init__` 里：

```python
for attr in dir(model):
    if not attr.startswith('__'):
        setattr(self, attr, getattr(model, attr))
```

`self.visual` 与原 `model.visual` 是**同一个对象**（Python 的赋值不复制对象）。所以：

- hook 是打在原模型的子模块上的，原 `model` 也被改了
- `requires_grad` 的开关同样作用在原模型的参数上

README 说的"不影响原模型"只对**前向数值**成立（输出完全一致）；如果之后还想用原模型做训练或对注意力求别的梯度，前若干层的 `requires_grad=False` 会造成困扰。安全的做法是给评测单独建一个模型实例。

## 6. 版本耦合：为什么 `requirements.txt` 钉死了上限

```text
open_clip_torch<=2.23.0
```

hook 是逐字拷贝的函数体，一旦被 hook 的库改了内部实现，就会出现两类问题：

| 变化 | 后果 |
| --- | --- |
| `ResidualAttentionBlock.forward` 换了属性名（如 `self.attn` → `self.attention`） | `AttributeError` |
| `forward` 签名变了（如 `q_x` → `x`） | `TypeError` |
| `F.multi_head_attention_forward` 内部实现调整 | hook 版与真实前向数值不一致 → 热图对应不上模型的真实计算 |

第三类最隐蔽：不报错，但解释的不再是模型真正在做的事。升级 open_clip / timm / torch 前，务必先跑一次 sanity check（同一张图同一 prompt，比对升级前后热图是否一致）。

`vit_dynamic_size_forward`（utils.py L436-490）也是同类拷贝——open_clip 的 `VisionTransformer.forward` 抄一遍，中间插入位置编码重采样，以支持任意输入分辨率：

```python
if x.shape[1] != self.positional_embedding.shape[1]:
    self.positional_embedding.data = resample_abs_pos_embed(
        self.positional_embedding.unsqueeze(0),
        new_size=[grid_h, grid_w], num_prefix_tokens=1,
        interpolation='bicubic', antialias=True)
```

注意它是 `.data = ` **原地替换**，不是临时变量。后果见 [[07-三条分支与复现坑位]]。

下一篇：[[05-玩具例子手算LeGrad]]
