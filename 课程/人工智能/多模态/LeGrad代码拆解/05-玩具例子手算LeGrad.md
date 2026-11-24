> [!info] 目标
> 用一个 **4 个 patch、1~2 个 head、随机权重**的迷你模型，把 LeGrad 全流程算一遍。梯度用"加一点点、看差多少"的**有限差分**手算，不需要 PyTorch，本地 numpy 就能跑（已实测，输出见下）。
> 读懂这一篇，再回头看 [[06-逐行拆解compute_legrad_clip|真实代码]] 就只是把 5 换成 197、把 1 换成 12 而已。

## 1. 迷你模型设定

| 量 | 玩具值 | 真实值（ViT-B/16, 224） | 含义 |
| --- | --- | --- | --- |
| $T$ | 5 | 197 | token 数 |
| $n$ | 4 | 196 | patch 数 |
| 网格 | $2\times2$ | $14\times14$ | patch 摆放方式 |
| $d$ | 4 | 768 | token 特征维 |
| $d_t$ | 3 | 512 | 图文共享空间维度 |
| $h$ | 1 → 2 | 12 | 注意力头数 |
| $P$ | 1 → 2 | prompt 个数 | 图像副本数 |
| $L$ | 2 | 12 | 层数 |

### 公共代码（Part A~C 都要用）

```python
import numpy as np

rng = np.random.default_rng(0)
eps = 1e-6                # 有限差分的扰动量

T = 5          # token 数 = 1 个 [CLS] + 4 个 patch
n = T - 1      # patch 数 = 4  -> 2x2 网格
d = 4          # token 特征维
dt = 3         # 图文共享空间维度


def softmax_rows(x):
    """对最后一维做 softmax，使每一行加起来等于 1。"""
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def minmax(m):
    """真实代码 utils.min_max 没有分母保护，热图全相等时会得到 nan；这里加 1e-12 便于演示。"""
    return (m - m.min()) / (m.max() - m.min() + 1e-12)


def upsample(m, k):
    """最近邻放大，模拟 F.interpolate(mode='bilinear')。"""
    return np.repeat(np.repeat(m, k, axis=0), k, axis=1)


# ---------- 假权重（相当于"训练好的模型"）----------
Wv = rng.normal(size=(d, d)) * 0.5        # value 投影
Wm = rng.normal(size=(d, d)) * 0.5        # MLP 的简化版
Wproj = rng.normal(size=(d, dt)) * 0.5    # 投到图文共享空间，相当于 visual.proj


def forward(A, Z0, tv):
    """单份图像副本 + 单个 prompt -> 标量 s。对应式(1) 一层 + 式(3) 聚合。"""
    V = Z0 @ Wv                          # (T, d)                     value
    attn_out = (A @ V).mean(axis=0)      # (h,T,T)@(T,d)->(h,T,d)->(T,d)  多头合并
    x = Z0 + attn_out                    # 式(1) 上半：MSA + 残差  -> Ẑ^l
    x = x + np.tanh(x @ Wm)              # 式(1) 下半：MLP + 残差  -> Z^l
    z_bar = x.mean(axis=0)               # (d,)   式(3)：对 T 个 token 平均
    img = z_bar @ Wproj                  # (dt,)  映射 C
    img = img / np.linalg.norm(img)      # 归一化 -> 单位向量
    return img @ tv                      # 式(3)：s = 图像向量 · 文本向量
```

三点说明：

- **多头合并被简化成"对 head 求平均"**。真实模型是 concat 后过 `out_proj`，但那样会让公式变长，而"平均"已足以演示 head 维是怎么被吃掉的。
- `np.tanh(x @ Wm)` 充当 MLP 的非线性。**这一项不能省**：若没有它，$s$ 对 $A$ 的依赖是纯线性的，梯度矩阵每行会完全相同，"对 query 行求平均"就看不出意义（下面第 2 节会解释）。
- 权重全是随机的，所以热图没有语义，只看**数值与形状的变化过程**。

## 2. Part A：手算 $\partial s/\partial A$

```python
h = 1
text = rng.normal(size=dt)
text = text / np.linalg.norm(text)          # 单位化的文本向量

Z0 = rng.normal(size=(T, d))                # 5 个 token，每个 4 维
A = softmax_rows(rng.normal(size=(h, T, T)))   # (1, 5, 5)，每行和为 1

s0 = forward(A, Z0, text)
print("A.shape =", A.shape, "   s =", round(s0, 6))
print(np.round(A[0], 4))
print("每行和 =", np.round(A[0].sum(axis=1), 6))

# ---------- 有限差分：逐格扰动，看 s 变多少 ----------
g = np.zeros_like(A)
for hh in range(h):
    for i in range(T):
        for j in range(T):
            A2 = A.copy()
            A2[hh, i, j] += eps                      # 只动一格
            g[hh, i, j] = (forward(A2, Z0, text) - s0) / eps
print("grad.shape =", g.shape)
print(np.round(g[0], 5))
print("负梯度个数 =", int((g < 0).sum()), "/", g.size)
```

实测输出：

```text
A.shape = (1, 5, 5)    s = 0.035453
[[0.4268 0.2147 0.0126 0.1201 0.2258]
 [0.2666 0.0527 0.6042 0.0261 0.0504]
 [0.1999 0.0824 0.5813 0.0948 0.0417]
 [0.138  0.0676 0.0561 0.3782 0.36  ]
 [0.2415 0.0311 0.3583 0.0496 0.3195]]
每行和 = [1. 1. 1. 1. 1.]

grad.shape = (1, 5, 5)
[[-0.67006  0.15156 -0.2242   0.28669  0.11666]
 [-0.12219  0.06149 -0.10043  0.1633   0.17174]
 [-0.3345   0.13964 -0.08472  0.13627  0.07252]
 [-0.33958  0.12373 -0.10994  0.16598  0.1014 ]
 [-0.52545  0.09977 -0.23005  0.3009   0.17819]]
负梯度个数 = 10 / 25
```

### 观察 1：梯度和 $A$ 形状完全一样

$A$ 是 $(1,5,5)$，梯度也是 $(1,5,5)$。25 个格子对应 25 个偏导数。这就是式(4)。

### 观察 2：有正有负，且负数占了 40%

`grad[0,0,0] = -0.67006` 的读法：把 $A_{0,0}$（`[CLS]` 看自己）从 0.4268 调大一点点，$s$ 会**下降** 0.67 倍的比例。这条注意力连接在拆台。

10 个负值如果不丢掉，会和正值互相抵消。这就是式(5) 里 ReLU 的必要性。

### 观察 3：`[CLS]` 那一列全是负的

看第 0 列：$-0.670,\ -0.122,\ -0.335,\ -0.340,\ -0.525$，全负。也就是说在这个玩具模型里，"任何 token 去关注 `[CLS]`"都在降低目标分数。ReLU 之后这一列全变 0，随后被 `[:, 1:]` 整列丢弃——**丢 `[CLS]` 列不只是为了对齐图像网格，也顺带丢掉了一批纯负值**。

### 观察 4：为什么梯度每行不同

因为有 `np.tanh(x @ Wm)` 这个非线性。若去掉它，$s$ 对 $A$ 的依赖退化成：

$$s \propto \Big(\tfrac{1}{T}\sum_i \sum_j A_{ij}v_j\Big)\cdot(\cdots)$$

只依赖 $A$ 的**列和**，与行索引 $i$ 无关，于是梯度每一行完全相同，"对 query 行求平均"就成了无意义的操作。真实 ViT 有 12 层 MLP 非线性，所以行与行之间差异显著——这也是为什么式(5) 要对行平均而不是随便取一行。

## 3. Part A 续：从梯度到热图（式 5、6）

```python
gp = np.maximum(g, 0)                        # 式(5)：ReLU，等价于 torch.clamp(min=0.)
print(np.round(gp[0], 5))

E = gp.mean(axis=0).mean(axis=0)[1:]         # 式(5)(6)
print("E_hat.shape =", E.shape, np.round(E, 6))

heat = minmax(E.reshape(2, 2))               # 式(6)(9)
print(np.round(heat, 4))

# ---------- 对照：raw attention ----------
raw = A[0].mean(axis=0)[1:]                  # 完全相同的聚合方式，只是把梯度换成注意力值本身
print(np.round(minmax(raw.reshape(2, 2)), 4))
```

实测输出：

```text
ReLU 之后:
[[0.      0.15156 0.      0.28669 0.11666]
 [0.      0.06149 0.      0.1633  0.17174]
 [0.      0.13964 0.      0.13627 0.07252]
 [0.      0.12373 0.      0.16598 0.1014 ]
 [0.      0.09977 0.      0.3009  0.17819]]

E_hat.shape = (4,) [0.115238 0.       0.210629 0.128103]

LeGrad 单层热图 (2x2):
[[0.5471 0.    ]
 [1.     0.6082]]

对照：raw attention 用完全相同的聚合方式:
[[0.     1.    ]
 [0.1892 0.4716]]
```

### 逐步核对形状

| 代码 | 形状 | 对应 | 消掉了哪一维 |
| --- | --- | --- | --- |
| `g` | $(1,5,5)$ | 式(4) | — |
| `np.maximum(g, 0)` | $(1,5,5)$ | 式(5) $(\cdot)^+$ | 无 |
| `.mean(axis=0)` 第 1 次 | $(5,5)$ | $\frac{1}{h}\sum_h$ | head 维 |
| `.mean(axis=0)` 第 2 次 | $(5,)$ | $\frac{1}{T}\sum_i$ | query 行维 |
| `[1:]` | $(4,)$ | 下标 $1:$ | 丢掉 `[CLS]` **列** |
| `.reshape(2,2)` | $(2,2)$ | 式(6) reshape | — |
| `minmax` | $(2,2)$ | 式(9) | — |

**注意剩下的 4 个数索引的是"列"（key token）**，也就是 patch0~patch3 这 4 个图像位置。所以 `reshape(2,2)` 才有意义：

```text
E_hat = [0.115238, 0., 0.210629, 0.128103]
           ↓ patch0  ↓patch1  ↓patch2   ↓patch3

reshape(2,2) →  ┌ 0.115  0.000 ┐   第 0 行 = patch0, patch1（图像上半）
                └ 0.211  0.128 ┘   第 1 行 = patch2, patch3（图像下半）
```

### LeGrad 与 raw attention 几乎完全相反

```text
LeGrad:        raw attention:
0.5471  0.       0.      1.
1.      0.6082   0.1892  0.4716
```

patch1（右上）在注意力里被看得很多（$A_{0,1}=0.2147$，$A_{2,1}$ 等也不小），但对目标分数的敏感度经过 ReLU 后是 0；patch2（左下）注意力值平平，敏感度却最高。**这就是原文说"注意力值大 ≠ 对预测有贡献"的数值证据**，也是 LeGrad 相对 rollout / raw attention 的核心区别。

## 4. Part B：多层累加（式 8、9）

```python
accum = np.zeros((4, 4))
for layer in range(2):
    A_l = softmax_rows(rng.normal(size=(h, T, T)))    # 每层有自己的注意力图
    s_l = forward(A_l, Z0, text)                      # 每层用自己的 s^l
    g_l = np.zeros_like(A_l)
    for hh in range(h):
        for i in range(T):
            for j in range(T):
                A2 = A_l.copy()
                A2[hh, i, j] += eps
                g_l[hh, i, j] = (forward(A2, Z0, text) - s_l) / eps
    E_l = np.maximum(g_l, 0).mean(axis=0).mean(axis=0)[1:]   # 原始值，先不归一化
    print("第", layer, "层 E_l =", np.round(E_l, 6))
    accum = accum + upsample(E_l.reshape(2, 2), 2)           # 每层先放大再累加

final = minmax(accum)                                        # 累加完才做一次 min-max
print(np.round(final, 4))
```

实测输出：

```text
第 0 层 E_l = [0.116587 0.017442 0.       0.      ]
第 1 层 E_l = [0.143376 0.00178  0.049412 0.023503]

累加后 (4x4):
[[1.     1.     0.     0.    ]
 [1.     1.     0.     0.    ]
 [0.1254 0.1254 0.0178 0.0178]
 [0.1254 0.1254 0.0178 0.0178]]
```

三个要点：

1. **每层独立产图**：每层用自己的 $A^l$ 和自己的 $s^l$，互不影响。这正是原文"中间 token 激活"的含义。
2. **累加的是未归一化的原始值**：两层 `E_l` 的量级接近（都是 0.0x~0.1x），直接相加不会被某一层淹没。原文强调"梯度信号跨层相加无需额外归一化"就是这个意思——若换成注意力值或激活值，各层量纲差好几个数量级，直接加就没意义了。
3. **min-max 只在最后做一次**：与真实代码一致（`accum_expl_map += expl_map` 在循环内，`min_max` 在循环外）。

## 5. Part C：2 个 head、2 个 prompt（`repeat` + 对角线的真相）

```python
h = 2
P = 2
textb = np.stack([text, -text])           # 两个语义相反的 prompt，便于看出热图应互补
Z0b = np.stack([Z0] * P, axis=1)          # (T, P, d)  ← image.repeat(P,1,1,1) 的效果
print("Z0b.shape =", Z0b.shape)

# PyTorch 存下来的注意力图形状就是 (P*h, T, T)
Ab = softmax_rows(rng.normal(size=(P, h, T, T))).reshape(P * h, T, T)
print("Ab.shape  =", Ab.shape)


def forward_batch(Ab, Z0b, textb):
    """一次前向算出所有副本；返回 trace(sim) = 对角线之和。"""
    A4 = Ab.reshape(P, h, T, T)           # rearrange('(b h) n m -> b h n m')
    total = 0.0
    for p in range(P):
        total += forward(A4[p], Z0b[:, p, :], textb[p])
    return total


sb = forward_batch(Ab, Z0b, textb)
print("s = Σ_p s_p =", round(sb, 6))

gb = np.zeros_like(Ab)
for idx in range(P * h):
    for i in range(T):
        for j in range(T):
            A2 = Ab.copy()
            A2[idx, i, j] += eps
            gb[idx, i, j] = (forward_batch(A2, Z0b, textb) - sb) / eps

g4 = gb.reshape(P, h, T, T)               # 拆开 batch 与 head
print("g4.shape =", g4.shape)

E_b = np.maximum(g4, 0).mean(axis=1).mean(axis=1)[:, 1:]
print("E_b.shape =", E_b.shape)
print(np.round(E_b, 6))
```

实测输出：

```text
Z0b.shape = (5, 2, 4)  (T, P, d)  LND 布局
Ab.shape  = (4, 5, 5)  (P*h, T, T)  batch 与 head 被压在同一维
s = Σ_p s_p = -0.256101
g4.shape = (2, 2, 5, 5)  (P, h, T, T)
E_b.shape = (2, 4)  (P, n)
[[0.045152 0.       0.112461 0.077256]
 [0.       0.       0.059893 0.065395]]
```

### 关键：`.mean(axis=1)` 两次的含义**变了**

| | Part A（$h=1$, $P=1$） | Part C（$h=2$, $P=2$） |
| --- | --- | --- |
| 起点形状 | $(1,5,5)$ = $(h,T,T)$ | $(2,2,5,5)$ = $(P,h,T,T)$ |
| 第 1 次 `mean(axis=…)` | 消 head | 消 **head**（axis=1） |
| 第 2 次 `mean(axis=…)` | 消 query 行 | 消 **query 行**（此时又回到 axis=1） |
| 结果 | $(5,)\to(4,)$ | $(2,5)\to(2,4)$ |

真实代码里因为多了 batch 维，起点是 $(P,h,T,T)$，所以两次都写 `dim=1`：第一次消 $h$，形状变 $(P,T,T)$，第二次消 $T$（行），形状变 $(P,T)$。**同一个 `dim=1` 在两次调用里指的不是同一件事**，这是全篇最容易看错的一行。

### 验证"各副本互不相干"

这是 CLIP 分支能用一次 backward 出 $P$ 张热图的全部理由。用有限差分可以直接验证：

```python
A_only = Ab[:h].reshape(h, T, T)                    # 只取第 0 份副本的 2 个 head
s_only = forward(A_only, Z0, textb[0])              # 只用 prompt 0
g_only = np.zeros_like(A_only)
for hh in range(h):
    for i in range(T):
        for j in range(T):
            A2 = A_only.copy()
            A2[hh, i, j] += eps
            g_only[hh, i, j] = (forward(A2, Z0, textb[0]) - s_only) / eps

print(np.abs(g_only - g4[0]).max())      # 与批量结果的第 0 份对比
```

实测输出：

```text
单独算 prompt0 的梯度 vs 批量算的第 0 份，最大差异 = 2.7755575615628914e-11
```

差异在浮点误差量级（$10^{-11}$）。说明把 $s$ 写成 $\sum_p s_p$ 再一次性求导，与"每个 prompt 单独求导"完全等价——因为第 $p$ 份副本的注意力图只出现在 $s_p$ 里，$p\ne q$ 时 $\partial s_p/\partial A_q = 0$。

真实代码里对应的就是这两行：

```python
one_hot = F.one_hot(torch.arange(0, num_prompts)).float()   # P×P 单位阵
one_hot = torch.sum(one_hot * sim)                          # trace(sim) = Σ_p sim[p,p]
```

### 每个 prompt 各自归一化

```python
heat_b = np.stack([minmax(E_b[p].reshape(2, 2)) for p in range(P)])
```

```text
prompt 0:               prompt 1:
[[0.4015 0.    ]        [[0.     0.    ]
 [1.     0.687 ]]        [0.9159 1.    ]]
```

两个方向相反的 prompt 得到了两张不同的热图，验证了 prompt 条件性。**注意每张图各自 min-max**，所以 `prompt 0` 的最大值和 `prompt 1` 的最大值都等于 1，两者数值不可直接比大小——这正是 `utils.min_max` 按 `(B, num_prompt)` 分别归一化的后果。

## 6. torch 对照版（有 GPU/torch 的环境可跑）

把有限差分那个双重循环换成一行 `autograd.grad`，其余完全一致：

```python
import torch
import torch.nn.functional as F

torch.manual_seed(0)
T, d, dt, h = 5, 4, 3, 2

Wv = torch.randn(d, d) * 0.5
Wm = torch.randn(d, d) * 0.5
Wproj = torch.randn(d, dt) * 0.5
text = F.normalize(torch.randn(dt), dim=-1)
Z0 = torch.randn(T, d)

scores = torch.randn(h, T, T, requires_grad=True)   # ← 叶子张量，必须开 requires_grad
A = scores.softmax(dim=-1)                          # ← 非叶子张量，就是式(4) 的求导对象

V = Z0 @ Wv
attn_out = (A @ V).mean(dim=0)
x = Z0 + attn_out
x = x + torch.tanh(x @ Wm)
z_bar = x.mean(dim=0)
img = F.normalize(z_bar @ Wproj, dim=-1)
s = img @ text

grad = torch.autograd.grad(s, A)[0]      # ← 等价于 numpy 版那个 T*T*h 次的双重循环
print(grad.shape)                        # torch.Size([2, 5, 5])

E = grad.clamp(min=0.).mean(dim=0).mean(dim=0)[1:]
print(E.reshape(2, 2))
```

对照关系：

| numpy 玩具版 | torch 真实版 |
| --- | --- |
| `for` 双重循环逐格扰动 | `torch.autograd.grad(s, A)` |
| `np.maximum(g, 0)` | `torch.clamp(grad, min=0.)` |
| `.mean(axis=0).mean(axis=0)[1:]` | `.mean(dim=1).mean(dim=1)[:, 1:]` |
| `E.reshape(2, 2)` | `rearrange(E, 'b (w h) -> 1 b w h')` |
| `upsample(m, k)`（最近邻） | `F.interpolate(x, scale_factor=16, mode='bilinear')` |
| `minmax(accum)` | `utils.min_max(accum_expl_map)` |
| `np.stack([Z0]*P, axis=1)` | `image.repeat(P, 1, 1, 1)` |
| `sum(forward(...) for p in range(P))` | `torch.sum(one_hot * sim)` |

`scores` 必须 `requires_grad=True` 而 `A` 不用设置——因为 `A` 是由 `scores` 算出来的**非叶子张量**，它在计算图里、可以被 `autograd.grad` 指定为求导目标。这与真实代码完全一致：`attention_maps` 是 softmax 的输出，从来没有人给它设过 `requires_grad`。

## 7. 玩具模型简化了什么

对照真实实现，以下 5 处是简化，读 [[06-逐行拆解compute_legrad_clip]] 时要补回来：

| 简化 | 真实情况 |
| --- | --- |
| 多头用 `mean(axis=0)` 合并 | concat 回 $d$ 维后过 `out_proj` 线性层 |
| MLP 写成 `tanh(x @ Wm)` | 两层线性 + GELU，中间升到 $4d$ 维 |
| 只有 1 层，$A$ 直接是输入 | 12 层串联，第 $l$ 层的 $A^l$ 要影响后面所有层 |
| `mean(axis=0)` 直接得整图向量 | 还要过 `ln_post` 和 `@ proj`（玩具版把它拆成了 `Wproj`） |
| 有限差分逐格扰动，$O(T^2)$ 次前向 | autograd 一次反向搞定，$O(1)$ 次 |

有限差分只是为了"看得见梯度是怎么来的"；真实实现绝不会这么算——$785\times785\times12$ 格要扰动 700 多万次。

## 8. 自查清单

读完这一篇应该能回答：

- [ ] 为什么 $\nabla A$ 的形状和 $A$ 一样？
- [ ] `grad[0,0,0] = -0.67` 这句话怎么用中文读出来？
- [ ] 为什么 ReLU 必须在"对 head 平均"之前？
- [ ] 两次 `mean(dim=1)` 分别消掉了哪一维？为什么第二次还是 `dim=1`？
- [ ] `[:, 1:]` 丢的是行还是列？为什么丢的是这一项？
- [ ] 为什么 $E$ 的索引对应图像位置，而 $A$ 的行索引不对应？
- [ ] `image.repeat(P,1,1,1)` + `trace(sim)` 凭什么能一次出 $P$ 张热图？
- [ ] 为什么各层的 $E^l$ 可以直接相加而不需要分别归一化？

答不上来的回到对应小节，或直接查 [[01-张量形状与常用API]]。

下一篇：[[06-逐行拆解compute_legrad_clip]]
