> [!info] 目标
> 把 $\nabla A^l = \partial s/\partial A^l$ 这一行公式彻底讲透：梯度是什么、为什么它能当解释信号、PyTorch 那几个参数（`requires_grad`、`retain_graph`、`create_graph`）各管什么。全部配 numpy 可跑的小例子。

## 1. 导数就是"斜率"

一个输入一个输出时：

```python
def f(x):
    return x * x          # f(x) = x²

x = 3.0
eps = 0.0001
(f(x + eps) - f(x)) / eps      # 6.0001  ≈ 2x = 6
```

含义：在 $x=3$ 附近，$x$ **每增加 1，$f$ 大约增加 6**。这个"每增加 1 会变多少"就是导数。

`eps` 叫扰动量。用"加一点点、看差多少"来逼近导数，叫**有限差分**。这是理解梯度的最实在的办法，[[05-玩具例子手算LeGrad]] 会用它把整个 LeGrad 手算一遍。

## 2. 多个输入 → 偏导数 → 梯度张量

现在输入不是一个数，而是一整张表 $A$（形状 $12\times197\times197$），输出还是一个数 $s$（目标分数）。

**偏导数** $\dfrac{\partial s}{\partial A_{h,i,j}}$ 的意思：只把 $A$ 的第 $(h,i,j)$ 格拧动一点点，其他格全部不动，看 $s$ 变多少。

$A$ 有多少格，就有多少个偏导数。把它们按同样的位置摆好，就得到一个**和 $A 形状完全相同**的张量：

$$\nabla A = \frac{\partial s}{\partial A} \in \mathbb{R}^{12\times197\times197}$$

这就是式(4)。**"对一个矩阵求梯度，得到的是同形状的矩阵"**——这是最容易卡住的一点，因为习惯上"导数是一个数"。

```python
# 用 numpy 有限差分验证：s 是 A 的加权和，梯度就应该等于权重
import numpy as np

W = np.array([[1.0, 5.0],
              [2.0, 0.0]])            # 固定权重

def s_of(A):
    return (A * W).sum()              # s = Σ A_ij * W_ij

A = np.array([[0.3, 0.1],
              [0.4, 0.2]])
eps = 1e-5

grad = np.zeros_like(A)
for i in range(2):
    for j in range(2):
        A2 = A.copy()
        A2[i, j] += eps               # 只动一格
        grad[i, j] = (s_of(A2) - s_of(A)) / eps

print(grad)      # [[1. 5.]
                 #  [2. 0.]]   ← 正好等于 W
```

结论：$s=\sum_{ij} A_{ij}W_{ij}$ 时，$\partial s/\partial A_{ij} = W_{ij}$。梯度的形状和 $A$ 一样，每格告诉你"这格对 $s$ 的话语权有多大"。

## 3. 正负号的含义

| 梯度值 | 含义 | LeGrad 怎么处理 |
| --- | --- | --- |
| 大正数 | 把这格调大 → 目标分数明显升高 → 这条注意力连接在**帮**模型认定目标 | 保留，数值大 = 重要 |
| 接近 0 | 动它没用 → 与当前判断无关 | 保留但数值小，热图上偏暗 |
| 负数 | 把这格调大 → 目标分数**下降** → 这条连接在**拆台** | `clamp(min=0)` 置零丢弃 |

`torch.clamp(grad, min=0.)` 就是式(5) 里的 $(\cdot)^+$，等价于 numpy 的 `np.maximum(grad, 0)`：

```python
g = np.array([-0.8, 0.3, -0.1, 1.2])
np.maximum(g, 0)     # array([0. , 0.3, 0. , 1.2])
```

**为什么必须丢负值？** 若不丢，正负会互相抵消——某个位置可能"被 A 头强烈支持、被 B 头强烈反对"，平均后接近 0，看起来像"不重要"，实际是"争议很大"。丢弃负值等于只统计支持票，这是 Grad-CAM 系列的标准做法，原文表 9 的消融也显示去掉 ReLU 会掉点。

**为什么必须按 head 分别 ReLU 再平均？** 顺序不能换：

```python
per_head = np.array([[0.9, -0.5],      # head0
                     [-0.7,  0.4]])    # head1

# 正确：先 ReLU 再平均
np.maximum(per_head, 0).mean(axis=0)   # array([0.45, 0.45])
# 错误：先平均再 ReLU
np.maximum(per_head.mean(axis=0), 0)   # array([0.1, 0. ])  ← 正负提前抵消了
```

这正是 [[04-hook机制|hook]] 必须删掉 PyTorch 那句"对 head 求平均"的原因——PyTorch 提前平均掉，LeGrad 就没法按 head 单独 ReLU 了。

## 4. 为什么用梯度，而不是直接用注意力值

一个反例说明：假设 patch $k$（天空）被所有 token 大量关注，$A_{i,k}$ 全都很大。但它对"这是不是一只猫"毫无帮助——把它抹掉，分数几乎不变，即 $\partial s/\partial A_{i,k}\approx 0$。

- **raw attention** 会把天空涂成最亮 → 原文图 6 里的背景伪影
- **LeGrad** 只看梯度 → 天空接近 0，自动被滤掉

这就是原文那句"对不同提示词都均匀响应的噪声区域，梯度接近零"的意思。

## 5. 链式法则：$s$ 是怎么依赖到 $A^l$ 的

$s$ 并不是 $A^l$ 的直接函数，中间隔着一长串运算：

```text
A^l (12,197,197)
  │  加权求和 value：out_i = Σ_j A_ij v_j
  ▼
注意力输出 (197,1,768)
  │  + 残差
  ▼
Ẑ^l
  │  MLP + 残差
  ▼
Z^l (197,1,768)
  │  第 l+1 ... L-1 层继续变换
  ▼
Z^{L-1}
  │  mean(dim=0)：197 个 token 平均
  ▼
z̄^l (1,768)
  │  ln_post → @ proj → normalize
  ▼
图像向量 (1,512)
  │  与文本向量点积
  ▼
s (标量)
```

链式法则说：总敏感度 = 沿路每一段敏感度**相乘**。

$$\frac{\partial s}{\partial A^l} = \frac{\partial s}{\partial \text{图像向量}}\cdot\frac{\partial \text{图像向量}}{\partial \bar z^l}\cdot\frac{\partial \bar z^l}{\partial Z^l}\cdot\frac{\partial Z^l}{\partial A^l}$$

手工算这一串是噩梦，但 PyTorch 的 autograd 会自动完成——前提是前向时它**记录了每一步运算**。

注意一个细节：当 `layer_index=-2` 时，第 10 层的 $s^{10}$ 只依赖第 10、11 层，链条很短；但如果对第 0 层求梯度，链条要穿过全部 12 层。层越靠前，反向传播越长、越耗显存。

## 6. 计算图与 `requires_grad`

### 6.1 计算图是什么

前向传播时，PyTorch 对每个"需要求导"的运算都记一笔账：输入是谁、做了什么操作、输出是谁。这些账串起来就是**计算图**（一张有向图）。反向传播就是沿着这张图从 $s$ 往回走，逐段套用链式法则。

```python
import torch
x = torch.tensor([2.0], requires_grad=True)
y = x * 3            # 记账：y = mul(x, 3)
s = y.sum()          # 记账：s = sum(y)
print(s.grad_fn)     # <SumBackward0>   ← 这就是"账本"的最后一笔
```

`grad_fn` 非 `None` 说明这个张量在图里、可以反向。

### 6.2 `requires_grad` 决定图从哪里开始建

只有当某个张量 `requires_grad=True` 时，由它参与的运算才会被记账。

```python
a = torch.tensor([1.0])                       # requires_grad=False（默认）
b = a * 2
print(b.requires_grad)     # False → 没建图，无法反向
```

LeGrad 的做法（wrapper.py L77-83）：

```python
for name, param in self.named_parameters():
    param.requires_grad = False               # 全部先关掉
    if name.startswith('visual.transformer.resblocks'):
        depth = int(name.split('visual.transformer.resblocks.')[-1].split('.')[0])
        if depth >= self.starting_depth:      # 只打开第 starting_depth 层及之后
            param.requires_grad = True
```

输入图像本身 `requires_grad=False`，所以：

- 第 0 ~ `starting_depth-1` 层：所有参数都不求导 → **这些层的激活根本不进计算图**，不占反向所需的显存
- 第 `starting_depth` 层开始：参数求导 → 从这里往后的每一步运算都被记账

这就是 LeGrad 能在 ViT-BigG/14（25 亿参数）上跑起来的原因：不需要为整个网络保存反向所需的中间量，只为最后几层保存。也解释了为什么"层数"这个超参同时决定精度和显存——`layer_index` 越靠前，建图的层越多。

### 6.3 叶子张量 vs 非叶子张量

| | 叶子张量 | 非叶子张量 |
| --- | --- | --- |
| 谁 | 直接创建的、`requires_grad=True` 的张量（通常是模型参数） | 由运算产生的中间结果 |
| 例子 | `proj`、`in_proj_weight` | `attention_maps`、`feat_post_mlp` |
| `.backward()` 后 `.grad` | 有值 | **是 `None`**（PyTorch 默认不给中间结果存梯度，省显存） |

注意力图 $A^l$ 是 softmax 的输出，属于**非叶子张量**。所以：

```python
s.backward()
print(attn_map.grad)     # None  ← 拿不到！
```

要拿到它对中间张量的梯度，有两条路：

```python
# 路 1（LeGrad 采用）：用 torch.autograd.grad 显式指定要对谁求导，直接返回结果
grad = torch.autograd.grad(s, [attn_map], retain_graph=True)[0]

# 路 2：前向时先调用 attn_map.retain_grad()，再 backward，之后读 attn_map.grad
```

**关键区别：`torch.autograd.grad` 把梯度作为返回值交出来，不写进任何 `.grad` 属性。** 这直接导致后面第 8 节的结论。

## 7. `torch.autograd.grad` 的参数逐个解释

真实代码这一行：

```python
grad = torch.autograd.grad(one_hot, [attn_map], retain_graph=True, create_graph=True)[0]
```

| 参数 | 值 | 含义 | LeGrad 里是否必需 |
| --- | --- | --- | --- |
| `outputs` | `one_hot` | 对哪个标量求导，即 $s$ | 必需 |
| `inputs` | `[attn_map]` | 对谁求导，返回列表，所以要 `[0]` 取出 | 必需 |
| `retain_graph` | `True` | 反向完成后**不要拆掉计算图** | **必需** |
| `create_graph` | `True` | 让求导过程本身也可导（为二阶导数准备） | **多余，可去掉** |

### 7.1 为什么 `retain_graph=True` 必需

反向传播默认会**释放**计算图（省显存）。而 LeGrad 的结构是：

```python
_ = self.encode_image(image)          # 只前向一次，把 12 层的 hook 全填满
for layer in range(...):              # 然后对每一层各反向一次
    grad = torch.autograd.grad(s_l, [attn_map_l], retain_graph=True)
```

**一次前向要被 $L$ 次反向复用**。如果第一次反向就把图拆了，第二层再求导会直接报错：

```text
RuntimeError: Trying to backward through the graph a second time,
but the saved intermediate results have already been freed.
Specify retain_graph=True when calling .backward() or autograd.grad() the first time.
```

SigLIP 分支还有第二个理由：同一层要对**同一张图**反向两次（一次对真实 prompt，一次对空 prompt），两次都得留图。

### 7.2 为什么 `create_graph=True` 可以去掉

`create_graph=True` 会让返回的 `grad` 本身也带着计算图（`grad.grad_fn` 非 `None`），这样才能对梯度再求梯度（Hessian、对抗样本生成等场景需要）。

LeGrad 全程只做一阶求导，拿到 `grad` 之后只有 `clamp`、`mean`、`reshape`、`interpolate`、相加、归一化，再没有第二次 `autograd.grad`。所以：

- 去掉 `create_graph=True`（或改成 `False`）**结果完全一样**
- 好处：梯度张量不挂图，每层累加时不会把 $L$ 份图都吊在内存里，显存明显下降

这是复现全量 4,276 张图之前最值得改的一处。

### 7.3 `zero_grad()` 为什么是空操作

代码每层循环开头都有 `self.visual.zero_grad()`（SigLIP 分支是 `self.zero_grad()`）。它的作用是清空各参数的 `.grad` 属性。但：

- `torch.autograd.grad` **只返回梯度、不写 `.grad`**，所以从头到尾没有任何 `.grad` 被累加
- 而且 PyTorch 2.x 里 `zero_grad()` 默认 `set_to_none=True`，本来就是把 `.grad` 设成 `None`

结论：这行是防御性写法，删掉不影响结果。同理 `one_hot.requires_grad_(True)`（把单位阵设成可求导）也没有作用，因为从没对它求过导。

## 8. 反向传播的"一次性"与显存

一次反向的显存开销 ∝ 建图的层数 × 每层要保存的中间量。对 ViT-B/16、448 输入、$P$ 个 prompt：

| 项目 | 单个张量大小 | 说明 |
| --- | --- | --- |
| $A^l$ | $P\cdot12\times785\times785\times4$ B | $P=1$ 时约 28 MB，$P=5$ 时约 140 MB |
| 反向需保存的中间量 | 与 $A^l$ 同量级 | 每个建图的层都要存 |

所以两个省显存的杠杆：
1. **`layer_index`**：只对最后 $k$ 层建图（论文默认 `layer_index=-2` 只用 2 层）
2. **prompt 数量 $P$**：CLIP 分支把图像复制 $P$ 份一起前向，$P$ 越大注意力图越大。prompt 多时应分批调用

## 9. 小结：三句话记住梯度部分

1. 对矩阵求梯度得到**同形状的矩阵**，每格 = "只动这格，目标分数变多少"
2. LeGrad 用 `autograd.grad` 而非 `.backward()`，因为注意力图是**非叶子张量**，`.grad` 拿不到
3. `retain_graph=True` 必需（一次前向多次反向），`create_graph=True` 多余（没有二阶求导），`zero_grad()` 是空操作

下一篇：[[04-hook机制]]
