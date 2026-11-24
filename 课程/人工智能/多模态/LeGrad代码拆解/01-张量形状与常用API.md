> [!info] 目标
> 看懂 LeGrad 代码里每一次形状变化。全部用 numpy 举例（本地已装 numpy，代码可直接跑），PyTorch 的 API 名字与 numpy 基本一一对应，差异单独标出。

## 1. 形状怎么读

一个张量的 `shape` 就是"每个维度有多长"，**从左到右是从外到内**。

```python
import numpy as np

g = np.zeros((3, 12, 785, 785))
print(g.shape)      # (3, 12, 785, 785)
print(g.ndim)       # 4  → 4 个维度
```

读法：`g` 是一个"3 层抽屉柜"，每层抽屉柜里有 12 个抽屉，每个抽屉里有一张 $785\times785$ 的表格。

对应到 LeGrad：

| 维度位置       | 长度  | 含义                            |
| ---------- | --- | ----------------------------- |
| `shape[0]` | 3   | 第几个 prompt（batch）             |
| `shape[1]` | 12  | 第几个注意力 head                   |
| `shape[2]` | 785 | 注意力表的**行** = query token（提问方） |
| `shape[3]` | 785 | 注意力表的**列** = key token（被看方）   |

`dim=0` 就是 `shape[0]` 那一维。负数索引表示从右往左数：`dim=-1` 是最后一维（785 列），`dim=-2` 是倒数第二维（785 行）。

## 2. 索引与切片

```python
a = np.arange(24).reshape(2, 3, 4)   # shape (2, 3, 4)

a[0]          # shape (3, 4)   取第 0 个"块"，维度减一
a[0, 1]       # shape (4,)     取第 0 块的第 1 行
a[0, 1, 2]    # 标量           一个具体数字
a[:, 1:]      # shape (2, 2, 4) 所有块，行从 1 开始到末尾
a[..., 0]     # shape (2, 3)   ... 表示"中间所有维都要"，等价于 a[:, :, 0]
```

**关键点：单个整数索引会吃掉那一维，切片不会。**

```python
a[0].shape       # (3, 4)   ← 少了第一维
a[0:1].shape     # (1, 3, 4) ← 维数不变，只是长度变 1
```

LeGrad 里 `grad.mean(dim=1).mean(dim=1)[:, 1:]` 的最后一步 `[:, 1:]` 是切片，所以维数不变，只是最后一维从 785 变成 784。

## 3. 求平均 `mean(dim=)`：把某一维"压扁"

`mean(dim=k)` 的意思是：**沿着第 $k$ 维把数字加起来除以个数，然后这一维消失**。

```python
a = np.array([[1., 2., 3.],
              [4., 5., 6.]])        # shape (2, 3)

a.mean(dim=0)   # array([2.5, 3.5, 4.5])  shape (3,)  ← 上下压扁：(1+4)/2, (2+5)/2, (3+6)/2
a.mean(dim=1)   # array([2., 5.])         shape (2,)  ← 左右压扁：(1+2+3)/3, (4+5+6)/3
a.mean()        # 3.5                                ← 不给 dim 就全压成一个数
```

记忆法：`mean(dim=0)` = "对第 0 维负责，把它消掉"，剩下的形状就是原形状去掉第 0 维。

LeGrad 里连续两次 `mean(dim=1)`：

```python
g = np.zeros((3, 12, 785, 785))     # (prompt, head, 行, 列)

g.mean(dim=1).shape          # (3, 785, 785)   head 维被消掉 → 12 个 head 求平均
g.mean(dim=1).mean(dim=1).shape   # (3, 785)   再消一次，这次消的是"行"（query）
```

**注意第二次的 `dim=1` 已经不是 head 了**：第一次平均之后形状变了，原来的第 2 维（行）升到了第 1 维。这是最容易看错的地方。

`keepdims=True` 可以保留维度（长度变 1），LeGrad 的 `min_max` 里就用到：

```python
a.mean(dim=1, keepdims=True).shape   # (2, 1) 而不是 (2,)
```

## 4. 求和与点积

```python
a = np.array([1., 2., 3.])
b = np.array([4., 5., 6.])

a.sum()          # 6.0        全部加起来 → 标量
(a * b)          # array([4., 10., 18.])   逐元素相乘（不是点积！）
(a * b).sum()    # 32.0       逐元素相乘再求和 = 点积 = 4+10+18
a @ b            # 32.0       numpy/torch 的点积写法，与上面等价
```

`*` 是逐元素乘，`@` 才是矩阵乘／点积。`(a*b).sum()` 和 `a@b` 结果相同，代码里两种写法都会出现。

## 5. 矩阵乘法 `@`：形状怎么变

规则：**$(m,k) @ (k,n) = (m,n)$**，中间那个 $k$ 必须相等，且会被消掉。

```python
A = np.zeros((2, 3))
B = np.zeros((3, 5))
(A @ B).shape          # (2, 5)

# 批量情形：前面多出来的维度当作"批次"，只在最后两维上做矩阵乘
X = np.zeros((7, 2, 3))
Y = np.zeros((7, 3, 5))
(X @ Y).shape          # (7, 2, 5)
```

LeGrad 里的 `text_embedding @ img_feat.transpose(-1, -2)`：

```python
text = np.zeros((3, 512))     # 3 个 prompt，每个 512 维
img  = np.zeros((3, 512))     # 3 份图像副本，每个 512 维
img.transpose(-1, -2).shape   # (512, 3)
(text @ img.transpose(-1, -2)).shape    # (3, 3)
```

得到的 $(3,3)$ 矩阵里，`sim[i, j]` = 第 $i$ 个 prompt 与第 $j$ 份图像的相似度。

`transpose(-1, -2)` 是"交换最后两维"，也就是矩阵转置。对二维数组 `A.T` 等价；对高维数组，`transpose` 只动指定的两维。`permute` 是 PyTorch 里可以任意重排所有维的版本。

## 6. 拉直与摆回：`view` / `reshape`

两者作用相同（PyTorch 里 `view` 要求内存连续，`reshape` 更宽松），**总元素个数必须不变**。

```python
x = np.arange(196).reshape(14, 14)    # (14, 14)
x.reshape(196).shape                  # (196,)   拉直：一行一行接起来
x.reshape(196)[0:3]                   # array([0, 1, 2])
```

拉直的顺序是**行优先**：先第 0 行的 14 个数，再第 1 行的 14 个数……所以 `reshape(196)` 之后再 `reshape(14, 14)` 能原样还原。

LeGrad 的第 11 步就是把 196 个重要性数字按这个顺序摆回 $14\times14$ 网格，位置 $(r, c)$ 对应第 $r\times14+c$ 个 patch，正好是图像上从上到下、从左到右的顺序。

`-1` 表示"这一维的长度你自己算"：

```python
g = np.zeros((36, 785, 785))
g.reshape(3, -1, 785, 785).shape    # (3, 12, 785, 785)  ← -1 被算成 12
```

## 7. 增减维度：`unsqueeze` / `squeeze` / `None`

```python
x = np.zeros((3, 28, 28))

x[None].shape                  # (1, 3, 28, 28)   在最前面插一维
np.expand_dims(x, 0).shape     # 同上，numpy 写法
x[:, None].shape               # (3, 1, 28, 28)   在第 1 位插一维
np.squeeze(x).shape            # (3, 28, 28)      去掉所有长度为 1 的维
```

LeGrad 的 `rearrange(image_relevance, 'b (w h) -> 1 b w h')` 里的 `1` 就是新插的一维，充当"通道数"占位，因为后面的 `F.interpolate` 要求输入是 4 维 $(N, C, H, W)$。

## 8. 复制：`repeat` / `tile` / 广播

```python
img = np.zeros((1, 3, 224, 224))

img.repeat(3, axis=0).shape       # (3, 3, 224, 224)  numpy：沿 axis 复制 3 次
# PyTorch: img.repeat(3, 1, 1, 1)  ← 每个维度各复制几倍，(3,1,1,1) 表示只复制第 0 维
```

PyTorch 的 `repeat(3, 1, 1, 1)` 读作"第 0 维变 3 倍，其余维不变"，结果是把同一张图**堆 3 份**。LeGrad 用它给 3 个 prompt 各配一份图像副本。

**广播（broadcasting）** 是另一种"隐式复制"：形状为 1 的维会自动扩展到匹配。

```python
a = np.zeros((3, 5))
b = np.zeros((1, 5))
(a + b).shape        # (3, 5)   b 被自动复制 3 次
```

`min_max` 里的 `(logits - logits_min)` 就靠广播：`logits_min` 形状 $(1,3,1,1)$，`logits` 形状 $(1,3,448,448)$，相减时前者被广播。

## 9. 其余零散 API

| API | 作用 | LeGrad 里的用途 |
| --- | --- | --- |
| `torch.clamp(x, min=0.)` | 小于 0 的截成 0，即 ReLU | 式(5) 的 $(\cdot)^+$，丢掉负梯度 |
| `F.normalize(x, dim=-1)` | 每个向量除以自己的长度 → 单位向量 | 让点积等于余弦相似度 |
| `F.one_hot(torch.arange(3))` | 造一个 $3\times3$ 单位阵 | 从 `sim` 里挑出对角线 |
| `F.interpolate(x, scale_factor=16, mode='bilinear')` | 双线性放大，边长 ×16 | $14\times14 \to 224\times224$ |
| `torch.autograd.grad(y, [x])` | 求 $\partial y/\partial x$ 并直接返回 | 式(4) |
| `x.bool_mask` / `x[mask] = 0` | 布尔索引，按条件取值或赋值 | SigLIP 的空提示校正 |
| `torch.arange(0, n)` | 生成 $0,1,\dots,n-1$ | 造单位阵的下标 |

`F.normalize` 具体做了什么：

```python
v = np.array([3., 4.])
v / np.linalg.norm(v)        # array([0.6, 0.8])  长度变成 1
```

两个单位向量的点积 = 夹角余弦，取值 $[-1, 1]$。CLIP 的相似度就是这么算的，所以代码里到处都要先 `normalize`。

布尔索引的例子（SigLIP 分支的校正）：

```python
heat = np.array([[0.1, 0.9], [0.5, 0.95]])
mask = heat > 0.8            # array([[False,  True], [False,  True]])
heat[mask] = 0               # 只把 True 的位置改成 0
print(heat)                  # [[0.1, 0. ], [0.5, 0. ]]
```

## 10. einops `rearrange` 专题

`rearrange` 是"带名字的形状变换"，比 `reshape`+`transpose` 好读得多。语法：

```python
rearrange(张量, '输入侧的轴名 -> 输出侧的轴名', 轴名=长度)
```

三条规则：

1. **单个字母 = 一个轴**。`'b h n m'` 表示这个张量有 4 个轴，分别叫 `b`、`h`、`n`、`m`。
2. **括号 = 合并**。`'(b h)'` 表示原来这一维其实是 `b` 和 `h` 乘起来的，要拆成两维（拆的时候必须告诉它 `b=` 多少或 `h=` 多少，另一个能算出来）。
3. **输出侧新出现的名字 = 插入长度为 1 的新维**；输出侧消失的名字 = 被合并掉。

LeGrad 里出现三次：

**① `rearrange(grad, '(b h) n m -> b h n m', b=num_prompts)`**

```python
grad.shape        # (36, 785, 785)   36 = 3 个 prompt × 12 个 head，被压在一起了
# 告诉它 b=3，它就拆成 (3, 12, 785, 785)
```

为什么必须拆：PyTorch 的多头注意力实现把 batch 和 head 揉在同一维（这是为了用 `bmm` 批量算），但式(5) 要求"只对 head 平均、不对 prompt 平均"，不拆开就分不清哪个是 head。

**② `rearrange(image_relevance, 'b (w h) -> 1 b w h', w=w, h=h)`**

```python
image_relevance.shape   # (3, 784)
# b=3, (w h)=784 且 w=h=28 → 输出 (1, 3, 28, 28)
```

`784` 被拆成 `28×28` 的网格，最前面插一个长度为 1 的通道维。等价于 numpy 的：

```python
image_relevance[:, None].reshape(3, 1, 28, 28)   # 先插维再摆网格
```

**③ 反向操作 `rearrange(x, 'b h n m -> (b h) n m')`** 在别处也常见，就是把拆开的再合回去。

一句话记忆：**箭头左边是"现在长什么样"，右边是"想变成什么样"，括号表示"这一维其实是几维乘起来的"。**

## 11. LeGrad 全部形状操作汇总

| 代码写法 | 形状变化 | 语义 |
| --- | --- | --- |
| `feat_post_mlp.mean(dim=0)` | $(197,P,768)\to(P,768)$ | 197 个 token 求平均（LND 布局，token 在第 0 维） |
| `ln_post(·) @ proj` | $(P,768)\to(P,512)$ | 投影到图文共享空间 |
| `text @ img.transpose(-1,-2)` | $(P,512)\times(P,512)\to(P,P)$ | 两两算余弦相似度 |
| `torch.sum(one_hot * sim)` | $(P,P)\to$ 标量 | 取对角线之和 |
| `autograd.grad(s, [A])` | 标量 $\to(P\cdot h,T,T)$ | 求敏感度表 |
| `rearrange('(b h) n m -> b h n m')` | $(P\cdot h,T,T)\to(P,h,T,T)$ | 拆开 batch 与 head |
| `clamp(min=0.)` | 形状不变 | 负敏感度置零 |
| `.mean(dim=1).mean(dim=1)` | $(P,h,T,T)\to(P,T)$ | 先平均 head，再平均 query 行 |
| `[:, 1:]` | $(P,T)\to(P,T-1)$ | 丢掉 `[CLS]` 那一列 |
| `rearrange('b (w h) -> 1 b w h')` | $(P,196)\to(1,P,14,14)$ | 摆回 2D 网格 + 插通道维 |
| `interpolate(scale_factor=16)` | $(1,P,14,14)\to(1,P,224,224)$ | 放大到原图尺寸 |
| `accum += expl_map` | 形状不变 | 逐层累加 |
| `min_max(accum)` | 形状不变 | 每个 prompt 各自归一化到 $[0,1]$ |

下一篇：[[02-ViT内部结构]]
