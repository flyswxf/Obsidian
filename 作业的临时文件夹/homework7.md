## 15.1-3
![[assets/钢条切割成本问题.png]]

由于收益与切割数挂钩, 基本的CUR-ROD算法将'不切割'也视作切割, 因此不再适用。需要调整循环结构和初始化

令 $r_n$ 为长度为 $n$ 的钢条切割后的最大收益。
如果不进行切割，收益为 $p_n$。
如果进行切割，假设第一段长度为 $i$ ($1 \le i < n$)，则我们需要支付切割成本 $c$，并获得第一段的收益 $p_i$ 以及剩余长度 $n-i$ 的最大收益 $r_{n-i}$。
因此，递归公式如下：
$$r_n = \max(p_n, \max_{1 \le i < n} (p_i + r_{n-i} - c))$$
其中 $r_0 = 0$。

伪代码如下：

```text
MODIFIED-CUT-ROD(p, n, c)
    let r[0..n] be a new array
    r[0] = 0
    for j = 1 to n
        q = p[j]  // 初始假设不切割
        for i = 1 to j - 1
            q = max(q, p[i] + r[j-i] - c)
        r[j] = q
    return r[n]
```

## 15.2-2
![[assets/矩阵链乘法递归算法.png]]

该算法利用 $s$ 表中存储的最优分割点 $k = s[i, j]$ 来递归地进行矩阵乘法。

```text
MATRIX-CHAIN-MULTIPLY(A, s, i, j)
    if i == j
        return A[i]
    k = s[i, j]
    X = MATRIX-CHAIN-MULTIPLY(A, s, i, k)
    Y = MATRIX-CHAIN-MULTIPLY(A, s, k + 1, j)
    return MATRIX-MULTIPLY(X, Y)
```

其中 `MATRIX-MULTIPLY(X, Y)` 是两个矩阵相乘的函数。

## 15.2-4
![[assets/矩阵链子问题图.png]]
**子问题图**:
子问题图是一个有向图，每个顶点代表一个子问题。对于矩阵链乘法，子问题由一对索引 $(i, j)$ 定义，表示计算 $A_{i \dots j}$ 的最优代价，其中 $1 \le i \le j \le n$。

**顶点数**:
顶点对应所有可能的子问题 $(i, j)$，满足 $1 \le i \le j \le n$。
顶点的总数为：
$$ \sum_{k=1}^{n} k = \frac{n(n+1)}{2} $$

**边**:
对于每个顶点 $(i, j)$，如果 $j > i$，需要考虑所有可能的分割点 $k$ ($i \le k < j$)。对于每个 $k$，计算 $(i, j)$ 需要用到子问题 $(i, k)$ 和 $(k+1, j)$ 的结果。
在子问题图中，这表示从顶点 $(i, j)$ 指向 $(i, k)$ 和 $(k+1, j)$ 的有向边。
对于给定的 $(i, j)$，有 $j-i$ 个可能的 $k$ 值。每个 $k$ 值对应 2 条边（分别指向两个子问题）。
因此，顶点 $(i, j)$ 的出度为 $2(j-i)$。

**边数**:
总边数为所有顶点出度之和：
$$ E = \sum_{1 \le i < j \le n} 2(j-i) $$
令 $L = j - i$ 为链长，则 $L$ 的取值范围是 $1$ 到 $n-1$。对于固定的 $L$，有 $n-L$ 个这样的子问题（即 $i$ 可以从 $1$ 取到 $n-L$）。
$$ \begin{aligned} E &= \sum_{L=1}^{n-1} (n-L) \times 2L \\ &= 2 \sum_{L=1}^{n-1} (nL - L^2) \\ &= 2 \left( n \frac{(n-1)n}{2} - \frac{(n-1)n(2n-1)}{6} \right) \\ &= n^2(n-1) - \frac{n(n-1)(2n-1)}{3} \\ &= n(n-1) \left( n - \frac{2n-1}{3} \right) \\ &= n(n-1) \frac{3n - 2n + 1}{3} \\ &= \frac{n(n-1)(n+1)}{3} \\ &= \frac{n^3 - n}{3} \end{aligned} $$

所以，顶点数为 $\frac{n(n+1)}{2}$，边数为 $\frac{n^3 - n}{3}$。
