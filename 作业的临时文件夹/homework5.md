## 8.1-3
![[assets/比较排序下界.png]]

使用决策树模型模拟比较排序, 则
- 深度等于比较次数
- $n!$ 个不同输入对应 $n!$ 个叶子。
若某算法在至少 $\alpha(n)$ 的比例输入上用不超过 $c\cdot n$ 次比较（$c$ 为常数），则其决策树中至少有 $\alpha(n)\cdot n!$ 个叶子深度不超过 $c\cdot n$。而二叉树在深度不超过 $c\cdot n$ 的叶子最多为 $2^{c n}$，因此必须有：

  $$\alpha(n)\cdot n! \le 2^{c n}$$
求对数并化简得到:
$$n(\log_2 n - \log_2 e) + \log_2 \alpha(n) \le c\cdot n$$


### 1. $\alpha(n)=1/2$
原式 $=c\cdot n \ge  n\log_2 n - \Theta(n)$，对足够大的 $n$ 不可能。
因此不存在在至少一半输入上线性时间的比较排序。

### 2. $\alpha(n)=1/n$
原式 $=c\cdot n \ge  n\log_2 n - \Theta(n)$，同样对足够大的 $n$ 不可能
因此不存在在至少 $1/n$ 的比例输入上线性时间的比较排序。

### 3. $\alpha(n)=1/2^n$
化简结果相同, 对足够大的 $n$ 不可能。
不存在在至少 $1/2^n$ 的比例输入上线性时间的比较排序。

## 8.2-4
![[assets/前缀和数组区间查询.png]]
额外使用一个计数数组`freq[0..k]` , 一个前缀和数组`pref[0..k]`
对`n`个数, 遍历它们, 对应`freq++`, 耗时$O(n)$
对计数数组使用前缀和的处理方式, 构建前缀和数组 `pref[0..k+1]`。对 $i=0..k$，`pref[i+1]=pref[i]+freq[i]`，耗时 $O(k)$。

这样对于每个合法的落在区间内的查询, 首先判断边界, $a\leftarrow\max(0,a)$, $b\leftarrow\min(k,b)$, 再执行${pref}[b+1]-\text{pref}[a]$
单次查询时间 $O(1)$

## 8.3-4
![[assets/线性时间排序问题.png]]

使用基数排序
根据基数排序时间公式
$$T(n,b)=\Theta(\frac{b}{r}(n+2^r))$$

其中
- `b`为基数的二进制位数, 在此题中=$lg(n^3)=3lgn$
- `r`为拆分后每个数的二进制位数
将原数平均分为3个word, 则$r=lgn$, $b=3lgn$, 则
$$T(n,b)=\Theta(3n)$$
即可在$O(n)$的时间内对0到$n^3-1$的$n$数组进行排序

## 8.4-2
![[assets/桶排序复杂度分析.png]]

为什么最坏时间是 $\Theta(n^2)$：
- 经典桶排序将输入映射到 $n$ 个桶，并在每个桶内用插入排序
- 设第 $i$ 个桶大小为 $m_i$，则桶内插入排序代价为 $\Theta(m_i^2)$，总时间为
  $$T(n)=\Theta(n)+\sum_{i=1}^{k}O(m_i^2).$$
- 最坏情况下所有元素落入同一桶，存在某个 $m_j=n$，于是 $T(n)=\Theta(n^2)$

如何保持平均线性并将最坏情况复杂度降到 $O(n\lg n)$：
将桶内排序算法替换为归并排序, 则桶内使用以 $O(m\lg m)$ 为最坏时间复杂度的排序
此时总时间变为
  $$T'(n)=\Theta(n)+\sum_{i=1}^{k}O(m_i\lg m_i).$$
因为 $m_i\le n$ 且 $\sum_i m_i=n$，有
  $$\sum_{i} m_i\lg m_i \le \sum_{i} m_i\lg n = n\lg n,$$
故最坏时间为 $O(n\lg n)$。
然后验证平均时间复杂度为线性: 
令第 $i$ 个桶的计数为 $m_i=\sum_{j=1}^{n} X_{ij}$，其中 $X_{ij}=\mathbf{1}\{A[j]\text{ 落在桶 }i\}$，且在均匀随机分布下 $\Pr(X_{ij}=1)=p=1/k$，并且对于不同的 $j$ 相互独立。
  $$\mathbf{E}[m_i^2]=\mathbf{E}\Big[\big(\sum_{j=1}^{n} X_{ij}\big)^2\Big]=\sum_{j=1}^{n}\mathbf{E}[X_{ij}]+2\sum_{1\le j<\ell\le n}\mathbf{E}[X_{ij}X_{i\ell}] 
  =np+n(n-1)p^2.$$
对任意整数 $m\ge 0$, 有 $\lg m\le m$, 则
  $$m\,\lg m\le m^2.$$
于是
  $$\mathbf{E}[m_i\,\lg m_i]\le \mathbf{E}[m_i^2]=np+n(n-1)p^2.$$
当桶数 $k=\Theta(n)$（经典设置 $k=n$）时，$p=1/k=\Theta(1/n)$，从而 $np=\Theta(1)$，$n^2p^2=\Theta(1)$，故
  $$\mathbf{E}[m_i\,\lg m_i]=O(1).$$
再对所有桶求和：
  $$\sum_{i=1}^{k}\mathbf{E}\bigg[ m_i\,\lg m_i\bigg]\le \sum_{i=1}^{n} O(1)=O(n),$$
  $$\mathbf{E}[T(n)]=\mathbf{E}[\Theta(n)+\sum_{i=1}^{k}O(m_i\lg m_i)]
=\Theta(n)+\sum_{i=1}^{k}O(\mathbf{E}[m_i\lg m_i])=\Theta(n)+O(n)=\Theta(n)
  $$
  因而在该修改下，总期望开销仍为线性。