# 1-1

对下列函数按增长阶（当 $n\to\infty$）从小到大排序。

$$
\begin{aligned}
&\lg(\lg^* n),\quad 2^{\lg^* n},\quad (\sqrt{2})^{\lg n},\quad n^2,\quad n!,\quad (\lg n)! \\
&\left(\tfrac{3}{2}\right)^n,\quad n^3,\quad (\lg n)^2,\quad \lg(n!),\quad 2^{2^n},\quad n^{1/\lg n} \\
&\ln\ln n,\quad \lg^* n,\quad n\cdot 2^n,\quad n^{\lg\lg n},\quad \ln n,\quad 1 \\
&2^{\lg n},\quad (\lg n)^{\lg n},\quad e^n,\quad 4^{\lg n},\quad (n+1)!,\quad \sqrt{\lg n} \\
&\lg^*(\lg n),\quad 2^{\sqrt{2\lg n}},\quad n,\quad 2^n,\quad n\lg n,\quad 2^{2^{n+1}}
\end{aligned}
$$

$\lg$ 为以 2 为底的对数，$\ln$ 为自然对数
$\lg^* n$ 定义如下, 即对 n 反复取以 2 为底的对数直到结果不大于 1 所需的次数
$$
\lg^* n = \min\{\,i\ge 0: \lg^{(i)} n\le 1\,\},\quad
\lg^{(0)} n=n,\ \ \lg^{(i)} n=\lg(\lg^{(i-1)} n)\ \ (i>0).
$$

结论：

$$
\begin{aligned}
&1\ \ <\ \ n^{1/\lg n}\;=\;2\ \ <\ \ \lg(\lg^* n)\ \ <\ \ \lg^*(\lg n)\ \ <\ \ \lg^* n\ \ <\ \ 2^{\lg^* n} \\
&<\ \ \ln\ln n\;=\;\Theta(\lg\lg n)\ \ <\ \ \sqrt{\lg n}\ \ <\ \ \ln n\;\ <\ \ (\lg n)^2\ \ <\ \ 2^{\sqrt{2\lg n}} \\
&<\ \ (\sqrt{2})^{\lg n}\;=\;\sqrt{n}\ \ <\ \ 2^{\lg n}\;=\;n\ \ <\ \ n\lg n\;\ \ <\ \ n^2\;=\;4^{\lg n}\ \ <\ \ n^3 \\
&<\ \ (\lg n)!\ \ <\ \ n^{\lg\lg n}\;=\;(\lg n)^{\lg n} \\
&<\ \ \left(\tfrac{3}{2}\right)^n\ \ <\ \ 2^n\ \ <\ \ n\cdot 2^n\ \ <\ \ e^n\ \ <\ \ n!\ \ <\ \ (n+1)! \\
&<\ \ 2^{2^n}\ \ <\ \ 2^{2^{n+1}}\;\, .
\end{aligned}
$$
1. $1\ \ <\ \ n^{1/\lg n}\; = 2$ :
	$n^{1/\lg n}=(2^{\lg n})^{1/\lg n}=2$
2. $lg(\lg^* n)\ \ <\ \ \lg^*(\lg n)$
	$lg(\lg^* n)$多求了一次对数
3. $\lg^*(\lg n)\ \ <\ \ \lg^* n$
	$\lg^*(\lg n)=\lg^* n-1$
4. $\lg^* n\ \ <\ \ 2^{\lg^* n}$
	左右同求$\lg$, 化简为$lg(\lg^* n)$和$\lg^* n$, 根据结论2,3, 可知$lg(\lg^* n) < \lg^* n$
5. $2^{\lg^* n} <\ \ \ln\ln n$
	左右同求$\lg$, 化简为$\lg^* n$和$\lg \ln\ln n$, n足够大时,左侧可视为进行无数次$lg$操作,而右侧进行有限次,因此右侧更大
6. $ln\ln n\;=\;\Theta(\lg\lg n)\ \ <\ \ \sqrt{\lg n}$
	$ln\ln n\;=\;\Theta(\lg\lg n)$: $lg$与$ln$仅影响常数因子
	$\Theta(\lg\lg n)\ \ <\ \ \sqrt{\lg n}$: $lg$比开根下降更快
7. $\sqrt{\lg n}<\ln n< (\lg n)^2$
	$lg$与$ln$仅影响常数因子
8. $(\lg n)^2<2^{\sqrt{2\lg n}}$
	指数比幂上升速度更快
9. $2^{\sqrt{2\lg n}} <\ \ (\sqrt{2})^{\lg n}$
	左右同求$\lg$, 化简为$\sqrt{2\lg n}$和$\tfrac{1}{2}\lg n$, 显然右侧更大
10. $(\sqrt{2})^{\lg n}\;=\;\sqrt{n}\ \ <\ \ 2^{\lg n}\;=\;n\ \ <\ \ n\lg n<\ \ n^2=\;4^{\lg n}\ \ <\ \ n^3$
	化简$\lg$即可
11. $n^3 <\ \ (\lg n)!$
	令$x=\lg n$, 右式变为$x!$, 左式变为$2^{3x}$, 阶乘比指数上升更快,有$x!>2^{3x}$
12. $(\lg n)!\ \ <\ \;(\lg n)^{\lg n}$
	$x(x-1)(x-2)...1<x*x*x...*x$
13. $(\lg n)^{\lg n}=n^{\lg\lg n}$
	$(\lg n)^{\lg n}=2^{\lg n\cdot\lg\lg n}=n^{\lg\lg n}$
14. $(\lg n)<\left(\tfrac{3}{2}\right)^n$
15. $\left(\tfrac{3}{2}\right)^n\ \ <\ \ 2^n\ \ <\ \ e^n$
	底数大的指数更快
16. $n\cdot2^n<e^n$
	左右同除$2^n$, 显然有$n<\left(\tfrac{e}{2}\right)^n$
17. $e^n\ \ <\ \ n!$
	阶乘比指数增长更快
18. $(n+1)! <\ \ 2^{2^n}$
	双指数比阶乘增长更快

---
# 1-2 

![[渐近函数性质分析.png]]
 f(n)、g(n) 为“渐近正”的函数，即存在 n0，使得对所有 n≥n0 都有 f(n)≥0、g(n)≥0

---

### a)$f(n)=O(g(n)) ⇒ g(n)=O(f(n))$

错误

#### 反例
令 $f(n)=1$、$g(n)=n$（两者均渐近正）
- $\forall n≥1$，$f(n)=1≤1·n=g(n)$，因此满足 $f(n)=O(g(n))$
- 若要 $g(n)=O(f(n))$ 成立，则需存在常数 $C>0$ 与n0，使得对所有 $n≥n0$ 有 $n≤C·1$，即 $n≤C$
- 这与 $n→∞$ 时 $n$ 无界矛盾，故断言不成立


---

### b) $f(n)+g(n)=Θ(min(f(n), g(n)))$

错误。对于渐近正函数，正确的量级是 $f(n)+g(n)=Θ(max(f(n), g(n)))$，而不是 $Θ(min)$

#### 反例：取 $f(n)=n、g(n)=1$。
- 则$min(f(n),g(n))=1$,  $f(n)+g(n)=n+1=Θ(n)$
- 由于 $Θ(1)≠Θ(n)$，故题述断言不成立

#### 补充证明$（f(n)+g(n)=Θ(max(f(n), g(n)))$：
对任意非负实数 a,b，有
- $max(a,b) ≤ a+b ≤ 2·max(a,b)$。
将 $a=f(n)$、$b=g(n)$ 带入, 得到
- $max(f(n),g(n)) ≤ f(n)+g(n) ≤ 2·max(f(n),g(n))$（当 n 足够大)
因此 $f(n)+g(n)=Θ(max(f(n),g(n)))$

---

### c) 在 $lg(g(n))≥1$ 且 $f(n)≥1$（当 n 充分大）条件下，若 $f(n)=O(g(n))$，则 $lg(f(n))=O(lg(g(n)))$

正确

#### 证明：
由 $f(n)=O(g(n))$，存在常数 $c>0$ 与 n0，使得当 $n≥n0$ 时，有 $0≤f(n)≤c·g(n)$
不妨将 c 替换为 $max${$c,1$}，从而可设 $c≥1$
于是当 $n≥n0$ 时
- $lg(f(n)) ≤ lg(c·g(n)) = lg c + lg(g(n))$
由题设 $lg(g(n))≥1$（将'当n充分大表示为n≥n1）以及 $\lg c≥0$（因为 c≥1），有
- $lg c ≤ (lg c)·lg(g(n))$，于是
- $lg(f(n)) ≤ lg c + lg(g(n)) ≤ (1+lg c)·lg(g(n))$
令 $N=max${$n0,n1$}、$C=1+lg c$，当 $n≥N$ 时成立 $lg(f(n)) ≤ C·lg(g(n))$，即 $lg(f(n))=O(lg(g(n)))$

---

### d) $f(n)=O(g(n)) ⇒ 2^{f(n)}=O(2^{g(n)})$

错误。

#### 反例
取 $g(n)=n$、$f(n)=2n$
则对所有 $n≥1$，有 $f(n)=2n≤2·n=2·g(n)$，满足 $f(n)=O(g(n))$。
但若 $2^{f(n)}=O(2^{g(n)})$ 成立，则需存在常数 $C>0$ 与 n0，使得当 $n≥n0$ 时
- $2^{2n} ≤ C·2^n$，即 $2^n ≤ C$。
这与 $n→∞$ 时 $2^n$ 无界矛盾，故断言不成立。

---
# 1-3 
![[两数和判定算法.png]]

给定整数集合 $S[1..n]$与目标整数 $x$，判定是否存在两个元素 $a,b\in S$ 使得 $a+b=x$

算法（伪代码）：

```
TwoSumExists(S, x):
    sort S in nondecreasing order           
    i ← 1                                    
    j ← n                                    
    while i < j:
        s ← S[i] + S[j]
        if s == x:                           
            return True
        else if s < x:                      
            i ← i + 1
        else:                                
            j ← j - 1
    return False
```

- 每次迭代后，区间 $[i,j]$ 保留所有可能成为解的候选对
	- 如果存在解，指针移动不会跳过该解
	- 当遇到 $S[i]+S[j]=x$ 即返回 True；否则最终 $i\ge j$，说明不存在满足条件的二元组

时间复杂度：
- 排序阶段使用归并排序耗时 $O(n\lg n)$
- 双指针扫描耗时 $O(n)$
- 总计 $O(n\lg n) + O(n) = O(n\lg n)$

---

# 1-4 
![[选择排序算法分析.png]]

对数组 $A[1..n]$，第 $r$ 轮（$r=n,n-1,\dots,2$）在前缀 $A[1..r]$ 中找到最大元素的位置 $p$，将其与 $A[r]$ 交换。如此迭代，最终得到非降序数组

#### 伪代码：
```
SelectionSortByMax(A,n):
    for r ← n down to 2:                    
        p ← 1                               
        for i ← 2 to r:                     
            if A[i] > A[p]:
                p ← i
        if p ≠ r:                           
            swap A[p], A[r]
```

#### 循环不变式：
在外层循环每次迭代开始时，后缀 $A[r+1..n]$ 已按非降序排列，并恰好由数组中最大的 $n-r$ 个元素构成。
- 初始化（$r=n$）：后缀为空，显然成立
- 维持：在前缀 $A[1..r]$ 中找到最大值并置于 $A[r]$，它不小于任何前缀元素，从而新的后缀 $A[r..n]$ 由最大的 $n-r+1$ 个元素构成且仍为非降序
- 终止：当外层循环结束（$r=1$）时，后缀 $A[2..n]$ 已有序且包含最大的 $n-1$ 个元素；剩余的 $A[1]$ 必然是最小元素，故整个数组非降序

#### 时间复杂度:
- **比较次数与输入无关**
- 第 $r$ 轮内层扫描做 $r-1$ 次比较，故总比较次数为
  $$\sum_{r=2}^{n} (r-1) = \frac{n(n-1)}{2} = O(n^2).$$
- 因此，无论最佳还是最差情况，总运行时间皆为 $O(n^2)$。

