# 4.1-4
![[Pasted image 20251012190140.png]]
### 额外改动
- 当初始数组就是空时, 返回空数组
```cpp
if (r < l)
        return {0, -1, -1}; // 空区间
```
- 在叶子节点额外判断空数组的情况
```cpp
if (l == r){
	if(a[l]>0) {
		return emptyArr;
	}
	return emptyArr;
}
```
-  在横跨左右子数组的case中额外判断空数组的情况
 ```cpp
if(leftBest<0) {
	leftBest=0;
	bestL=mid;
}
```

### 完整算法
```cpp
struct Ans
{
    long long sum;
    int L, R;
};
const Ans emptyArr = {0, -1, -1};

Ans cross(const vector<int> &a, int l, int mid, int r)
{
    long long s = 0, leftBest = LLONG_MIN;
    int bestL = mid;
    for (int i = mid; i >= l; --i)
    {
        s += a[i];
        if (s > leftBest)
        {
            leftBest = s;
            bestL = i;
        }
    }
    if(leftBest<0) {
        leftBest=0;
        bestL=mid;
    }
    s = 0;
    long long rightBest = LLONG_MIN;
    int bestR = mid + 1;
    for (int i = mid + 1; i <= r; ++i)
    {
        s += a[i];
        if (s > rightBest)
        {
            rightBest = s;
            bestR = i;
        }
    }
    if(rightBest<0) {
        rightBest=0;
        bestR=mid;
    }
    return {leftBest + rightBest, bestL, bestR};
}

Ans solve(const vector<int> &a, int l, int r)
{
    if (l == r){
        if(a[l]>0) {
            return emptyArr;
        }
        return emptyArr;
    }
    if (r < l)
        return emptyArr;
    int mid = (l + r) / 2;
    Ans L = solve(a, l, mid);
    Ans R = solve(a, mid + 1, r);
    Ans C = cross(a, l, mid, r);
    return max({L, R, C}, [](const Ans &x, const Ans &y)
               { return x.sum < y.sum; });
}
```

# 4.1-5
![[Pasted image 20251012192005.png]]
`A[1..j+1]` 的最大子数组要么是 `A[1..j]` 的最大子数组，要么是形如 `A[i..j+1]`（以 `j+1` 结尾的某个子数组）。
因此维护两个量：
- `bestEnd`：以当前索引处结尾的最大子数组和；
- `bestSum`：截至目前出现过的最大子数组和。

伪代码：
```cpp
bestEnd = A[0]; bestSum = A[0]
start = 0; end = 0; tempStart = 0
for j = 1..n-1:
    if bestEnd + A[j] < A[j]:
        bestEnd = A[j]
        tempStart = j
    else:
        bestEnd = bestEnd + A[j]
    if bestEnd > bestSum:
        bestSum = bestEnd
        start = tempStart; end = j
return (start, end, bestSum)
```

# 4.2-4
![[Pasted image 20251012194524.png]]
参考 Strassen 思路, 将 n×n 矩阵划分为 3×3 的块矩阵（每块大小约为 n/3）。若能用 k 次块乘法完成一次 3×3 块矩阵乘法，则有递推：

$$
T(n) = k \cdot T\!\left(\frac{n}{3}\right) + \Theta(n^2),
$$

由主定理可得：

$$
T(n) = \Theta\!\big(n^{\log_3 k}\big).
$$
要满足时间为 $o\!\big(n^{\lg 7}\big)$ , 需要$\log_3 k < \lg 7 \quad \Longleftrightarrow \quad k < 3^{\lg 7}$, 而$3^{\lg 7} \approx 21.8$
因此最大的整数$k = 21$

运行时间为

$$
T(n) = \Theta\!\big(n^{\log_3 21}\big) \approx \Theta\!\big(n^{2.771}\big),
$$

严格优于

$$
\Theta\!\big(n^{\lg 7}\big) \approx \Theta\!\big(n^{2.807}\big).
$$

# 4.5-5
![[Pasted image 20251012195627.png]]
取

$$
a=2, \quad b=2, \quad \varepsilon=1, \quad f(n)=n^2\,(2+\sin\ln n).
$$

#### 验证符合主定理第三种情况(不包括正则条件)：
因为 $2+\sin\ln n\in[1,3]$，故 $f(n)\ge n^{\log_b a+\varepsilon}=n^2$，因此

$$
f(n)=\Omega\big(n^{\log_b a+\varepsilon}\big)=\Omega(n^2),
$$


#### 检验正则条件：

$$
a\,f\!\left(\frac{n}{b}\right)=2\left(\frac{n}{2}\right)^2\big(2+\sin(\ln n-\ln 2)\big)=\frac{1}{2}\,n^2\big(2+\sin(\ln n-\ln 2)\big).
$$

于是

$$
\frac{a\,f(n/b)}{f(n)}=\frac{1}{2}\cdot\frac{2+\sin(\ln n-\ln 2)}{2+\sin\ln n}.
$$

由于 $\ln 2/(2\pi)$ 是无理数，故序列 $\{m\ln 2\}_{m\in\mathbb N}$ 在模 $2\pi$ 意义下稠密
故存在无穷多个 $n=2^m$ 使得 $\sin\ln n=\sin (m\ln 2)<0$ 而 $\sin(\ln n-\ln 2)=\sin((m-1)\ln2)>0$，从而

$$
\frac{a\,f(n/b)}{f(n)}>1.
$$

因此不存在常数 $c<1$ 使得 $a\,f(n/b)\le c\,f(n)$ 对所有充分大的 $n$ 成立，正则条件不满足
除了该条件外，其它条件均满足

# 4-1 Recurrence examples
给出尽可能紧的上下界，并简单说明理由。设 $T(n)$ 在 $n\le 2$ 时为常数。

a) $$T(n)=2T(n/2)+n^4$$
- $a=2, b=2,\; n^{\log_b a}=n$；$f(n)=n^4$ 多项式地大于 $n$。
- 正则性：$a f(n/b)=2(n/2)^4=n^4/8\le c f(n)$（取 $c=1/8$）。
- 结论：$$T(n)=\Theta(n^4).$$

b) $$T(n)=T(7n/10)+n$$
- 深度约为 $\log_{10/7} n$，每层代价几何衰减，总和为常数倍的 $n$。
- 上、下界都由顶层的 $n$ 控制。
- 结论：$$T(n)=\Theta(n).$$

c) $$T(n)=16T(n/4)+n^2$$
- $a=16, b=4,\; n^{\log_b a}=n^{\log_4 16}=n^2$，与 $f(n)$ 同阶，为主定理第 2 种情况。
- 结论：$$T(n)=\Theta\big(n^2\log n\big).$$

d) $$T(n)=7T(n/3)+n^2$$
- $a=7, b=3,\; n^{\log_b a}=n^{\log_3 7}\approx n^{1.771}$，$f(n)=n^2$ 多项式地更大；
- 正则性：$a f(n/b)=7(n/3)^2=\tfrac{7}{9}n^2\le c n^2$（取 $c=7/9$）。
- 结论：$$T(n)=\Theta(n^2).$$

e) $$T(n)=7T(n/2)+n^2$$
- $a=7, b=2,\; n^{\log_b a}=n^{\log_2 7}\approx n^{2.807}$，$f(n)=n^2$ 多项式地更小，为第 1 种情况。
- 结论：$$T(n)=\Theta\big(n^{\log_2 7}\big).$$

f) $$T(n)=2T(n/4)+\sqrt{n}$$
- $a=2, b=4,\; n^{\log_b a}=n^{\log_4 2}=n^{1/2}$，与 $f(n)$ 同阶，为第 2 种情况。
- 结论：$$T(n)=\Theta\big(\sqrt{n}\,\log n\big).$$

g) $$T(n)=T(n-2)+n^2$$
- 展开求和：$$T(n)=\Theta\!\left(\sum_{i=0}^{\lfloor n/2\rfloor}(n-2i)^2\right)=\Theta\!\left(\sum_{j=1}^{n} j^2\right)=\Theta(n^3).$$
- 结论：$$T(n)=\Theta(n^3).$$
