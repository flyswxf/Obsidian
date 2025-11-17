- 目标：在数组 `A[p..q]` 中找到第 `i` 小的元素。
- 思想：随机选主元，原地划分，只在更小的一侧继续查找。
- 复杂度：期望时间 `O(n)`，最坏 `O(n^2)`；期望递归深度 `O(log n)`，最坏 `O(n)`；额外空间 `O(1)`（原地分区）。

## 伪代码
```
Rand-Select(A, p, q, i)
    if p = q then return A[p]
    r ← Rand-Partition(A, p, q)      // 随机主元，返回最终位置 r
    k ← r - p + 1                     // A[r] 在 A[p..q] 中的位次
    if i = k then return A[r]
    if i < k then
        return Rand-Select(A, p, r-1, i)
    else
        return Rand-Select(A, r+1, q, i - k)
```

### Rand-Partition（随机划分）
- 随机选择一个 `pivot = A[t]`。
- 将区间 `A[p..q]` 原地重排为 `<= pivot | pivot | >= pivot`。
- 返回 `pivot` 的最终下标 `r`。
