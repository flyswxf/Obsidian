## using 简化类型别名与优先队列写法

```cpp
using P = pair<long long, int>; // {到达时间, 节点}
priority_queue<P, vector<P>, greater<P>> pq;
```

- 这种写法可以节省打字并提升可读性。
- 只是为 `pair<long long, int>` 起了别名 `P`，逻辑不变。
- 之后依然可用常见操作，例如：

```cpp
auto [t, u] = pq.top();     // 结构化绑定，直接解包
pq.emplace(nt, v);          // 原地构造并入队
```