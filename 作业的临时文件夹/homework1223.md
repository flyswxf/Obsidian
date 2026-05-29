## 24.1-4
![[assets/贝尔曼算法负环.png]]
标准的 Bellman-Ford 算法在进行 $|V|-1$ 轮松弛操作后，可以正确计算出没有负权回路的图的单源最短路径。如果图中存在从源点可达的负权回路，标准算法可以通过第 $|V|$ 轮检查来判断是否存在。根据题目要求，只需要修改算法使得在找到负权回路后, 还要将受影响的所有节点的 $v.d$ 设为 $-\infty$。

**修改后的算法步骤**：
1.  **初始化**：`INITIALIZE-SINGLE-SOURCE(G, s)`，将所有顶点的 $d$ 值设为 $\infty$，起点 $s.d=0$。
2.  **松弛**：重复 $|V|-1$ 次，对图中的每条边进行松弛操作 `RELAX(u, v, w)`。
3.  **标记负权回路节点**：
    *   初始化一个空队列 $Q$。
    *   遍历图中的每一条边 $(u, v)$。
    *   如果 $v.d > u.d + w(u, v)$，说明存在负权回路影响。将 $v.d$ 设为 $-\infty$，并将 $v$ 加入 $Q$。
4.  **传播**：
    *   当 $Q$ 不为空时，取出队首元素 $u$。
    *   遍历 $u$ 的所有邻接点 $v$。
    *   为了避免重复计算节点, 只有当 $v.d \neq -\infty$, 也就是该节点还未处理过，才将 $v.d$ 设为 $-\infty$，并将 $v$ 加入 $Q$。
    *   重复该过程直到 $Q$ 为空, 

**伪代码**：
```
MODIFIED-BELLMAN-FORD(G, w, s)
    INITIALIZE-SINGLE-SOURCE(G, s)
    
    // 标准过程
    for i = 1 to |G.V| - 1
        for each edge (u, v) in G.E
            RELAX(u, v, w)
            
    // 额外记录负环上的节点(不一定全)
    Q = empty
    for each edge (u, v) in G.E
        if v.d > u.d + w(u, v)
            v.d = -infinity
            ENQUEUE(Q, v)
            
    // 遍历负环上的节点及其能到达的所有节点
    while Q is not empty
        u = DEQUEUE(Q)
        for each vertex v in G.Adj[u]
            if v.d != -infinity
                v.d = -infinity
                ENQUEUE(Q, v)
```

## 24.3-4
![[assets/负环处理.png]]
正确的最短路径树必须满足以下条件：
1.  **源点属性**：$s.d = 0$ 且 $s.\pi = \text{NIL}$。
2.  **树边的一致性**：对于任意顶点 $v \neq s$，如果 $v.\pi \neq \text{NIL}$，则必须满足 $v.d = v.\pi.d + w(v.\pi, v)$。这保证了树上的路径权重与记录的距离一致。
3.  **松弛性质**：对于图中的每一条边 $(u, v) \in E$，必须满足 $v.d \le u.d + w(u, v)$。这保证了没有更短的路径存在。
4.  **路径可达性**：对于任意 $v \neq s$，如果 $v.d < \infty$，则 $v.\pi$ 不应为 $\text{NIL}$ , 除非 $v=s$。
5.  **无环性**：$\pi$ 属性构成的必须是一棵树，不能包含环。由于边权非负，如果满足条件 2 和 3，且所有边权均为正，则不可能存在环。如果存在 0 权边，则需额外检查 $\pi$ 指针是否构成环。

**验证算法步骤**：
1.  **检查源点**：$O(1)$
    *   若 $s.d \neq 0$ 或 $s.\pi \neq \text{NIL}$，返回 `False`。
2.  **检查顶点属性与树边一致性**：$O(V)$
    *   遍历每个顶点 $v \in V \setminus \{s\}$：
        *   若 $v.\pi \neq \text{NIL}$：检查是否 $v.d = v.\pi.d + w(v.\pi, v)$。如果不等，返回 `False`。
        *   若 $v.\pi = \text{NIL}$：检查是否 $v.d = \infty$。如果不等，返回 `False`。
3.  **检查松弛性质**：$O(E)$
    *   遍历每条边 $(u, v) \in E$：
        *   检查是否 $v.d \le u.d + w(u, v)$。如果不满足，返回 `False`。
4.  **若存在 0 权边, 检查 $\pi$ 是否构成环：$O(V)$
    *   使用着色法（DFS）检查。

该验证算法能够在 $O(V+E)$ 时间内判断输出是否构成一棵合法的最短路径树。

## 24.3-6
![[assets/最可靠路径问题.png]]

要求最大化路径 $p = \langle v_0, v_1, \dots, v_k \rangle$ 的可靠性乘积 $\prod_{i=1}^k r(v_{i-1}, v_i)$。
由于乘积最大化问题可以通过取对数转化为加法问题：
$$ \log(\prod_{i=1}^k r(v_{i-1}, v_i)) = \sum_{i=1}^k \log(r(v_{i-1}, v_i)) $$
因为 $0 \le r(u, v) \le 1$，所以 $\log(r(u, v)) \le 0$。为了使用 Dijkstra 算法求单源最短路, 需要边权非负，因此定义新的边权重为：
$$ w(u, v) = -\log(r(u, v)) $$
此时 $w(u, v) \ge 0$。最大化 $\sum \log(r)$ 等价于最小化 $\sum -\log(r)$，即最小化 $\sum w(u, v)$。

于是问题转化为在一个非负权重的图中寻找最短路径。

算法首先遍历所有边, 将其权重替换为 $w(u, v)$ ($O(E)$)。初始化需要将起点的距离设置为$-\log(1)=0$ , 之后正常运行 Dijkstra 算法，计算起点到目标点的最短路径即可($O(E \log V)$)。

算法总时间复杂度为 $O(E \log V)$。