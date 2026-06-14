为了进行模型检测，系统必须被形式化为一个 **[[Kripke结构|Kripke 结构]]** $\mathcal{K}$：
$$ \mathcal{K} = \langle S, \rightarrow_{\mathcal{K}}, L \rangle $$

1.  **$S$ (Set of States)**: 所有可能状态的集合。
2.  **$\rightarrow_{\mathcal{K}}$ (Transition Relation)**: 状态转换关系 ($S \times S$)。
    *   **关键特性 - Totality (完全性)**: 关系必须是完全的。即对于每一个状态 $s \in S$，都必须存在一个后继状态 $s'$ 使得 $s \rightarrow_{\mathcal{K}} s'$。
    *   *直观理解*: 系统不能有“死胡同”（死锁）。如果系统设计中有终止状态，通常通过添加指向自身的自环 (Self-loop) 来满足完全性。
3.  **$L$ (Labeling Function)**: 标记函数 $L: S \rightarrow 2^{AP}$。
    *   它将每个状态映射到一组在该状态下为 **True** 的**原子命题 (Atomic Propositions, AP)**。
    *   例如: $L(s_{on}) = \{on\}$ 表示在 $s_{on}$ 状态下，命题 "on" 是真的。
