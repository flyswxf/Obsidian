## 核心语法 (Syntax)

CTL (Computation Tree Logic) 公式由**原子命题**、**布尔连接词**和**时序连接词**构成。

### BNF 定义
$$
\begin{aligned}
\phi ::= \ & \bot \ | \ \top \ | \ p \ | \ (\neg \phi) \ | \ (\phi \land \phi) \ | \ (\phi \lor \phi) \ | \ (\phi \rightarrow \phi) \\
& | \ AX \phi \ | \ EX \phi \ | \ A[\phi U \phi] \ | \ E[\phi U \phi] \ | \ AG \phi \ | \ EG \phi \ | \ AF \phi \ | \ EF \phi
\end{aligned}
$$

### 时序连接词 (Temporal Connectives)
每个连接词由两个字母组成：**路径量词 (Path Quantifier)** + **时序算子 (Temporal Operator)**。

| 第一字母（广度/选路） | 含义 | 口诀 |
| :--- | :--- | :--- |
| **A** (All) | **所有**路径都满足 | "注定..." |
| **E** (Exists) | **存在**至少一条路径满足 | "有可能..." |

| 第二字母（深度/走路） | 含义 | 口诀 |
| :--- | :--- | :--- |
| **X** (Next) | **下一步**状态 | "明天" |
| **F** (Future) | **未来**某个时刻 | "早晚有一天" |
| **G** (Global) | **永远**（全局） | "从今往后一直" |
| **U** (Until) | **直到**...发生 | "坚持到..." |

### 优先级 (Priorities)
从高到低排列：
1. **一元连接词**：$AX, EX, AG, EG, AF, EF$ 及 $\neg$
2. **与/或**：$\land, \lor$
3. **二元连接词**：$\rightarrow, AU, EU$

---

## 典型实例解析 (Examples)

### 基础组合
- **$AX \phi$**：在**所有**的下一步，$\phi$ 都成立。（注定明天 $\phi$）
- **$EX \phi$**：**存在**一个下一步，$\phi$ 成立。（明天可能 $\phi$）
- **$AG \phi$**：在**所有**未来的**所有**时刻，$\phi$ 总是成立。
	- ![[全局必然.png]]
- **$EG \phi$**：**存在**一条路径，在这条路上 $\phi$ **永远**成立。
	- ![[存在路径永真树.png]]
- **$AF \phi$**：在**所有**路径上，$\phi$ **早晚**会发生。
	- ![[语法与实例.png]]
- **$EF \phi$**：**存在**一条路径，$\phi$ 在**未来**会发生。
	- ![[语法与实例_1.png]]

### 复杂嵌套
- **$A [p U q]$**：在**所有**路径上，$p$ 必须一直成立，**直到** $q$ 发生。
- **$AG (q \rightarrow EG r)$**：
    - **含义**：无论何时 ($AG$)，一旦 $q$ 发生，系统就有可能 ($E$) 进入一个 $r$ 永远成立 ($G$) 的状态。
- **$A [p U EF r]$**：
    - **含义**：$p$ 必须一直维持，直到“$EF r$（有可能实现 $r$）”这件事变成真的。

### 等价变换

#### 取反规则 (Negation / Duality)
**口诀**：当 $\neg$ 移入算子内部时，路径量词和时序算子都要**取反**。
- $\neg AF \phi \equiv EG \neg \phi$
- $\neg EF \phi \equiv AG \neg \phi$
- $\neg AX \phi \equiv EX \neg \phi$

#### 用 Until 表达 Future
**口诀**：$F$ (未来) 是 $U$ (直到) 的特例。即“早晚发生 $\phi$” 等价于 “一直 True，直到 $\phi$ 发生”。
- $AF \phi \equiv A[\top U \phi]$
- $EF \phi \equiv E[\top U \phi]$

---

## 语义模型 (Semantics Model)

CTL 在 [[Kripke结构|Kripke 结构]] $\mathcal{M} = (S, \to, L)$ 上定义：
- **$S$**：状态集合。
- **$\to$**：转换关系（必须是完全的，即每个状态都有后继）。
- **$L$**：标签函数（指出每个状态下哪些原子命题为真）。

### 实例图示

- $S = \{s_0, s_1, s_2\}$
- $\to = \{(s_0, s_1), (s_0, s_2), (s_1, s_0), (s_1, s_2), (s_2, s_2)\}$
- $L(s_0) = \{p, q\}, \ L(s_1) = \{q, r\}, \ L(s_2) = \{r\}$

![[结构实例.png]]

以此可绘制 CTL 树：
![[计算树逻辑树.png]]

**公式验证与判定原因**：

| 公式 | 结果 | 判定原因 |
| :--- | :--- | :--- |
| $\mathcal{M}, s_0 \models p \land q$ | **Yes** | $s_0$ 的标签 $L(s_0)$ 中包含 $p$ 和 $q$。 |
| $\mathcal{M}, s_0 \models EX (q \land r)$ | **Yes** | 存在后继 $s_1$，且 $L(s_1)=\{q, r\}$，满足 $q \land r$。 |
| $\mathcal{M}, s_0 \models \neg AX (q \land r)$ | **Yes** | 并非所有后继都满足 $q \land r$（例如 $s_2$ 只有 $r$，不满足 $q$）。 |
| $\mathcal{M}, s_0 \models \neg EF (p \land r)$ | **Yes** | 无论怎么走，都找不到一个状态既有 $p$ 又有 $r$。 |
| $\mathcal{M}, s_2 \models AG r$ | **Yes** | 从 $s_2$ 出发只有一条路（原地转圈），路上所有点（就是 $s_2$ 自己）都有 $r$。 |
| $\mathcal{M}, s_0 \models AF r$ | **Yes** | 无论走哪条路（去 $s_1$ 还是 $s_2$），下一步都有 $r$，所以早晚会遇到 $r$。 |
| $\mathcal{M}, s_0 \models E[(p \land q) U r]$ | **Yes** | 存在路径 $s_0 \to s_2$，其中 $s_0$ 满足 $p \land q$，直到 $s_2$ 满足 $r$。 |

### 满足性关系 ($\mathcal{M}, s \models \phi$)
- $s \models AX \phi \iff$ 对所有 $s \to s'$，都有 $s' \models \phi$。
- $s \models AG \phi \iff$ 对从 $s$ 开始的所有路径的所有节点，都满足 $\phi$。
- $s \models AF \phi \iff$ 对从 $s$ 开始的所有路径，每条路上都**存在**一个节点满足 $\phi$。