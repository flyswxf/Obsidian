## 1. 考虑下面文法，构造 SLR 分析表
$$
\begin{aligned}
E &\to E + T \mid T \\
T &\to T F \mid F \\
F &\to F * \mid a \mid b
\end{aligned}
$$

### 拓广并编号
(0) $S' \to E$
(1) $E \to E + T$
(2) $E \to T$
(3) $T \to T F$
(4) $T \to F$
(5) $F \to F *$
(6) $F \to a$
(7) $F \to b$

### 1.1 计算 FIRST 和 FOLLOW 集合

- $FIRST(F) = \{a, b\}$
- $FIRST(T) = FIRST(F) = \{a, b\}$
- $FIRST(E) = FIRST(T) = \{a, b\}$

- $FOLLOW(E) = \{+, \$\}$
- $FOLLOW(T) = FOLLOW(E) \cup FIRST(F) \cup FOLLOW(T) = \{+, \$, a, b\}$
- $FOLLOW(F) = \{*\} \cup FOLLOW(T) = \{+, \$, a, b, *\}$

### 1.2 构造 LR(0) 项目集族

**$I_0$**:
$S' \to \cdot E$
$E \to \cdot E + T$
$E \to \cdot T$
$T \to \cdot T F$
$T \to \cdot F$
$F \to \cdot F *$
$F \to \cdot a$
$F \to \cdot b$

**$I_1 = \text{Goto}(I_0, E)$**:
$S' \to E \cdot$
$E \to E \cdot + T$

**$I_2 = \text{Goto}(I_0, T)$**:
$E \to T \cdot$
$T \to T \cdot F$
$F \to \cdot F *$
$F \to \cdot a$
$F \to \cdot b$

**$I_3 = \text{Goto}(I_0, F) = \text{Goto}(I_6, F)$**:
$T \to F \cdot$
$F \to F \cdot *$

**$I_4 = \text{Goto}(I_0, a) = \text{Goto}(I_2, a) = \text{Goto}(I_6, a) = \text{Goto}(I_9, a)$**:
$F \to a \cdot$

**$I_5 = \text{Goto}(I_0, b) = \text{Goto}(I_2, b) = \text{Goto}(I_6, b) = \text{Goto}(I_9, b)$**:
$F \to b \cdot$

**$I_6 = \text{Goto}(I_1, +)$**:
$E \to E + \cdot T$
$T \to \cdot T F$
$T \to \cdot F$
$F \to \cdot F *$
$F \to \cdot a$
$F \to \cdot b$

**$I_7 = \text{Goto}(I_2, F) = \text{Goto}(I_9, F)$**:
$T \to T F \cdot$
$F \to F \cdot *$

**$I_8 = \text{Goto}(I_3, *) = \text{Goto}(I_7, *)$**:
$F \to F * \cdot$

**$I_9 = \text{Goto}(I_6, T)$**:
$E \to E + T \cdot$
$T \to T \cdot F$
$F \to \cdot F *$
$F \to \cdot a$
$F \to \cdot b$

### 1.3 构造 SLR 分析表

根据 LR(0) 项目集族和 FOLLOW 集合构造 SLR 分析表：
- 状态 2 中的归约项目 $E \to T \cdot$ ，仅在 $FOLLOW(E) = \{+, \$\}$ 下归约 $r_2$。
- 状态 3 中的归约项目 $T \to F \cdot$ ，仅在 $FOLLOW(T) = \{+, \$, a, b\}$ 下归约 $r_4$。
- 状态 4 中的归约项目 $F \to a \cdot$ ，在 $FOLLOW(F) = \{+, \$, a, b, *\}$ 下归约 $r_6$。
- 状态 5 中的归约项目 $F \to b \cdot$ ，在 $FOLLOW(F) = \{+, \$, a, b, *\}$ 下归约 $r_7$。
- 状态 7 中的归约项目 $T \to T F \cdot$ ，在 $FOLLOW(T) = \{+, \$, a, b\}$ 下归约 $r_3$。
- 状态 8 中的归约项目 $F \to F * \cdot$ ，在 $FOLLOW(F) = \{+, \$, a, b, *\}$ 下归约 $r_5$。
- 状态 9 中的归约项目 $E \to E + T \cdot$ ，在 $FOLLOW(E) = \{+, \$\}$ 下归约 $r_1$。

**SLR(1) 分析表：**

| 状态 | + | * | a | b | $ | E | T | F |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | | | $s_4$ | $s_5$ | | 1 | 2 | 3 |
| **1** | $s_6$ | | | | $\text{acc}$ | | | |
| **2** | $r_2$ | | $s_4$ | $s_5$ | $r_2$ | | | 7 |
| **3** | $r_4$ | $s_8$ | $r_4$ | $r_4$ | $r_4$ | | | |
| **4** | $r_6$ | $r_6$ | $r_6$ | $r_6$ | $r_6$ | | | |
| **5** | $r_7$ | $r_7$ | $r_7$ | $r_7$ | $r_7$ | | | |
| **6** | | | $s_4$ | $s_5$ | | | 9 | 3 |
| **7** | $r_3$ | $s_8$ | $r_3$ | $r_3$ | $r_3$ | | | |
| **8** | $r_5$ | $r_5$ | $r_5$ | $r_5$ | $r_5$ | | | |
| **9** | $r_1$ | | $s_4$ | $s_5$ | $r_1$ | | | 7 |

---

## 2. 对下面文法，证明它是 LL(1) 文法，但不是 SLR 文法。
$$
\begin{aligned}
S &\to A a A b \mid B b B a \\
A &\to \epsilon \\
B &\to \epsilon
\end{aligned}
$$

### 2.1 证明它是 LL(1) 文法

计算 FIRST 集：
- $FIRST(A) = \{\epsilon\}$
- $FIRST(B) = \{\epsilon\}$
- $FIRST(A a A b) = \{a\}$
- $FIRST(B b B a) = \{b\}$

计算 FOLLOW 集：
- $FOLLOW(S) = \{\$\}$
- $FOLLOW(A)$：由 $S \to A a A b$ 得到 $FOLLOW(A) = \{a, b\}$。
- $FOLLOW(B)$：由 $S \to B b B a$ 得到 $FOLLOW(B) = \{a, b\}$。

计算产生式的 SELECT 集：
- $SELECT(S \to A a A b) = FIRST(A a A b) = \{a\}$
- $SELECT(S \to B b B a) = FIRST(B b B a) = \{b\}$
由于 $SELECT(S \to A a A b) \cap SELECT(S \to B b B a) = \{a\} \cap \{b\} = \emptyset$，对于 $S$ 的产生式没有冲突。

而对于 $A$ 和 $B$ 均只有一条产生式，无需判断冲突。

**结论**：在任何情况下均能唯一确定选用哪个产生式，因此该文法是 LL(1) 文法。

### 2.2 证明它不是 SLR 文法

为了证明它不是 SLR 文法，构造其 LR(0) 项目集族的初始状态 $I_0$：

**$I_0$**:
$S' \to \cdot S$
$S \to \cdot A a A b$
$S \to \cdot B b B a$
$A \to \cdot$
$B \to \cdot$

在状态 $I_0$ 中，存在两个归约项目：
- $A \to \cdot$
- $B \to \cdot$

在 SLR(1) 分析法中，归约动作取决于产生式左部的 FOLLOW 集。
由前面计算可知：
- $FOLLOW(A) = \{a, b\}$，因此在面临输入符号 $a$ 或 $b$ 时，指示按 $A \to \epsilon$ 归约。
- $FOLLOW(B) = \{a, b\}$，因此在面临输入符号 $a$ 或 $b$ 时，也指示按 $B \to \epsilon$ 归约。

这导致在状态 $I_0$ 下，当向前查看输入符号为 $a$ 或是 $b$ 时，分析器同时面临 $A \to \epsilon$ 和 $B \to \epsilon$ 两种归约选择，从而产生了归约-归约冲突。

**结论**：由于 SLR(1) 无法解决该归约冲突，该文法不是 SLR 文法。