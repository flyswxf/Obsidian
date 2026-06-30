考虑下面文法，构造分别构造LR(1)分析表和LALR分析表

## 1. 文法一

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

### 1.1 构造 LR(1) 项目集族

计算 FIRST 集：$FIRST(F) = \{a, b\}$，$FIRST(T) = \{a, b\}$，$FIRST(E) = \{a, b\}$。

**$I_0$**:
$S' \to \cdot E, \$$
$E \to \cdot E + T, \{+, \$\}$
$E \to \cdot T, \{+, \$\}$
$T \to \cdot T F, \{+, \$, a, b\}$
$T \to \cdot F, \{+, \$, a, b\}$
$F \to \cdot F *, \{+, \$, a, b, *\}$
$F \to \cdot a, \{+, \$, a, b, *\}$
$F \to \cdot b, \{+, \$, a, b, *\}$

**$I_1 = \text{Goto}(I_0, E)$**:
$S' \to E \cdot, \$$
$E \to E \cdot + T, \{+, \$\}$

**$I_2 = \text{Goto}(I_0, T)$**:
$E \to T \cdot, \{+, \$\}$
$T \to T \cdot F, \{+, \$, a, b\}$
$F \to \cdot F *, \{+, \$, a, b, *\}$
$F \to \cdot a, \{+, \$, a, b, *\}$
$F \to \cdot b, \{+, \$, a, b, *\}$

**$I_3 = \text{Goto}(I_0, F) = \text{Goto}(I_6, F)$**:
$T \to F \cdot, \{+, \$, a, b\}$
$F \to F \cdot *, \{+, \$, a, b, *\}$

**$I_4 = \text{Goto}(I_0, a) = \text{Goto}(I_2, a) = \text{Goto}(I_6, a) = \text{Goto}(I_9, a)$**:
$F \to a \cdot, \{+, \$, a, b, *\}$

**$I_5 = \text{Goto}(I_0, b) = \text{Goto}(I_2, b) = \text{Goto}(I_6, b) = \text{Goto}(I_9, b)$**:
$F \to b \cdot, \{+, \$, a, b, *\}$

**$I_6 = \text{Goto}(I_1, +)$**:
$E \to E + \cdot T, \{+, \$\}$
$T \to \cdot T F, \{+, \$, a, b\}$
$T \to \cdot F, \{+, \$, a, b\}$
$F \to \cdot F *, \{+, \$, a, b, *\}$
$F \to \cdot a, \{+, \$, a, b, *\}$
$F \to \cdot b, \{+, \$, a, b, *\}$

**$I_7 = \text{Goto}(I_2, F) = \text{Goto}(I_9, F)$**:
$T \to T F \cdot, \{+, \$, a, b\}$
$F \to F \cdot *, \{+, \$, a, b, *\}$

**$I_8 = \text{Goto}(I_3, *) = \text{Goto}(I_7, *)$**:
$F \to F * \cdot, \{+, \$, a, b, *\}$

**$I_9 = \text{Goto}(I_6, T)$**:
$E \to E + T \cdot, \{+, \$\}$
$T \to T \cdot F, \{+, \$, a, b\}$
$F \to \cdot F *, \{+, \$, a, b, *\}$
$F \to \cdot a, \{+, \$, a, b, *\}$
$F \to \cdot b, \{+, \$, a, b, *\}$

### 1.2 构造分析表

在上述构造的 10 个 LR(1) 状态中，不存在任何两个状态具有相同的核心。由于 LALR(1) 分析表是通过合并具有相同核心的 LR(1) 状态而来的，因此对于本文法，LALR(1) 分析表与 LR(1) 分析表完全相同。

**LR(1) 及 LALR(1) 分析表：**

|  状态   |   +   |   *   |   a   |   b   |      $       |  E  |  T  |  F  |
| :---: | :---: | :---: | :---: | :---: | :----------: | :-: | :-: | :-: |
| **0** |       |       | $s_4$ | $s_5$ |              |  1  |  2  |  3  |
| **1** | $s_6$ |       |       |       | $\text{acc}$ |     |     |     |
| **2** | $r_2$ |       | $s_4$ | $s_5$ |    $r_2$     |     |     |  7  |
| **3** | $r_4$ | $s_8$ | $r_4$ | $r_4$ |    $r_4$     |     |     |     |
| **4** | $r_6$ | $r_6$ | $r_6$ | $r_6$ |    $r_6$     |     |     |     |
| **5** | $r_7$ | $r_7$ | $r_7$ | $r_7$ |    $r_7$     |     |     |     |
| **6** |       |       | $s_4$ | $s_5$ |              |     |  9  |  3  |
| **7** | $r_3$ | $s_8$ | $r_3$ | $r_3$ |    $r_3$     |     |     |     |
| **8** | $r_5$ | $r_5$ | $r_5$ | $r_5$ |    $r_5$     |     |     |     |
| **9** | $r_1$ |       | $s_4$ | $s_5$ |    $r_1$     |     |     |  7  |

---

## 2. 文法二

$$
\begin{aligned}
S &\to A a A b \mid B b B a \\
A &\to \epsilon \\
B &\to \epsilon
\end{aligned}
$$

### 拓广并编号
(0) $S' \to S$
(1) $S \to A a A b$
(2) $S \to B b B a$
(3) $A \to \epsilon$
(4) $B \to \epsilon$

### 2.1 构造 LR(1) 项目集族

计算 FIRST 集：$FIRST(A) = \{\epsilon\}$，$FIRST(B) = \{\epsilon\}$。

**$I_0$**:
$S' \to \cdot S, \$$
$S \to \cdot A a A b, \$$
$S \to \cdot B b B a, \$$
$A \to \cdot, a$
$B \to \cdot, b$

**$I_1 = \text{Goto}(I_0, S)$**:
$S' \to S \cdot, \$$

**$I_2 = \text{Goto}(I_0, A)$**:
$S \to A \cdot a A b, \$$

**$I_3 = \text{Goto}(I_0, B)$**:
$S \to B \cdot b B a, \$$

**$I_4 = \text{Goto}(I_2, a)$**:
$S \to A a \cdot A b, \$$
$A \to \cdot, b$

**$I_5 = \text{Goto}(I_3, b)$**:
$S \to B b \cdot B a, \$$
$B \to \cdot, a$

**$I_6 = \text{Goto}(I_4, A)$**:
$S \to A a A \cdot b, \$$

**$I_7 = \text{Goto}(I_5, B)$**:
$S \to B b B \cdot a, \$$

**$I_8 = \text{Goto}(I_6, b)$**:
$S \to A a A b \cdot, \$$

**$I_9 = \text{Goto}(I_7, a)$**:
$S \to B b B a \cdot, \$$

### 2.2 构造分析表

在这 10 个 LR(1) 状态中，没有任何两个状态拥有相同的核心。因此，本项目集族无需且无法进行任何状态合并。本文法的 LALR(1) 分析表与 LR(1) 分析表完全相同。

**LR(1) 及 LALR(1) 分析表：**

| 状态  |   a   |   b   |      $       |   S   |   A   |   B   |
| :---: | :---: | :---: | :----------: | :---: | :---: | :---: |
| **0** | $r_3$ | $r_4$ |              |   1   |   2   |   3   |
| **1** |       |       | $\text{acc}$ |       |       |       |
| **2** | $s_4$ |       |              |       |       |       |
| **3** |       | $s_5$ |              |       |       |       |
| **4** |       | $r_3$ |              |       |   6   |       |
| **5** | $r_4$ |       |              |       |       |   7   |
| **6** |       | $s_8$ |              |       |       |       |
| **7** | $s_9$ |       |              |       |       |       |
| **8** |       |       |    $r_1$     |       |       |       |
| **9** |       |       |    $r_2$     |       |       |       |