## 霍尔三元组 (Hoare Triple)

为了表达“如果程序片段 $P$ 的执行开始于满足 $\Phi$ 的状态，那么 $P$ 的执行将结束于满足 $\Psi$ 的状态”，我们引入**霍尔三元组 (Hoare triple)**，记作：

$$
(\!| \Phi |\!) P (\!| \Psi |\!)
$$

*   $\Phi$ 被称为**前置条件 (precondition)**。
*   $\Psi$ 被称为**后置条件 (postcondition)**。

**示例**：
程序 $P$ 的规范是“计算一个平方小于 $x$ 的数”。
$$ (\!| x > 0 |\!) P (\!| y \cdot y < x |\!) $$
这意味着：如果从 $x > 0$ 的状态开始执行，那么 $P$ 的执行将结束于 $y^2 < x$ 的状态。如果初始 $x \le 0$，则行为未定义。

## 部分正确性 vs 完全正确性

*   **部分正确性 (Partial correctness)**：**不要求**程序终止。如果执行了，并且终止了，那么后置条件必须成立。记作：$\models_{par} (\!| \Phi |\!) P (\!| \Psi |\!)$。
*   **完全正确性 (Total correctness)**：**要求**程序终止。如果前置条件满足，程序**保证终止**且后置条件成立。记作：$\models_{tot} (\!| \Phi |\!) P (\!| \Psi |\!)$。

## 程序变量与逻辑变量

当程序的初始值在执行过程中被覆盖时，需要引入逻辑变量（如 $x_0$）来记录初始状态。
$$ \models_{tot} (\!| x = x_0 \land x \ge 0 |\!) \text{ Fac2 } (\!| y = x_0! |\!) $$

## 部分正确性的证明演算规则

1.  **Composition (顺序组合)**
    $$ \frac{(\!| \phi |\!) C_1 (\!| \eta |\!) \quad (\!| \eta |\!) C_2 (\!| \psi |\!)}{(\!| \phi |\!) C_1; C_2 (\!| \psi |\!)} $$

2.  **Assignment (赋值)**
    $$ \overline{(\!| \psi[E/x] |\!) x = E (\!| \psi |\!)} $$
    **逆向推理**：为了使 $x = E$ 执行后满足 $\psi$，前置条件必须是 $\psi$ 中所有的 $x$ 被 $E$ 替换。

3.  **Implied / Consequence (蕴含/推论)**
    $$ \frac{\vdash \phi' \rightarrow \phi \quad (\!| \phi |\!) C (\!| \psi |\!) \quad \vdash \psi \rightarrow \psi'}{(\!| \phi' |\!) C (\!| \psi' |\!)} $$
    允许加强前置条件或减弱后置条件。

4.  **If-statement (条件语句)**
    $$ \frac{(\!| \phi \land B |\!) C_1 (\!| \psi |\!) \quad (\!| \phi \land \neg B |\!) C_2 (\!| \psi |\!)}{(\!| \phi |\!) \text{if } B \{C_1\} \text{ else } \{C_2\} (\!| \psi |\!)} $$

5.  **Partial-while (While 循环)**
    $$ \frac{(\!| \psi \land B |\!) C (\!| \psi |\!)}{(\!| \psi |\!) \text{while } B \{C\} (\!| \psi \land \neg B |\!)} $$
    依赖于**循环不变量 (loop invariant)** $\psi$。

6.  **Total-while (Total Correctness)**
    $$ \frac{(\!| \eta \land B \land 0 \le E = E_0 |\!) C (\!| \eta \land 0 \le E < E_0 |\!)}{(\!| \eta \land 0 \le E |\!) \text{while } B \{C\} (\!| \eta \land \neg B |\!)} $$
    要求证明循环终止，需引入**变体 (variant)** 表达式 $E$（非负整数，每次迭代严格减小）。

## 证明表格 (Proof Tableaux)

一种方便展示程序逻辑证明的方法，每一步状态转换都对应一个证明规则。

```text
       (| Φ0 |)
C1;
       (| Φ1 |)    理由 (justification)
C2;
       (| Φ2 |)    理由 (justification)
```

### 示例 1: 赋值语句

证明 $\vdash_{par} (\!| y = 5 |\!) x = y + 1 (\!| x = 6 |\!)$
```text
       (| y = 5 |)
       (| y + 1 = 6 |)    Implied (由 y=5 推导 5+1=6)
x = y + 1;
       (| x = 6 |)        Assignment
```

### 示例 2: 循环 - Total Correctness (阶乘计算)

选择变体 $E = x - z$。

```text
       (| x >= 0 |)
       (| 1 = 0! /\ 0 <= x - 0 |)            Implied
y = 1;
       (| y = 0! /\ 0 <= x - 0 |)            Assignment
z = 0;
       (| y = z! /\ 0 <= x - z |)            Assignment
while (x != z) {
       (| y = z! /\ x != z /\ 0 <= x - z = E0 |)      Invariant Hyp. /\ guard
       (| y * (z + 1) = (z + 1)! /\ 0 <= x - (z + 1) < E0 |) Implied (变体 E = x-z 减小)
    z = z + 1;
       (| y * z = z! /\ 0 <= x - z < E0 |)            Assignment
    y = y * z;
       (| y = z! /\ 0 <= x - z < E0 |)                Assignment
}
       (| y = z! /\ x = z |)                 Total-while
       (| y = x! |)                          Implied
```
