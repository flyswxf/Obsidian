## Tseitin 转换概念

将任意命题逻辑公式转换为合取范式 (CNF) 时，如果使用标准算法（如分配律内推），在最坏情况下会导致公式大小呈指数级爆炸。

Tseitin 转换（Tseitin Transformation）是一种用于将任意命题逻辑公式转换为**等可满足 (Equisatisfiable)** 的 CNF 公式的算法。其核心优势在于转换后的公式大小与原公式大小呈**线性关系**。

> **等可满足性 (Equisatisfiability)**：转换后的公式与原公式在逻辑上不一定等价，但如果原公式是可满足的，转换后的公式也是可满足的；反之亦然。

## 转换算法流程

Tseitin 转换通过为原公式解析树中的每个内部节点（即每个子公式）引入一个新的命题变量来实现。

### 1. 引入新变量
对于给定的公式 $\Phi$，为其每一个子公式分配一个新的变量。
例如，对于公式 $\phi = (p \lor q) \land r$：
* 为子公式 $p \lor q$ 引入新变量 $x_1$，即 $x_1 \leftrightarrow (p \lor q)$。
* 为整个公式 $(p \lor q) \land r$ 引入新变量 $x_2$，即 $x_2 \leftrightarrow (x_1 \land r)$。

### 2. 构建约束条件
将每个新变量与其对应的子公式之间的等价关系（双蕴含 $\leftrightarrow$）转换为 CNF 子句。这些子句被称为**约束条件**。

常见逻辑操作的 CNF 约束转换规则：
* **合取 $x \leftrightarrow (y \land z)$**:
  转换为：$(\neg x \lor y) \land (\neg x \lor z) \land (\neg y \lor \neg z \lor x)$
* **析取 $x \leftrightarrow (y \lor z)$**:
  转换为：$(\neg y \lor x) \land (\neg z \lor x) \land (\neg x \lor y \lor z)$
* **否定 $x \leftrightarrow \neg y$**:
  转换为：$(\neg x \lor \neg y) \land (x \lor y)$
* **蕴含 $x \leftrightarrow (y \rightarrow z)$**:
  转换为：$(\neg x \lor \neg y \lor z) \land (y \lor x) \land (\neg z \lor x)$

### 3. 生成最终公式
最终的 CNF 公式由代表整个原公式根节点的新变量，与所有构建的约束条件子句进行**合取**得到。

## 完整转换示例

将公式 $\phi = (p \lor q) \land r$ 转换为 CNF。

**步骤 1：引入新变量**
* $x_1 \leftrightarrow (p \lor q)$
* $x_2 \leftrightarrow (x_1 \land r)$

**步骤 2：转换约束条件为 CNF**
* 对于 $x_1 \leftrightarrow (p \lor q)$：
  等价于 $(\neg p \lor x_1) \land (\neg q \lor x_1) \land (\neg x_1 \lor p \lor q)$
* 对于 $x_2 \leftrightarrow (x_1 \land r)$：
  等价于 $(\neg x_2 \lor x_1) \land (\neg x_2 \lor r) \land (\neg x_1 \lor \neg r \lor x_2)$

**步骤 3：生成最终公式**
根节点变量为 $x_2$。最终的 CNF 公式为：
$$ x_2 \land (\neg p \lor x_1) \land (\neg q \lor x_1) \land (\neg x_1 \lor p \lor q) \land (\neg x_2 \lor x_1) \land (\neg x_2 \lor r) \land (\neg x_1 \lor \neg r \lor x_2) $$
