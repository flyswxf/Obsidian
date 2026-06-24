语法制导翻译方案（Syntax-Directed Translation Scheme, SDT）是[[语法制导定义(SDD)]]的具体实现机制。与仅仅描述逻辑依赖的 SDD 不同，SDT 是一种在产生式右部显式嵌入**语义动作**（Semantic Actions）的上下文无关文法。

这些被花括号 `{ }` 包裹的程序代码片段，指明了属性计算的确切时机。当语法分析器匹配到产生式中动作所在的位置时，就会执行该动作。

## SDT的实现策略

根据嵌入动作的位置和分析方法的不同，SDT 的实现策略通常分为两类：

### 后缀 SDT (Postfix SDT)

在后缀 SDT 中，所有的语义动作都被放置在产生式右部的**最末尾**。
- **匹配文法**：这种方案完美契合 [[语法制导定义(SDD)#S-属性定义 (S-Attributed SDD)|S-属性定义（S-Attributed SDD）]]。
- **实现方式**：非常适合在**自底向上分析**（如 [[语法分析器#4.2 LR 分析法|LR 分析器]]）中实现。当分析器将产生式右部规约（Reduce）为左部非终结符时，执行动作代码。此时右部符号的综合属性已存在于分析栈中，可直接出栈参与计算。

### L-属性 SDT

L-属性 SDT 是为了实现 [[语法制导定义(SDD)#L-属性定义 (L-Attributed SDD)|L-属性定义（L-Attributed SDD）]]而设计的翻译方案。其动作可以出现在产生式右部的**任意位置**。
- **放置规则**：
  1. 计算某符号[[属性文法#继承属性 (Inherited Attributes)|继承属性]]的动作，必须放置在该符号**之前**。
  2. 计算产生式左部符号[[属性文法#综合属性 (Synthesized Attributes)|综合属性]]的动作，必须放置在产生式**最末尾**。
- **实现方式**：
  - **自顶向下分析**：非常契合 [[LL(1)分析法|LL 分析器]]（如递归下降分析）。由于是从左到右推导，遇到动作即执行，完全符合继承属性向右传递的流向。
  - **自底向上分析**：在 [[语法分析器#4.2 LR 分析法|LR 分析]]中实现 L-属性 SDT 较为困难，通常需要通过引入**标记非终结符**（Marker Nonterminals）并将其转化为带空产生式的[[语法制导翻译方案(SDT)#后缀 SDT (Postfix SDT)|后缀 SDT]]，利用栈的绝对位置引用来传递继承属性。

## 语义动作的具体示例

在语法制导翻译方案（Translation Scheme）中，语义动作（Semantic Actions）的具体位置决定了属性求值的时机。以下是两种典型场景的例子：

### 示例 1：后缀 SDT 实现表达式求值（S-属性）

这是一个经典的桌面计算器例子。所有的语义动作都位于产生式的最末尾，通过[[属性文法#综合属性 (Synthesized Attributes)|综合属性]] `val` 自底向上计算表达式的值：

$$
\begin{aligned}
L &\rightarrow E \textbf{n} \quad &\{ \text{print}(E.val); \} \\
E &\rightarrow E_1 + T \quad &\{ E.val = E_1.val + T.val; \} \\
E &\rightarrow T \quad &\{ E.val = T.val; \} \\
T &\rightarrow T_1 * F \quad &\{ T.val = T_1.val \times F.val; \} \\
T &\rightarrow F \quad &\{ T.val = F.val; \} \\
F &\rightarrow ( E ) \quad &\{ F.val = E.val; \} \\
F &\rightarrow \text{digit} \quad &\{ F.val = \text{digit}.lexval; \}
\end{aligned}
$$

*说明*：当语法分析器将句柄规约为对应的非终结符时，立刻执行对应的动作。例如，规约 $F \rightarrow \text{digit}$ 时，从词法单元中提取数值赋给 $F.val$。

### 示例 2：动作嵌入产生式内部的 L-属性 SDT

在处理变量声明时，通常需要将类型信息作为[[属性文法#继承属性 (Inherited Attributes)|继承属性]]传递给后续的标识符。此时语义动作会嵌入在产生式内部：

$$
\begin{aligned}
D &\rightarrow T \quad \{ L.inh = T.type; \} \quad L \\
T &\rightarrow \textbf{int} \quad \{ T.type = \text{integer}; \} \\
T &\rightarrow \textbf{float} \quad \{ T.type = \text{float}; \} \\
L &\rightarrow L_1 , \textbf{id} \quad \{ L_1.inh = L.inh; \text{addType}(\textbf{id}.entry, L.inh); \} \\
L &\rightarrow \textbf{id} \quad \{ \text{addType}(\textbf{id}.entry, L.inh); \}
\end{aligned}
$$

*说明*：在产生式 $D \rightarrow T L$ 中，动作 $\{ L.inh = T.type; \}$ 位于 $T$ 和 $L$ 之间。当分析器完成 $T$ 的推导并得到其 `type` 属性后，立刻执行该动作，将类型信息赋予 $L$ 的继承属性 `inh`，以便后续在处理 $L$（如标识符列表）时将其写入[[符号表]]。

通过合理编排 SDT，语义分析器可以在生成语法树的同时，完成[[符号表]]的构建与[[类型检查]]等上下文验证。