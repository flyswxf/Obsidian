一个上下文无关文法（CFG）被称为**二义性文法**，如果存在至少一个句子，可以对应**两棵或更多棵不同的语法树**。

等价地，如果一个句子能找到**两个不同的最左推导**（或两个不同的最右推导），则该文法是二义性的。

> 注意：必须是**同类型**的两个不同推导才能证明二义性。一个最左推导和一个最右推导不同是正常现象，不能用来证明二义性。

## 2. 判定方法

要证明一个文法是二义性的，只需找到一个句子，使得该句子可以：
- 由两棵不同的语法树生成，或者
- 由两个不同的最左推导得到，或者
- 由两个不同的最右推导得到

反过来，要证明一个文法是**无二义性**的，则需要证明每个句子都只有一棵唯一的语法树——这通常更加困难。

## 3. 例子：证明文法是二义性的

考虑文法：
$$
S \rightarrow aSbS \mid bSaS \mid \epsilon
$$

选择句子 `abab`，构造两个不同的最左推导：

**最左推导 1**：将第一个 $S$ 直接消为 $\epsilon$
$$
S \Rightarrow aSbS \Rightarrow abS \Rightarrow abaSbS \Rightarrow ababS \Rightarrow abab
$$

**最左推导 2**：将第一个 $S$ 继续展开为 $bSaS$
$$
S \Rightarrow aSbS \Rightarrow abSaSbS \Rightarrow abaSbS \Rightarrow ababS \Rightarrow abab
$$

两个推导在第二步时产生了分歧（一个用 $S \rightarrow \epsilon$，另一个用 $S \rightarrow bSaS$），因此该文法是二义性的。

## 4. 实际意义

在编译器设计中，二义性文法是有问题的——编译器面对一个句子时，不知道该用哪棵语法树来解析，就会产生歧义。实际编译器中通常需要通过**改写文法**或引入**优先级/结合性规则**来消除二义性。

经典的例子是算术表达式文法 $E \rightarrow E + E \mid E * E \mid (E) \mid id$，它对 $a + b * c$ 是二义的（先算加法还是乘法？）。通过[[二义性文法(Ambiguous Grammar)#改写文法消除二义性：|改写文法]]或[[二义性文法(Ambiguous Grammar)#声明运算符优先级|声明运算符优先级]]即可消除二义。

### 改写文法消除二义性：
通过引入新的非终结符（$T$ 代表项，$F$ 代表因子）来强制规定运算符的优先级（乘法高于加法）和结合性，可将其改写为等价的无二义性文法：

$$
\begin{aligned}
E \rightarrow E + E \mid E * E \mid (E) \mid id \quad \Rightarrow \quad E &\rightarrow E + T \mid T \\
T &\rightarrow T * F \mid F \\
F &\rightarrow (E) \mid id
\end{aligned}
$$


### 声明运算符优先级
在 Yacc / Bison 等工具中，我们通过在文法之外**单独声明运算符的优先级（Precedence）和结合性（Associativity）**，来人为地“偏袒”某一种动作，从而打破僵局。最终，分析器在运行时依然是确定的、无二义性的。