## 合取范式概念

合取范式 (CNF) 是命题逻辑公式的一种标准形式。一个公式是 CNF，当且仅当它是子句的合取，而每个子句是文字的析取。

**CNF 的形式示例**：
- $(p \lor q) \land (\neg p \lor r)$
- $p \land q \land (\neg r \lor s)$

**非 CNF 的示例**：
- $\neg(p \lor q)$ （否定不在文字层面）
- $(p \land q) \lor r$ （合取在内部，析取在外部）

## CNF 转换算法

将任意公式转换为 CNF，主要有两种算法：一种是标准的等价转换算法，另一种是避免指数级爆炸的 [[Tseitin转换]]。

### 1. 标准转换算法 (等价转换)

将任意公式 $\Phi$ 转换为等价的 CNF 的标准算法流程如下：

$$ \text{CNF}(\text{NNF}(\text{IMPL\_FREE}(\Phi))) $$

#### 1.1 IMPL_FREE (消除蕴含)
消除公式中的蕴含 ($\rightarrow$) 和双蕴含 ($\leftrightarrow$)。保留所有 $\neg$，消去所有 $\rightarrow$：
* $\Phi$ 是文字: return $\Phi$
* $\Phi$ 是 $\neg\Phi_1$: return $\neg\text{IMPL\_FREE}(\Phi_1)$
* $\Phi$ 是 $\Phi_1 \land \Phi_2$: return $\text{IMPL\_FREE}(\Phi_1) \land \text{IMPL\_FREE}(\Phi_2)$
* $\Phi$ 是 $\Phi_1 \lor \Phi_2$: return $\text{IMPL\_FREE}(\Phi_1) \lor \text{IMPL\_FREE}(\Phi_2)$
* $\Phi$ 是 $\Phi_1 \rightarrow \Phi_2$: return $\neg\text{IMPL\_FREE}(\Phi_1) \lor \text{IMPL\_FREE}(\Phi_2)$

#### 1.2 NNF (否定范式)
将否定符号内推，直到它们仅出现在原子命题之前。
* $\Phi$ 是文字: return $\Phi$
* $\Phi$ 是 $\neg\neg\Phi_1$: return $\text{NNF}(\Phi_1)$
* $\Phi$ 是 $\Phi_1 \land \Phi_2$: return $\text{NNF}(\Phi_1) \land \text{NNF}(\Phi_2)$
* $\Phi$ 是 $\Phi_1 \lor \Phi_2$: return $\text{NNF}(\Phi_1) \lor \text{NNF}(\Phi_2)$
* $\Phi$ 是 $\neg(\Phi_1 \land \Phi_2)$: return $\text{NNF}(\neg\Phi_1 \lor \neg\Phi_2)$
* $\Phi$ 是 $\neg(\Phi_1 \lor \Phi_2)$: return $\text{NNF}(\neg\Phi_1 \land \neg\Phi_2)$

#### 1.3 CNF & DISTR (利用分配律转换为 CNF)
利用分配律处理析取项，确保析取位于最内层。
* $\Phi$ 是文字: return $\Phi$
* $\Phi$ 是 $\Phi_1 \land \Phi_2$: return $\text{CNF}(\Phi_1) \land \text{CNF}(\Phi_2)$
* $\Phi$ 是 $\Phi_1 \lor \Phi_2$: return $\text{DISTR}(\text{CNF}(\Phi_1), \text{CNF}(\Phi_2))$

**DISTR 函数定义**:
* $\Phi_1$ 是 $\Phi_{11} \land \Phi_{12}$: return $\text{DISTR}(\Phi_{11}, \Phi_2) \land \text{DISTR}(\Phi_{12}, \Phi_2)$
* $\Phi_2$ 是 $\Phi_{21} \land \Phi_{22}$: return $\text{DISTR}(\Phi_1, \Phi_{21}) \land \text{DISTR}(\Phi_1, \Phi_{22})$
* 否则: return $\Phi_1 \lor \Phi_2$

---

#### 标准转换算法完整示例

对公式 $\neg(r \land (\neg(q \rightarrow (\neg p \rightarrow (q \land r)))))$ 的转换过程：

##### 步骤 A: IMPL_FREE
```text
IMPL_FREE(¬(r ∧ (¬(q → (¬p → (q ∧ r)))))) 
 = ¬(r ∧ (¬(¬q ∨ IMPL_FREE(¬p → (q ∧ r))))) 
 = ¬(r ∧ (¬(¬q ∨ (¬¬p ∨ (q ∧ r))))) 
```

##### 步骤 B: NNF
```text
NNF(¬(r ∧ (¬(¬q ∨ (¬¬p ∨ (q ∧ r)))))) 
 = NNF(¬r) ∨ NNF(¬¬(¬q ∨ (¬¬p ∨ (q ∧ r)))) 
 = ¬r ∨ NNF(¬q ∨ (¬¬p ∨ (q ∧ r))) 
 = ¬r ∨ ¬q ∨ NNF(¬¬p) ∨ (q ∧ r) 
 = ¬r ∨ ¬q ∨ p ∨ (q ∧ r) 
```

##### 步骤 C: CNF
```text
CNF(¬r ∨ ¬q ∨ p ∨ (q ∧ r)) 
 = DISTR(CNF(¬r ∨ ¬q ∨ p), CNF(q ∧ r)) 
 = DISTR(DISTR(CNF(¬r), CNF(¬q ∨ p)), CNF(q) ∧ CNF(r)) 
 = DISTR(¬r ∨ ¬q ∨ p, q ∧ r) 
 = DISTR(¬r ∨ ¬q ∨ p, q) ∧ DISTR(¬r ∨ ¬q ∨ p, r) 
 = (¬r ∨ ¬q ∨ p ∨ q) ∧ (¬r ∨ ¬q ∨ p ∨ r)
```

### 2. Tseitin 转换算法 (等可满足转换)

标准算法在使用分配律时，最坏情况下会导致公式大小呈指数级爆炸。为了解决这个问题，可以采用 **Tseitin 转换**。

该算法通过为内部子公式引入新变量，将原公式转换为**等可满足 (Equisatisfiable)** 的 CNF 公式，且转换后的公式大小呈线性增长。

详细算法流程与示例请见：[[Tseitin转换]]。