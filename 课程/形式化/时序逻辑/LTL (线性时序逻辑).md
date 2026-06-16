## 1. LTL 线性时序逻辑 (Syntax & Semantics)

LTL 用于描述沿着时间轴（无限执行路径）的属性。

### 1.1 核心与扩展时序算子 (Temporal Operators)

| 符号 (Math) | 名称                    | 含义 (Intuition)                                                                                              |
| :---------- | :---------------------- | :------------------------------------------------------------------------------------------------------------ |
| $X P$       | **Next** (下一步)       | 在**紧接的下一个**状态，P 必须成立。                                                                          |
| $P_1 U P_2$ | **Until** (直到)        | $P_1$ 必须一直保持成立，**直到** $P_2$ 成立的那一刻。注意：**$P_2$ 必须在未来某一刻真的发生**，不能无限推迟。 |
| $G P$       | **Always** (总是)       | 在路径的**每一个**状态（从现在到无穷远），P 都必须成立。                                                      |
| $F P$       | **Eventually** (最终)   | 在未来的**某个**时刻，P 终将成立。                                                                            |
| $P_1 R P_2$ | **Release** (释放)      | $P_2$ 必须一直成立，直到 $P_1$ 发生并释放它。注意：**$P_1$ 可以永远不发生**，此时 $P_2$ 永远成立。            |
| $P_1 W P_2$ | **Weak Until** (弱直到) | $P_1$ 一直保持直到 $P_2$ 发生。注意：**不强制要求 $P_2$ 在未来发生**。                                        |

### 1.2 逻辑连接词 (Boolean Connectives)
*   $\neg P$ : 非
*   $P_1 \land P_2$ : 与
*   $P_1 \lor P_2$ : 或
*   $P_1 \rightarrow P_2$ : 蕴含

### 1.3 等价变换

#### 取反规则 (Negation / Duality)
**口诀**：当 $\neg$ 穿过时序算子时，算子变为其对偶形式。
- $\neg G P \equiv F \neg P$
- $\neg F P \equiv G \neg P$
- $\neg X P \equiv X \neg P$
- $\neg (P_1 U P_2) \equiv \neg P_1 R \neg P_2$

#### 用 Until 表达 Future 与 Always
**口诀**：$F$ 和 $G$ 都可以通过 $U$ 来表达。
- $F P \equiv \text{True} U P$ (一直 True，直到 $P$ 发生)
- $G P \equiv \neg F \neg P \equiv \neg (\text{True} U \neg P)$

#### 扩展算子的等价表达
- $P_1 R P_2 \equiv \neg (\neg P_1 U \neg P_2)$
- $P_1 W P_2 \equiv (P_1 U P_2) \lor G P_1$

---

## 2. 属性分类: Safety vs. Liveness

| 类型                | 描述                                                      | 典型公式模式        | 例子                                                                         |
| :------------------ | :-------------------------------------------------------- | :------------------ | :--------------------------------------------------------------------------- |
| **Safety** (安全性) | **"Nothing bad ever happens"**<br>(坏事永不发生)          | $G \neg \text{bad}$ | **互斥**: 两个进程永远不能同时在临界区。<br>$G \neg (crit(A) \land crit(B))$ |
| **Liveness** (活性) | **"Something good eventually happens"**<br>(好事终将发生) | $F \text{good}$     | **响应**: 请求最终会被处理。<br>$G (req \rightarrow F ack)$                  |

---

## 3. 实战演练
将自然语言需求转化为严格的 LTL 公式。

### Exercise 1: Safe temperature (Safety)
> **Requirement**: From now on, the temperature is always within a safe range.
> (从现在起，温度永远保持在安全范围内。)

*   **Atomic Proposition**: `tempOK`
*   **LTL Formula**:
    $$ G \text{tempOK} $$

### Exercise 2: Alarm must be reset (Response)
> **Requirement**: Whenever the alarm is active, it must eventually be reset.
> (每当报警器激活，它必须最终被重置。)

*   **Atomic Propositions**: `alarm`, `reset`
*   **LTL Formula**:
    $$ G (\text{alarm} \rightarrow F \text{reset}) $$

### Exercise 3: At least one reboot (Liveness)
> **Requirement**: The controller will reboot at least once in the future.
> (控制器在未来至少会重启一次。)

*   **Atomic Proposition**: `reboot`
*   **LTL Formula**:
    $$ F \text{reboot} $$

### Exercise 4: Initialization (Complex Until)
> **Requirement**: The system stays in initialization mode until it becomes configured, and while it is in initialization mode it must not be running.
> (系统保持初始化模式直到配置完成；且在初始化期间，它绝不能处于运行状态。)

*   **Atomic Propositions**: `init`, `configured`, `running`
*   **LTL Formula**:
    $$ (\text{init } U \text{ configured}) \land G (\text{init} \rightarrow \neg \text{running}) $$
*   **解析**:
    1.  前半部分 `init U configured` 描述了状态流转：一直是 `init` 直到 `configured`。
    2.  后半部分 `G (init -> ~ running)` 是一个附加的不变式约束：只要是在 `init` 状态，`running` 就必须是假。

### Exercise 5: Maintenance (Next & Until)
> **Requirement**: Whenever a maintenance request occurs, then at the **next step** the system must enter maintenance mode and remain in maintenance mode **until** maintenance is done.
> (每当收到维护请求，下一步系统必须进入维护模式，并保持在该模式直到维护完成。)

*   **Atomic Propositions**: `maintenanceRequest`, `maintenance`, `done`
*   **LTL Formula**:
    $$ G (\text{maintenanceRequest} \rightarrow X (\text{maintenance} \land (\text{maintenance } U \text{ done}))) $$
*   **解析**:
    *   $G (\text{req} \rightarrow ...)$: 全局监视请求。
    *   $X (...)$: 约束**下一时刻**。
    *   $\text{maint} \land (\text{maint } U \text{ done})$: 
        *   进入维护模式 (`maintenance` is true)。
        *   并且保持维护模式直到完成 (`maintenance U done`)。
