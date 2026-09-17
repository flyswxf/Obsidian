# ICU 项目面试技术亮点

> 基于 GraphCare (ICLR'24) 改进的 ICU 场景图神经网络医疗辅助决策系统。
> 代码仓库：`D:\code\GraphCare-improvement`

---

## 一、项目总览

- 我这个项目是在做一个 ICU 场景下的图神经网络医疗辅助决策系统。输入是患者的结构化电子病历数据，包括诊断、手术、药物、生命体征等信息；输出一方面是药物推荐，另一方面是心源性休克的高风险状态预测。
- 之所以用图建模，是因为医疗数据天然存在很强的关系结构，比如疾病和并发症之间、疾病和治疗之间、药物和禁忌之间都有明确的语义关联。其次，医学相关项目的一个目标是要有可解释性，图结构可以简洁清晰的可视化，所以选择图神经网络。
- 在实现上，我先根据患者相关的医学概念组织成个性化子图，再通过图神经网络进行消息传递。
- 训练阶段的难点主要在于医疗数据极度不平衡。很多常见药正样本量很大，但真正关键罕见药物正样本量非常小。如果直接用标准交叉熵，模型很容易被多数类主导，所以我引入了 Focal Loss，让模型在反向传播时更关注难分样本。同时在评估时，我重点看 Macro-F1，因为它能更真实反映模型对少数类的识别能力，而不是被多数类的高频正确率掩盖。
- 除此之外，我还做了一个反馈闭环。第一次推理完成后，医生可以输入自然语言反馈，比如补充风险因素、禁忌症或者强调某些临床信息。医生的自然语言反馈经过向量化，然后通过 HNSW 近似最近邻搜索，映射成对当前患者图中激活节点的动态增删，再触发二次推理。
- 我负责并实现了数据清洗、训练、优化，构造了反馈的 pipeline。

### 🇬🇧 English Version (for English Interviews)

> 以下为口语化版本，适合英文面试时直接说出来。

So this project is a GNN-based clinical decision support system for ICU patients. It's built on top of GraphCare from ICLR 2024.

The input is structured electronic health records — things like diagnoses, procedures, medications, vital signs. And the output is twofold: multi-label drug recommendation, and cardiogenic shock risk prediction.

**Why GNN?** Medical data has strong relational structure — diseases connect to complications, drugs connect to contraindications. Plus, interpretability matters a lot in healthcare, and graphs are easy to visualize. So GNN is a natural fit.

**How it works:** I build a personalized subgraph for each patient from their medical concepts, then run message passing through a custom GNN layer. Each message carries three things: neighbor features, a temporal-decay attention weight that emphasizes recent visits, and medical relation features on the edges.

**The main challenge** is extreme class imbalance — common drugs have tons of samples, but rare critical drugs have very few. Standard cross-entropy gets dominated by majority classes. So I use Focal Loss to down-weight easy samples and focus on hard ones. And for evaluation, I focus on Macro-F1 because it's more sensitive to minority classes.

**I also built a feedback loop.** After the first inference, a doctor can give natural language feedback — like "this patient has severe heart failure, emphasize related risk factors." An LLM breaks that into add/remove keywords, I embed them, and use HNSW to find the matching cluster nodes in the patient's graph. Then I add or remove those nodes and trigger a second inference — so the doctor gets an updated prediction without retraining.

I was responsible for the full pipeline — data preprocessing, training, optimization, and the feedback system.

---

## 二、图神经网络（Message Passing）

### 技术原理

图神经网络的核心机制是**消息传递（Message Passing）**。经过多层网络后，每个节点不仅包含自身特征，还融合了局部图的拓扑结构信息。消息传递被拆解为三个标准步骤：

1. **Message（消息计算）**：决定邻居节点 $j$ 要发给目标节点 $i$ 什么内容。通常是 $h_j$、$h_i$ 以及边特征 $e_{ij}$ 的函数。
2. **Aggregate（聚合）**：目标节点 $i$ 收到多个邻居消息，用对称函数（Sum/Mean/Max）聚合，保证**置换不变性**（Permutation Invariance，即不管邻居按什么顺序输入，结果都一样）。
3. **Update（状态更新）**：将聚合后的邻居消息与自身特征结合（通常是拼接或相加），然后通过 MLP 和激活函数（ReLU）得到新特征。

#### GNN 统一范式公式

$$h_i^{(l+1)} = \gamma \left( h_i^{(l)}, \square_{j \in \mathcal{N}(i)} \phi \left(h_i^{(l)}, h_j^{(l)}, e_{ij} \right) \right)$$

- $\phi$ (Phi) = **Message** 函数
- $\square$ = **Aggregate** 函数（如 Sum）
- $\gamma$ (Gamma) = **Update** 函数
- $h_i^{(l)}$ 表示节点 $i$ 在第 $l$ 层的特征，$\mathcal{N}(i)$ 表示节点 $i$ 的邻居集合

#### GraphCare 项目中的三维度消息

在 GraphCare 的 `BiAttentionGNNConv` 中，一条完整的 Message 包含 3 个维度：

1. **邻居特征** (`x_j`)：源节点经过线性变换后的特征。
2. **注意力权重** (`attn`)：基于时间衰减的注意力机制，作为权重乘在邻居特征上，表示不同邻居对当前患者的重要程度不同。
   - **Beta Attention**（就诊级）：`tanh(beta_attn(visit_node)) * lambda_j`，形状 `[B, V, 1]`，引入指数衰减 $\lambda_j = \exp(\lambda \cdot (V - j))$ 强化近期就诊
   - 在就诊维度求和，映射到边级别
3. **医学关系特征** (`edge_attr`)：节点间有具体的医学关系（如"导致"、"治疗"），经 `W_R` 线性映射后加到消息里 (`w_rel * edge_attr`)。

在 **Aggregate** 阶段使用加和池化（`aggr='add'`），**Update** 阶段将聚合消息与节点自身特征相加（`out + (1 + eps) * x_r`），再经 ReLU 激活。

### 代码实现

```python
# graphcare_/model.py - BiAttentionGNNConv
class BiAttentionGNNConv(MessagePassing):
    def __init__(self, nn, eps=0., edge_dim=None, edge_attn=True, **kwargs):
        kwargs.setdefault('aggr', 'add')          # Aggregate: 加和池化
        super().__init__(**kwargs)
        self.nn = nn
        self.edge_attn = edge_attn
        if edge_attn:
            self.W_R = torch.nn.Linear(edge_dim, 1)  # 关系特征 → 标量权重
        # ...

    def forward(self, x, edge_index, edge_attr=None, attn=None):
        # propagate 内部会调用 message() 和 aggregate()
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, attn=attn)
        # Update: 聚合消息 + (1 + eps) * 自身特征
        out = out + (1 + self.eps) * x[1]
        return self.nn(out), w_rel

    def message(self, x_j, edge_attr, attn):
        w_rel = self.W_R(edge_attr)                      # 关系特征 → 标量权重
        out = (x_j * attn + w_rel * edge_attr).relu()    # 三维度消息融合
        return out
```

```python
# graphcare_/model.py - GraphCare.forward() 中的注意力计算
# Beta Attention（就诊级）：引入指数衰减强化近期就诊
beta = torch.tanh(self.beta_attn[str(layer)](visit_node.float())) * self.lambda_j
# 在就诊维度求和，映射到边级别
attn = torch.sum(beta, dim=1)
attn = attn[xj_batch, xj_node_ids].reshape(-1, 1)
```

### 💡 答辩话术

"图神经网络的核心是消息传递机制，分为 Message、Aggregate、Update 三步。在我的 GraphCare 项目中，`BiAttentionGNNConv` 的消息包含三个维度：邻居特征、注意力权重和医学关系特征。

其中注意力权重是我改进的重点——我设计了 Beta Attention 机制，引入指数时间衰减来强化近期就诊的权重，映射到边级别后作为权重乘在邻居特征上。

医学关系特征方面，节点间的边不仅有连接关系，还有具体的医学语义（比如'导致'、'治疗'），我通过线性映射将其融入消息中。这样每条消息既包含了'谁传给谁'，也包含了'传什么'和'有多重要'。"

> **加分项**：如果在白板面试中，可以一边说一边写出 GNN 统一范式公式 $h_i^{(l+1)} = \gamma(h_i^{(l)}, \square_{j \in \mathcal{N}(i)} \phi(h_i^{(l)}, h_j^{(l)}, e_{ij}))$。

---

## 三、边稀疏化（Edge Sparsification）

### 技术原理

我的实现采用的是 **基于 Gumbel-Softmax 的可微边选择机制**。核心思路是：先用可学习的 `EdgeScorer` 给每条边打分，再把"保留这条边 / 删除这条边"建模成一个二分类采样问题。直接做 0/1 采样是不可微的，所以训练时使用 Gumbel-Softmax 对离散采样做连续松弛，让边选择模块也能参与端到端反向传播。

#### 1. 边选择建模

- **EdgeScorer**：一个多层感知机，接收源节点、目标节点和边的初始特征，输出每条边的保留倾向分数 `edge_logits`。
- **二分类门控**：对每条边构造两个 logit，分别对应 `drop` 和 `keep`。这样边稀疏化就被转成一个离散门控问题。

#### 2. Gumbel-Softmax 的作用

如果直接做 Bernoulli 采样或阈值化，梯度会在采样处中断。Gumbel-Softmax 的做法是：在 logit 上加入 Gumbel 噪声后，再经过 Softmax 得到一个**接近 one-hot、但仍然可微**的门控向量。对于每条边，可以写成：

$$
y_i = \frac{\exp((\log \pi_i + g_i)/\tau)}{\sum_j \exp((\log \pi_j + g_j)/\tau)}
$$

其中：

- $\pi_i$ 是保留/删除的原始打分
- $g_i$ 是 Gumbel 噪声
- $\tau$ 是温度参数

训练早期用较高温度，采样更平滑，便于探索；训练后期逐步降低温度，门控结果会越来越接近真正的 0/1 决策。这样既能模拟离散边选择，又能保留梯度。

#### 3. 稀疏性与连通性约束

为了避免模型把所有边都保留下来，训练时还加入两个辅助约束：

- **L1 稀疏性惩罚**：压低整体的保留概率，鼓励模型只保留关键边
- **连通性保持惩罚**：约束每个节点至少保留一定强度的连边，避免图被切得过碎

通过这种设计，每条边都不是被静态裁掉，而是在训练中动态竞争。重要边会在主任务损失驱动下保留下来，不重要的边会被稀疏性约束压低，从而实现结构自适应的边剪枝。

### 代码实现

```python
# SparseModel.py - EdgeScorer: 为每条边学习 keep / drop 的打分
class EdgeScorer(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim=64):
        super().__init__()
        self.node_transform = nn.Linear(node_dim, hidden_dim)
        self.edge_transform = nn.Linear(edge_dim, hidden_dim)
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),   # src + tgt + edge
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)            # 输出 keep logit
        )

    def forward(self, x, edge_index, edge_attr):
        src = self.node_transform(x[edge_index[0]])   # [E, H/2]
        tgt = self.node_transform(x[edge_index[1]])   # [E, H/2]
        edge = self.edge_transform(edge_attr)          # [E, H/2]
        combined = torch.cat([src, tgt, edge], dim=1)  # [E, 3H/2]
        keep_logits = self.scorer(combined)            # [E, 1]
        drop_logits = torch.zeros_like(keep_logits)    # [E, 1]
        return torch.cat([drop_logits, keep_logits], dim=1)  # [E, 2]
```

```python
# SparseModel.py - forward() 中的 Gumbel-Softmax 边选择
edge_logits = self.edge_scorer(x, edge_index, edge_attr)       # [E, 2]
edge_gates = F.gumbel_softmax(edge_logits, tau=self.tau, hard=False, dim=1)
edge_weights_input = edge_gates[:, 1].view(-1, 1)              # keep 概率

# 如果希望前向更接近离散采样，也可以用 straight-through
# edge_gates = F.gumbel_softmax(edge_logits, tau=self.tau, hard=True, dim=1)
# edge_weights_input = edge_gates[:, 1].view(-1, 1)

x, w_rel = self.conv[str(layer)](x, edge_index, edge_attr,
                                  attn=attn_edges, edge_weights=edge_weights_input)
```

```python
# SparseModel.py - 辅助稀疏化损失
def compute_sparsification_loss(self, edge_weights, edge_index):
    # L1 稀疏性惩罚：全局压低保留概率
    l1_loss = torch.mean(edge_weights)
    # 连通性保持惩罚：每个节点的边权重之和不应低于阈值
    num_nodes = torch.max(edge_index) + 1
    edge_counts = torch.zeros(num_nodes, device=edge_weights.device)
    edge_counts.scatter_add_(0, edge_index[0], edge_weights.squeeze())
    connectivity_loss = torch.mean(torch.relu(1.0 - edge_counts))
    return self.l1_lambda * l1_loss + self.connectivity_lambda * connectivity_loss
```

```python
# SparseModel.py - message() 中应用边权重
def message(self, x_j, edge_attr=None, attn=None, edge_weights=None):
    if self.edge_attn and edge_attr is not None and self.W_R is not None:
        w_rel = self.W_R(edge_attr)
        if attn is not None:
            msg = (x_j * attn + w_rel * edge_attr).relu()
        else:
            msg = (x_j + w_rel * edge_attr).relu()
    else:
        msg = (x_j * attn).relu() if attn is not None else x_j.relu()
    # 应用稀疏化边权重
    if edge_weights is not None:
        msg = msg * edge_weights.view(-1, 1)
    return msg
```

### 💡 答辩话术

"我的边稀疏化实现采用的是 Gumbel-Softmax。核心原因是边保留本质上是一个离散 0/1 决策，如果直接做阈值裁剪，梯度会在采样处中断，边选择模块就很难和主任务一起端到端训练。

具体做法是，我先用一个 EdgeScorer 网络读取源节点、目标节点和边特征，输出每条边的 keep / drop logits；然后对这个二分类分布做 Gumbel-Softmax 采样，得到一个接近 0-1、但仍然可微的边门控权重，再把这个权重乘到消息传递上。

训练时我还做了温度退火。前期温度高，采样更平滑，方便模型探索哪些边重要；后期温度逐步降低，边门控会越来越接近真正的 0/1 选择。同时我又加了 L1 稀疏约束和连通性约束，既鼓励模型剪掉冗余边，又避免把图结构破坏得太严重。

这样做的好处是，边稀疏化不再是一个训练外的启发式剪枝，而是可以直接纳入端到端优化过程里，既减少了消息传递的计算量，也提升了模型对关键医学关系的筛选能力。"

---

## 四、训练策略（Focal Loss）

### 技术原理

Focal Loss 是何恺明在 2017 年提出的专为解决**极端类别不平衡**问题的损失函数。

#### $p_t$ 定义

$p_t$ = 模型预测该样本为**真实类别**的概率：
- 正样本（$y=1$）：$p_t = p$
- 负样本（$y=0$）：$p_t = 1 - p$

**$p_t$ 越大 → 样本越容易分；$p_t$ 越小 → 样本越难分。**

#### 标准交叉熵的问题

$$CE(p_t) = -\log(p_t)$$

假设 100 个正样本（休克患者），100,000 个负样本（普通患者）：
- 易分负样本 $p_t=0.99$，单样本 Loss $\approx 0.004$，但 $100,000 \times 0.004 = 400$
- 难分正样本 $p_t=0.1$，Loss $\approx 2.3$，$100 \times 2.3 = 230$

**结论**：海量"易分负样本"的微小 Loss 累加，彻底淹没"难分正样本"的 Loss。

#### 引入 $\gamma$ 解决"难易"问题

$$FL(p_t) = -(1 - p_t)^\gamma \log(p_t)$$

$\gamma=2$ 时：
- 易分样本（$p_t=0.9$）：$(1-0.9)^2=0.01$，Loss **缩小 100 倍**
- 难分样本（$p_t=0.1$）：$(1-0.1)^2=0.81$，Loss **几乎不变**

$\gamma$ 的作用是**按难度动态分配权重**：模型预测得越准，惩罚力度衰减得越狠。当 $\gamma = 0$ 时，Focal Loss 退化为普通交叉熵。

#### 引入 $\alpha$ 解决"数量"问题

$$FL(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

- $\alpha$ 从**物理数量**上平衡正负类（如 $\alpha=0.75$ 给正样本更高权重）
- $\gamma$ 从**学习难度**上平衡易分/难分样本

### 代码实现

```python
# runSparseModel.py - FocalLoss
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        # logits: (N, C), targets: (N, C)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1 - probs) * (1 - targets)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        if self.reduction == 'mean':
            return focal.mean()
        elif self.reduction == 'sum':
            return focal.sum()
        return focal
```

### 💡 答辩话术

"在我的 GraphCare ICU 辅助决策项目中，多标签药物推荐和心源性休克风险预测都面临极其严重的数据长尾分布问题。如果直接使用 Cross Entropy，模型会被海量的、容易预测的常见病和常规用药主导，导致在罕见药和休克高危患者上的 Recall 非常低。

为了解决这个问题，我引入了 Focal Loss，核心公式 $-\alpha_t (1 - p_t)^\gamma \log(p_t)$。

这里面有两个精妙的参数设计：

第一个是 $\gamma$（聚焦参数），我在项目中设为 2。它的作用是对 Loss 施加动态缩放因子 $(1-p_t)^\gamma$。如果一个常规药模型已经预测得很准了（$p_t=0.9$），这个因子会把它的 Loss 缩小 100 倍；而对于预测不准的罕见特效药，Loss 几乎不衰减。这就强迫 GNN 在反向传播时，把梯度更新的重心放在攻克'难分样本'上。

第二个是 $\alpha$（平衡参数），我设为 0.25。虽然 $\gamma$ 解决了难易问题，但正负样本的绝对数量差异依然存在。所以我根据数据集中正负标签的先验分布，为少数类分配了更高的 $\alpha$ 权重，从宏观数量层面进一步平衡了 Loss。

通过 $\alpha$ 和 $\gamma$ 的双管齐下，模型不再做'和事佬'去迎合大众数据，从而大幅提升了少数类的识别率，这也直接反映在了最终 Macro-F1 指标的显著提升上。"

---

## 五、验证与评估（Micro-F1 vs Macro-F1）

### 技术原理

在多标签分类中（一个病人可能同时被推荐多种药物），需要将多个类别的 F1-Score 聚合成一个全局指标。

**基础概念**：F1-Score 是精确率（Precision）和召回率（Recall）的调和平均数。
- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)

| 指标           | 计算方式                                | 核心特点                           | 适用场景                     |
| ------------ | ----------------------------------- | ------------------------------ | ------------------------ |
| **Micro-F1** | 全局统算：打破类别界限，累加所有 TP/FP/FN 后算一个总的 F1 | **受多数类主导**，样本量越大的类别影响越大        | 类别分布均衡，或只关心总体正确数         |
| **Macro-F1** | 算术平均：先独立计算每个类别的 F1，再取平均             | **众生平等，对少数类高度敏感**，每个类别权重都是 1/N | 严重数据不平衡，且要求模型在少数类上也有良好表现 |

### 代码实现

```python
# runSparseModel.py - 验证阶段指标计算
# 搜索每类 F1 最优阈值
per_class_thr_opt = find_best_per_class_thresholds(calc_y_true, calc_y_prob)

# 使用搜索到的每类阈值进行决策
y_pred_val = multilabel_decision(
    calc_y_prob, strategy='threshold',
    threshold=args.threshold, topk=args.topk,
    per_class_thresholds=per_class_thr_opt
)

# 多标签指标
val_f1 = f1_score(calc_y_true, y_pred_val, average="samples", zero_division=1)
val_precision = precision_score(calc_y_true, y_pred_val, average="samples", zero_division=1)
val_recall = recall_score(calc_y_true, y_pred_val, average="samples", zero_division=1)
```

```python
# runSparseModel.py - 搜索每类最优阈值
def find_best_per_class_thresholds(y_true, y_prob, grid_size=200):
    """对每个类别，在概率分位网格上搜索使该类 F1 最大的阈值"""
    N, C = y_prob.shape
    thresholds = []
    for c in range(C):
        yt = y_true[:, c].astype(int)
        yp = y_prob[:, c].astype(float)
        if np.sum(yt) == 0:
            thresholds.append(0.5)
            continue
        cand = np.quantile(yp, np.linspace(0.01, 0.99, grid_size))
        best_t, best_f1 = 0.5, -1.0
        for t in cand:
            yhat = (yp >= float(t)).astype(int)
            f1c = f1_score(yt, yhat, average='binary', zero_division=0)
            if f1c > best_f1:
                best_f1, best_t = f1c, float(t)
        thresholds.append(best_t)
    return thresholds
```

```python
# runSparseModel.py - 多标签决策策略
def multilabel_decision(y_prob, strategy='threshold', threshold=0.5,
                        topk=10, per_class_thresholds=None):
    if strategy == 'threshold':
        # 全局或每类阈值
        if per_class_thresholds is not None:
            thr = np.array(per_class_thresholds)
            return (y_prob >= thr[None, :]).astype(int)
        return (y_prob >= float(threshold)).astype(int)
    elif strategy == 'topk':
        # Top-K 选择
        N, C = y_prob.shape
        y_bin = np.zeros_like(y_prob, dtype=int)
        k = max(1, min(C, int(topk)))
        top_idx = np.argpartition(-y_prob, kth=k-1, axis=1)[:, :k]
        y_bin[np.arange(N)[:, None], top_idx] = 1
        return y_bin
    elif strategy == 'hybrid':
        # 阈值优先，无选中时回退到 Top-K=1
        y_bin = multilabel_decision(y_prob, strategy='threshold',
                                    threshold=threshold,
                                    per_class_thresholds=per_class_thresholds)
        fallback_rows = np.where(y_bin.sum(axis=1) == 0)[0]
        if len(fallback_rows) > 0:
            k = max(1, min(y_prob.shape[1], int(topk)))
            top_idx = np.argpartition(-y_prob[fallback_rows], kth=k-1, axis=1)[:, :k]
            y_bin[fallback_rows[:, None], top_idx] = 1
        return y_bin
```

### 💡 答辩话术

"在多标签评估中，Micro-F1 和 Macro-F1 的核心区别在于它们对待数据不平衡的态度。Micro-F1 是全局样本统计，受多数类主导；而 Macro-F1 是各类平均，对少数类的表现非常敏感。

在我的 GraphCare ICU 医疗辅助决策项目中，我们面临着极度严重的长尾分布。比如，β-内酰胺抗菌药、青霉素类用得很多，样本量极大；而特效药样本量极小。

如果我们的评估指标只看 Micro-F1，模型只要无脑推荐通用药，分数就会看起来非常高，但这在临床决策中是毫无意义的。**因此，我们在该项目中重点关注 Macro-F1**，因为它能真实反映模型对罕见药物和高危休克风险的联合预测能力。"

---

## 六、反馈模块（Feedback Loop）

### 技术原理

第一次推理完成后，医生可以输入自然语言反馈（如"患者有严重心衰，需要强化相关风险因素"或"患者对某类药物存在禁忌，需要弱化对应推荐"）。系统不会直接把文本拼进 prompt 就结束，而是将其转换成图模型可执行的结构化操作。

#### 整体流程（6 步）

1. **LLM 拆解**：将自然语言反馈拆成 add/remove 两类关键词
2. **向量化**：将关键词编码成 embedding
3. **HNSW 搜索**：在预先构建的 cluster embedding 索引中做近似最近邻搜索，快速召回最相关的 cluster 候选
4. **重排**：如有多个关键词，将候选集合并，再做精确相似度重排（余弦相似度求和）
5. **图更新**：把最终得到的 cluster index 用于修改患者样本里的 `ehr_node_set`，增加或移除某些 cluster 节点
6. **二次推理**：触发二次图推理

#### HNSW 的作用

- **被搜索向量**：cluster embedding（聚类节点的向量表示）
- **查询向量**：关键词的 embedding
- 功能：给一个关键词 embedding，快速找到最相近的若干个 cluster embedding，反映聚类节点与关键词的相关性
- **重排的作用**：将每一个候选 cluster 的 embedding 与所有关键词 embedding 的余弦相似度求和，反映聚类节点与新增描述的相关性

#### HNSW 参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `max_elements` | 动态 | 根据 cluster 数量决定 |
| `M` | 32 | 每节点最多 32 条近邻边，保证连通性 |
| `ef_construction` | 200 | 建图时候选范围，换取高质量索引 |
| `ef_search` | 100 | 查询时候选范围，换取更好召回率 |

### 代码实现

```python
# graphcare.py - label_ehr_nodes() 支持反馈增删
def label_ehr_nodes(task, sample_dataset, max_nodes, ccscm_id2clus,
                    ccsproc_id2clus, atc3_id2clus,
                    feedback_add_clusters=None, feedback_remove_clusters=None):
    rem_set = set(feedback_remove_clusters or [])
    add_set = set(feedback_add_clusters or [])

    for patient in sample_dataset:
        nodes = []
        # 遍历诊断 → 映射到 cluster 节点，跳过 remove 列表
        for condition in flatten(patient['conditions']):
            ehr_node = ccscm_id2clus[condition]
            if int(ehr_node) not in rem_set:
                nodes.append(int(ehr_node))
        # 遍历手术和药物（同理）
        # ...
        # 添加 add 列表中的节点
        if add_set:
            for n in add_set:
                if 0 <= int(n) < int(max_nodes):
                    nodes.append(int(n))
        # 生成 one-hot 编码
        node_vec = np.zeros(max_nodes)
        node_vec[nodes] = 1
        patient['ehr_node_set'] = torch.tensor(node_vec)
    return sample_dataset
```

```python
# keyword_extractor.py - LLM 拆解反馈为 add/remove 关键词
def _llm_extract_keywords(feedback_text, task, num_add, num_remove):
    client = ChatECNU(model="ecnu-max")
    system_prompt = (
        "你是一名医疗知识图谱助手。根据用户反馈和任务类型，将文本拆分为两组关键词：\n"
        "- add：表示需要添加或强调的概念/药物/手术等；\n"
        "- remove：表示需要移除或弱化的概念/药物/手术等；\n"
        "请只输出JSON：{\"add\": [...], \"remove\": [...]}。"
    )
    client.set_system_message(system_prompt)
    user_prompt = f"任务类型：{task}\n用户反馈：{feedback_text}"
    msg = client.chat(user_prompt)
    parsed = json.loads(msg.content)
    return {"add": parsed.get("add", []), "remove": parsed.get("remove", [])}
```

```python
# cluster_mapper.py - 将关键词映射到 cluster 索引
def map_keywords_to_clusters(task=None, topk_add=10, topk_remove=10):
    clusters = _load_clusters(cluster_file)
    keywords = _load_keywords()  # {"add": [...], "remove": [...]}

    # 为关键词计算 embedding
    add_embs = [(_embed(k), k) for k in keywords.get("add", [])]
    rem_embs = [(_embed(k), k) for k in keywords.get("remove", [])]

    # 对每个 cluster，计算与所有关键词的余弦相似度之和
    add_scores, rem_scores = [], []
    for idx_str, info in clusters.items():
        emb = info.get("embedding")
        if not emb:
            continue
        s_add = sum(_cosine(emb, e) for e, _k in add_embs)
        s_rem = sum(_cosine(emb, e) for e, _k in rem_embs)
        add_scores.append((int(idx_str), float(s_add)))
        rem_scores.append((int(idx_str), float(s_rem)))

    # 取 Top-K
    add_scores.sort(key=lambda x: x[1], reverse=True)
    rem_scores.sort(key=lambda x: x[1], reverse=True)
    result = {"add": [i for i, _ in add_scores[:topk_add]],
              "remove": [i for i, _ in rem_scores[:topk_remove]]}

    with open(CLUSTER_INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result
```

```bash
# 反馈推理的启动命令
python runSparseModel.py --infer --feedback --patient_id 21 \
    --weights_path ./data/weights/saved_weights_mimic3_drugrec_sparse.pkl
```

### 💡 答辩话术

"我做的反馈闭环分四步：首先用大模型将医生的自然语言反馈拆解成 add/remove 两组关键词；然后把这些关键词编码成 embedding；接着用 HNSW 在预先构建好的 cluster embedding 索引中做近似最近邻搜索，快速召回最相关的 cluster 候选，如果有多个关键词，我会把候选集合并，再做一次精确相似度重排；最后把得到的 cluster index 用于修改患者样本里的 ehr_node_set，也就是增加或移除某些 cluster 节点，再触发二次图推理。

HNSW 的作用主要是把原本线性扫描所有 cluster 的高耗时过程，变成低延迟的近似最近邻检索，让这个反馈闭环可以做到交互式响应。我在推理时动态改变当前患者子图中被激活的医学概念节点，而不是重新训练模型，这样既灵活又高效。"

---

## 七、患者表征聚合

### 技术原理

`patient_mode="joint"` 模式下，拼接两种表征，兼顾图结构信息与直接临床信息：

- **Graph-level**：通过全局均值池化（`global_mean_pool`）获取患者子图的整体拓扑信息
- **Node-level**：通过 EHR 节点的加权求和获取患者直接的医疗事件特征
- 两者拼接后经 MLP 输出最终预测

### 代码实现

```python
# graphcare_/model.py - GraphCare.forward() 中的患者表征聚合
# Graph-level: 全局图池化
x_graph = global_mean_pool(x, batch)                    # [B, H]
x_graph = F.dropout(x_graph, p=self.dropout, training=self.training)

# Node-level: EHR 加权聚合
x_node = torch.stack([
    ehr_nodes[i].view(1, -1) @ self.node_emb.weight    # one-hot × embedding 矩阵
    / torch.clamp(torch.sum(ehr_nodes[i]), min=1e-6)   # 归一化
    for i in range(batch_size)
])
x_node = self.lin_node(x_node).squeeze(1)
x_node = F.dropout(x_node, p=self.dropout, training=self.training)

# 拼接 → MLP 输出
x_concat = torch.cat((x_graph, x_node), dim=1)          # [B, 2H]
logits = self.MLP(x_concat)                             # [B, C]
```

### 💡 答辩话术

"在患者表征聚合上，我采用了 joint 模式，同时利用图级和节点级两种表征。图级表征通过全局均值池化获取患者子图的整体拓扑信息，能捕捉疾病之间的关联结构；节点级表征通过 EHR 节点的加权求和获取患者直接的医疗事件特征，保留了临床信息的细节。两者拼接后送入 MLP 输出预测，这样既不丢失图结构信息，也不丢失直接的临床信息。"

---

## 八、心源性休克风险预测

### 技术原理

在 `drugrec` 任务中，当启用 `Heart` 参数时，输出维度为 $D_m + 1$：

- 前 $D_m$ 维：药物推荐的多标签概率
- 最后 1 维：心源性休克风险概率

**标签泄露防护**：输入侧剔除心源性休克诊断 $d_{\mathrm{CS}}$，仅在标签端作为独立风险维度。数据层使用 $\bar{S}_i^{\text{diag}}(t) = S_i^{\text{diag}}(t) \setminus \{d_{\mathrm{CS}}\}$，防止模型直接从输入中"抄答案"。

### 代码实现

```python
# graphcare.py - 输出维度配置
def get_mode_and_out_channels_and_loss_func(task, sample_dataset, Heart=False):
    if task == "drugrec":
        mode = "multilabel"
        out_channels = len(sample_dataset[0]["drugs_ind"])
        # Heart 模式下 out_channels = D_m + 1，最后一位是心源性休克
        loss_function = F.binary_cross_entropy_with_logits
    return mode, out_channels, loss_function
```

```python
# runSparseModel.py - 推理输出分离
if Heart and task == 'drugrec' and mode == "multilabel":
    C_full = y_prob_all.shape[1]
    c_idx = C_full - 1
    # 提取心源性休克概率
    cardiogenic_prob = float(y_prob_all[0][c_idx])
    # 截断用于药物推荐评估
    y_prob_all = y_prob_all[:, :c_idx]
    y_true_all = y_true_all[:, :c_idx]
    result["cardiogenic_shock"] = float(cardiogenic_prob)
```

### 💡 答辩话术

"在心源性休克风险预测上，我将输出维度从 $D_m$ 扩展到 $D_m + 1$，最后一位专门用于预测心源性休克风险概率。推理时将这一位分离出来单独输出，不参与药物推荐的评估。

一个重要的设计是标签泄露防护——我在输入侧剔除了心源性休克的诊断信息，只在标签端作为独立的风险维度。这样防止模型直接从输入中'看到答案'，迫使它通过其他诊断、手术和药物信息来间接推理休克风险。"

---

## 九、关键性下沉分配机制

### 技术原理

为增强可解释性，将超节点级关键性沿聚类成员关系与患者时序激活轨迹，分配至原始诊断/手术节点：

- **时序存在性权重** $p_i(u)$：近期出现的节点获得更高权重
- **结构桥接权重** $q_i(u)$：跨簇连接强的节点获得更高权重
- **语义贴合权重** $s(u)$：与簇中心语义接近的节点获得更高权重

$$\gamma_i(u)=\gamma(C_a)\cdot \rho(u\mid C_a),\qquad \sum_{u\in U(C_a)}\gamma_i(u)=\gamma(C_a)$$

保证关键性总量守恒，支持原始节点级的可视化与路径溯源。

### 💡 答辩话术

"为了增强模型的可解释性，我设计了关键性下沉分配机制。GNN 输出的关键性是在超节点（即 cluster 节点）级别的，但医生需要知道具体是哪个诊断或手术起了关键作用。所以我把超节点级的关键性沿着三个维度分配到原始节点：时序存在性（近期出现的节点权重更高）、结构桥接（跨簇连接强的节点权重更高）、语义贴合（与簇中心语义接近的节点权重更高）。分配过程保证关键性总量守恒，这样医生可以在原始节点级别看到可视化的关键路径溯源。"
