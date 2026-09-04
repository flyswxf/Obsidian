<<<<<<< HEAD
=======
<<<<<<< HEAD
# ICU 项目面试技术亮点

> 基于 GraphCare (ICLR'24) 改进的 ICU 场景图神经网络医疗辅助决策系统。
> 代码仓库：`D:\code\GraphCare-improvement`

## 总的介绍

- 我这个项目是在做一个 ICU 场景下的图神经网络医疗辅助决策系统。输入是患者的结构化电子病历数据，包括诊断、手术、药物、生命体征等信息；输出一方面是药物推荐，另一方面是心源性休克的高风险状态预测。
- 之所以用图建模，是因为医疗数据天然存在很强的关系结构，比如疾病和并发症之间、疾病和治疗之间、药物和禁忌之间都有明确的语义关联。其次，医学相关项目的一个目标是要有可解释性，图结构可以简洁清晰的可视化，所以选择图神经网络。
- 在实现上，我先根据患者相关的医学概念组织成个性化子图，再通过图神经网络进行消息传递。
- 训练阶段的难点主要在于医疗数据极度不平衡。很多常见药正样本量很大，但真正关键罕见药物正样本量非常小。如果直接用标准交叉熵，模型很容易被多数类主导，所以我引入了 Focal Loss，让模型在反向传播时更关注难分样本。同时在评估时，我重点看 Macro-F1，因为它能更真实反映模型对少数类的识别能力，而不是被多数类的高频正确率掩盖。
- 除此之外，我还做了一个反馈闭环。第一次推理完成后，医生可以输入自然语言反馈，比如补充风险因素、禁忌症或者强调某些临床信息。医生的自然语言反馈经过向量化，然后通过 HNSW 近似最近邻搜索，映射成对当前患者图中激活节点的动态增删，再触发二次推理。
- 我负责并实现了数据清洗、训练、优化，构造了反馈的 pipeline。

---
=======
>>>>>>> 3c428a2 (chore: 初始化仓库，清理目录结构，合并编译原理作业)
# 总的介绍
- 我这个项目是在做一个 ICU 场景下的图神经网络医疗辅助决策系统。输入是患者的结构化电子病历数据，包括诊断、手术、药物、生命体征等信息；输出一方面是药物推荐，另一方面是心源性休克的高风险状态预测。
- 之所以用图建模，是因为医疗数据天然存在很强的关系结构，比如疾病和并发症之间、疾病和治疗之间、药物和禁忌之间都有明确的语义关联。其次，医学相关项目的一个目标是要有可解释性, 图结构可以简洁清晰的可视化, 所以选择图神经网络”
- 在实现上，我先根据患者相关的医学概念组织成个性化子图，再通过图神经网络进行消息传递。
- 训练阶段的难点主要在于医疗数据极度不平衡。很多常见药正样本量很大，但真正关键罕见药物正样本量非常小。如果直接用标准交叉熵，模型很容易被多数类主导，所以我引入了 Focal Loss，让模型在反向传播时更关注难分样本。同时在评估时，我重点看 Macro-F1，因为它能更真实反映模型对少数类的识别能力，而不是被多数类的高频正确率掩盖。
- 除此之外，我还做了一个反馈闭环。第一次推理完成后，医生可以输入自然语言反馈，比如补充风险因素、禁忌症或者强调某些临床信息。医生的自然语言反馈经过向量化, 然后再HNSW近似最近邻搜索，映射成对当前患者图中激活节点的动态增删，再触发二次推理
- 我负责并实现了数据清洗, 训练, 优化, 构造了反馈的pipline. 
<<<<<<< HEAD
=======
>>>>>>> f81516b86c4e8635008168039021ce6d584b6c55
>>>>>>> 3c428a2 (chore: 初始化仓库，清理目录结构，合并编译原理作业)

## 图神经网络部分

### Message Passing
<<<<<<< HEAD
=======
<<<<<<< HEAD

图神经网络的核心机制是**消息传递（Message Passing）**。经过多层网络后，每个节点不仅包含自身特征，还融合了局部图的拓扑结构信息。消息传递被拆解为三个标准步骤：

1. **Message（消息计算）**：决定邻居节点 $j$ 要发给目标节点 $i$ 什么内容。通常是 $h_j$、$h_i$ 以及边特征 $e_{ij}$ 的函数。
2. **Aggregate（聚合）**：目标节点 $i$ 收到多个邻居消息，用对称函数（Sum/Mean/Max）聚合，保证置换不变性。
3. **Update（状态更新）**：将聚合后的邻居消息与自身特征结合，通过 MLP 和激活函数得到新特征。

### GraphCare 项目中的三维度消息

在 GraphCare 的 `BiAttentionGNNConv` 中，一条完整的 Message 包含 3 个维度：

1. **邻居特征** (`x_j`)：源节点经过线性变换后的特征。
2. **注意力权重** (`attn`)：基于时间衰减的双注意力（Alpha + Beta），作为权重乘在邻居特征上，表示不同邻居对当前患者的重要程度不同。
   - **Alpha Attention**（节点级）：`softmax(alpha_attn(visit_node))`，形状 `[B, V, N]`，学习每个就诊中各节点的重要性
   - **Beta Attention**（就诊级）：`tanh(beta_attn(visit_node)) * lambda_j`，形状 `[B, V, 1]`，引入指数衰减 $\lambda_j = \exp(\lambda \cdot (V - j))$ 强化近期就诊
   - 两者相乘后在就诊维度求和，映射到边级别
3. **医学关系特征** (`edge_attr`)：节点间有具体的医学关系（如"导致"、"治疗"），经 `W_R` 线性映射后加到消息里 (`w_rel * edge_attr`)。

在 **Aggregate** 阶段使用加和池化（`aggr='add'`），**Update** 阶段将聚合消息与节点自身特征相加（`out + (1 + eps) * x_r`），再经 ReLU 激活。

### 代码实现要点

```python
# graphcare_/model.py - BiAttentionGNNConv.message()
def message(self, x_j, edge_attr, attn):
    w_rel = self.W_R(edge_attr)  # 关系特征 → 标量权重
    out = (x_j * attn + w_rel * edge_attr).relu()  # 三维度消息融合
    return out
```

### 💡 GNN 统一范式公式

$$h_i^{(l+1)} = \gamma \left( h_i^{(l)}, \square_{j \in \mathcal{N}(i)} \phi \left(h_i^{(l)}, h_j^{(l)}, e_{ij} \right) \right)$$

- $\phi$ = Message 函数
- $\square$ = Aggregate 函数（Sum）
- $\gamma$ = Update 函数

---

## 边稀疏化部分

我的实现采用了一种工程化、稳定的 **"软权重 + 硬掩码 + 辅助正则化 Loss"** 策略，而非 Gumbel-Softmax。

### 1. Top-K 掩码是怎么实现的？

**回答话术：**

"在我的代码中，Top-K 掩码通过两步走实现：

首先，我设计了一个多层感知机 `EdgeScorer`。它接收源节点、目标节点以及边的初始特征，拼接后经过映射和 Sigmoid 激活，输出一个 0 到 1 之间的 `edge_scores`，表示这条边的重要性。

```python
# SparseModel.py - EdgeScorer
class EdgeScorer(nn.Module):
    def forward(self, x, edge_index, edge_attr):
        src = self.node_transform(x[edge_index[0]])  # [E, H/2]
        tgt = self.node_transform(x[edge_index[1]])  # [E, H/2]
        edge = self.edge_transform(edge_attr)         # [E, H/2]
        combined = torch.cat([src, tgt, edge], dim=1)  # [E, 3H/2]
        return self.scorer(combined)  # Sigmoid → [E, 1] ∈ [0,1]
```

然后，根据预设的稀疏化比例（如保留 Top 10%），利用 `torch.topk` 找到当前 Batch 中重要性得分的阈值。将所有得分大于等于该阈值的边标记为 1，其余为 0，生成非 0 即 1 的 `sparsification_mask`。

```python
# SparseModel.py - forward() 中的 Top-K
k_keep = max(1, int(num_edges * self.sparsification_ratio))
topk_vals, _ = torch.topk(edge_scores.squeeze(), k_keep)
thresh = topk_vals.min()
sparsification_mask = (edge_scores.squeeze() >= thresh).float().view(-1, 1)
```

### 2. Hard Mask 的梯度截断如何处理？

"为解决 Hard Mask 带来的梯度截断问题，我采用了**掩码乘法结合辅助 Loss** 的策略：

**对于前向传播的主任务（保留的边）**：
将可微的软得分与硬掩码相乘：`edge_weights_input = edge_scores * sparsification_mask`。因为 mask 只是常数（1 或 0），所以对于被保留的边（Mask=1），主任务的梯度可以顺畅地通过 `edge_scores` 回传给 EdgeScorer。

**对于被丢弃的边（Mask=0）**：
它们无法从最终的分类 Loss 中获得梯度。为防止'一死百了'，我在应用 Mask **之前**，直接基于原始的 `edge_scores` 引入了**辅助稀疏化损失**，包含两部分：

- **L1 稀疏性惩罚**：`l1_loss = mean(edge_scores)`，全局压低所有边的得分
- **连通性保持惩罚**：计算每个节点的边权重之和，如果低于最低阈值则给予惩罚

```python
# SparseModel.py - compute_sparsification_loss()
def compute_sparsification_loss(self, edge_scores, edge_index):
    l1_loss = torch.mean(edge_scores)
    edge_counts = torch.zeros(num_nodes).scatter_add_(0, edge_index[0], edge_scores.squeeze())
    connectivity_loss = torch.mean(torch.relu(1.0 - edge_counts))
    return self.l1_lambda * l1_loss + self.connectivity_lambda * connectivity_loss
```

通过这种设计，即使一条边当前被 Mask 掉了，它依然能收到来自连通性辅助 Loss 的梯度。如果在后续的 Epoch 中，主任务需要用到这个节点，辅助 Loss 会推高它的 `edge_scores`，一旦超过阈值，它就会在下一次前向传播中被 Top-K 重新'复活'。这样既实现了物理上的稀疏计算，又保证了全局梯度的流动与动态探索。"

---

## 训练部分

### Focal Loss

Focal Loss 是何恺明在 2017 年提出的专为解决**极端类别不平衡**问题的损失函数。

#### $p_t$ 定义

$p_t$ = 模型预测该样本为**真实类别**的概率：
- 正样本（$y=1$）：$p_t = p$
- 负样本（$y=0$）：$p_t = 1 - p$

**$p_t$ 越大 → 样本越容易分；$p_t$ 越小 → 样本越难分。**

#### 标准交叉熵的问题

$$CE(p_t) = -\log(p_t)$$

假设 100 个正样本（休克患者），100,000 个负样本（普通患者）。易分负样本 $p_t=0.99$，单样本 Loss $\approx 0.004$，但 $100,000 \times 0.004 = 400$。难分正样本 $p_t=0.1$，Loss $\approx 2.3$，$100 \times 2.3 = 230$。

**结论**：海量"易分负样本"的微小 Loss 累加，彻底淹没"难分正样本"的 Loss。

#### 引入 $\gamma$ 解决"难易"问题

$$FL(p_t) = -(1 - p_t)^\gamma \log(p_t)$$

$\gamma=2$ 时：
- 易分样本（$p_t=0.9$）：$(1-0.9)^2=0.01$，Loss 缩小 100 倍
- 难分样本（$p_t=0.1$）：$(1-0.1)^2=0.81$，Loss 几乎不变

#### 引入 $\alpha$ 解决"数量"问题

$$FL(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

$\alpha$ 从物理数量上平衡正负类，$\gamma$ 从学习难度上平衡易分/难分样本。

### 代码实现

```python
# runSparseModel.py
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1 - probs) * (1 - targets)
        return (self.alpha * (1 - pt) ** self.gamma * bce).mean()
```

### 💡 面试高分话术

"在我的 GraphCare ICU 辅助决策项目中，多标签药物推荐和心源性休克风险预测都面临极其严重的长尾分布问题。如果直接使用 Cross Entropy，模型会被海量的、容易预测的常见病和常规用药主导，导致在罕见药和休克高危患者上的 Recall 非常低。

我引入了 Focal Loss，核心公式 $-\alpha_t (1 - p_t)^\gamma \log(p_t)$。$\gamma$ 通常设为 2，对 Loss 施加动态缩放因子 $(1-p_t)^\gamma$：常规药已预测很准（$p_t=0.9$）时 Loss 缩小 100 倍，而罕见特效药 Loss 几乎不衰减。$\alpha$ 设为 0.25，为少数类分配更高权重。

通过 $\alpha$ 和 $\gamma$ 双管齐下，模型不再迎合大众数据，大幅提升了少数类的识别率，直接反映在 Macro-F1 指标的显著提升上。"

---

## 验证部分

### Micro-F1 vs Macro-F1

- **Micro-F1**：全局统算，受多数类主导。适用于整体类别分布均衡的场景。
- **Macro-F1**：各类独立计算 F1 后取算术平均，对少数类高度敏感。适用于严重数据不平衡且要求模型在少数类上也有良好表现。

### 代码实现

```python
# runSparseModel.py - 验证阶段
val_f1 = f1_score(calc_y_true, y_pred_val, average="samples", zero_division=1)
# 同时计算 per-class 最优阈值
per_class_thr_opt = find_best_per_class_thresholds(calc_y_true, calc_y_prob)
```

### 多标签决策策略

支持三种策略：`threshold`（全局阈值）、`topk`（Top-K 选择）、`hybrid`（阈值优先，无选中时回退到 Top-K=1）。

### 💡 面试高分话术

"在多标签评估中，Micro-F1 和 Macro-F1 的核心区别在于对待数据不平衡的态度。Micro-F1 受多数类主导；Macro-F1 对少数类高度敏感。

在我的 GraphCare ICU 项目中，面临极度严重的长尾分布。β-内酰胺抗菌药、青霉素类样本量极大，而特效药样本量极小。如果只看 Micro-F1，模型无脑推荐通用药分数就会很高，但这在临床决策中毫无意义。因此我们重点关注 Macro-F1，它能真实反映模型对罕见药物和高危休克风险的联合预测能力。"

---

## 反馈模块

### 整体流程

第一次推理完成后，医生输入自然语言反馈（如"患者有严重心衰，需要强化相关风险因素"），系统将其转换成图模型可执行的结构化操作：

1. **LLM 拆解**：将反馈拆成 add/remove 两类关键词
2. **向量化**：将关键词编码成 embedding
3. **HNSW 搜索**：在预先构建的 cluster embedding 索引中做近似最近邻搜索，快速召回最相关的 cluster 候选
4. **重排**：如有多个关键词，将候选集合并，再做精确相似度重排
5. **图更新**：把最终得到的 cluster index 用于修改患者样本里的 `ehr_node_set`，增加或移除某些 cluster 节点
6. **二次推理**：触发二次图推理

### HNSW 的作用

- **被搜索向量**：cluster embedding
- **查询向量**：关键词 embedding
- 功能：给一个关键词 embedding，快速找到最相近的若干个 cluster embedding，反映聚类节点与关键词的相关性

### 参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `max_elements` | 动态 | 根据 cluster 数量决定 |
| `M` | 32 | 每节点最多 32 条近邻边，保证连通性 |
| `ef_construction` | 200 | 建图时候选范围，换取高质量索引 |
| `ef_search` | 100 | 查询时候选范围，换取更好召回率 |

### 代码实现

```python
# graphcare.py - label_ehr_nodes() 支持反馈
def label_ehr_nodes(task, sample_dataset, max_nodes, ...,
                    feedback_add_clusters=None, feedback_remove_clusters=None):
    rem_set = set(feedback_remove_clusters or [])
    add_set = set(feedback_add_clusters or [])
    # 移除指定节点 + 添加指定节点 → 更新 ehr_node_set
```

反馈系统通过 `--feedback` 参数启用，从 `clusterIndex.txt` 读取 add/remove 列表：

```bash
python runSparseModel.py --infer --feedback --patient_id 21 \
    --weights_path ./data/weights/saved_weights.pkl
```

### 💡 面试话术

"我在推理时动态改变当前患者子图中被激活的医学概念节点。HNSW 的作用主要是把原本线性扫描所有 cluster 的高耗时过程，变成低延迟的近似最近邻检索，让这个反馈闭环可以做到交互式响应。"

---

## 患者表征聚合

`patient_mode="joint"` 模式下，拼接两种表征：

```python
# graphcare_/model.py
x_graph = global_mean_pool(x, batch)                    # 全局图池化 [B, H]
x_node = ehr_nodes @ node_emb.weight / sum(ehr_nodes)  # EHR 加权聚合 [B, H]
logits = MLP(torch.cat((x_graph, x_node), dim=1))       # 拼接 → MLP [B, C]
```

- **Graph-level**：通过全局均值池化获取患者子图的整体拓扑信息
- **Node-level**：通过 EHR 节点的加权求和获取患者直接的医疗事件特征
- 两者拼接后经 MLP 输出，兼顾图结构信息与直接临床信息

---

## 心源性休克风险预测

在 `drugrec` 任务中，当启用 `Heart` 参数时，输出维度为 $D_m + 1$：

- 前 $D_m$ 维：药物推荐的多标签概率
- 最后 1 维：心源性休克风险概率

```python
# runSparseModel.py - 推理输出
if Heart and task == 'drugrec' and mode == "multilabel":
    cardiogenic_prob = float(y_prob_all[0][c_idx])  # 最后一位
    y_prob_all = y_prob_all[:, :c_idx]  # 截断用于药物推荐评估
    result["cardiogenic_shock"] = float(cardiogenic_prob)
```

> **标签泄露防护**：输入侧剔除心源性休克诊断 $d_{\mathrm{CS}}$，仅在标签端作为独立风险维度。数据层使用 $\bar{S}_i^{\text{diag}}(t) = S_i^{\text{diag}}(t) \setminus \{d_{\mathrm{CS}}\}$。

---

## 关键性下沉分配机制

为增强可解释性，将超节点级关键性沿聚类成员关系与患者时序激活轨迹，分配至原始诊断/手术节点：

- **时序存在性权重** $p_i(u)$：近期出现的节点获得更高权重
- **结构桥接权重** $q_i(u)$：跨簇连接强的节点获得更高权重
- **语义贴合权重** $s(u)$：与簇中心语义接近的节点获得更高权重

$$\gamma_i(u)=\gamma(C_a)\cdot \rho(u\mid C_a),\qquad \sum_{u\in U(C_a)}\gamma_i(u)=\gamma(C_a)$$

保证关键性总量守恒，支持原始节点级的可视化与路径溯源。
=======
>>>>>>> 3c428a2 (chore: 初始化仓库，清理目录结构，合并编译原理作业)
图神经网络的核心机制是**消息传递（Message Passing）**
经过多层网络（多轮交流）后，每个节点不仅包含了自身的特征，还融合了整个局部图的拓扑结构信息。
消息传递被拆解为三个标准步骤：**Message（计算消息）、Aggregate（聚合消息）和 Update（更新状态）**。

1. **Message (消息计算阶段)**：
   决定了一个邻居节点 $j$ 要发给目标节点 $i$ 什么内容。通常是邻居节点特征 $h_j$、目标节点特征 $h_i$ 以及它们之间边特征 $e_{ij}$ 的函数。
2. **Aggregate (聚合阶段)**：
   目标节点 $i$ 会收到多个邻居发来的消息。为了保证图模型的**置换不变性**（Permutation Invariance，即不管邻居按什么顺序输入，结果都一样），我们需要用一个对称函数来聚合这些消息。最常用的是 `Sum`（求和）、`Mean`（求平均）或 `Max`（最大值池化）。
3. **Update (状态更新阶段)**：
   目标节点 $i$ 将聚合后的邻居消息，与它自己上一层的特征 $h_i^{(l)}$ 进行结合（通常是拼接或相加），然后通过一个神经网络（如 MLP）和激活函数（ReLU），得到当前层的新特征 $h_i^{(l+1)}$。”

### GraphCare 项目

一条完整的 Message 包含了3个维度：
1. **邻居特征** (`x_j`)。
2. **注意力权重** (`attn`)：基于时间衰减的注意力，作为权重乘在邻居特征上，表示不同邻居对当前患者的重要程度不同。
3. **医学关系特征** (`edge_attr`)：节点间不仅仅有连接，还有具体的医学关系（比如‘导致’、‘治疗’）代表这条边（比如某种并发症关系）本身的语义信息(设置为一种固定的权重)。我将关系特征`w_rel`经过线性映射后，加到了消息里 (`w_rel * edge_attr`)。

在 **Aggregate 阶段**，我使用了加和池化（`Add` Aggregation）。
在 **Update 阶段**，将聚合来的邻居特征与节点自身特征相加，再通过 ReLU 激活

---

### 💡 附加加分项：如果在白板上让你写公式
如果你在现场面试或视频面试有白板，可以一边说一边写下这个极其经典的 GNN 统一范式公式：

$$h_i^{(l+1)} = \gamma \left( h_i^{(l)}, \square_{j \in \mathcal{N}(i)} \phi \left(h_i^{(l)}, h_j^{(l)}, e_{ij} \right) \right)$$

*   $\phi$ (Phi) 就是 **Message** 函数。
*   $\square$ (方框) 就是 **Aggregate** 函数（比如 Sum）。
*   $\gamma$ (Gamma) 就是 **Update** 函数。
*   $h_i^{(l)}$ 表示节点 $i$ 在第 $l$ 层的特征，$\mathcal{N}(i)$ 表示节点 $i$ 的邻居集合。

## 稀疏化部分
你的实现并没有使用复杂的 Gumbel-Softmax，而是采用了一种更工程化、更稳定的 **“软权重 + 硬掩码 + 辅助正则化 Loss” (Soft-Weighting + Hard Mask + Auxiliary Loss)** 的策略。

以下是针对面试官问题的标准回答思路，你可以直接借鉴：

### 1. 你的 Top-K 掩码是怎么实现的？

**回答话术：**
“在我的代码中，Top-K 掩码是通过一个两步走的机制实现的：
首先，我设计了一个多层感知机 `EdgeScorer`。它接收源节点、目标节点以及边的初始特征，经过映射和 Sigmoid 激活后，输出一个 0 到 1 之间的 `edge_scores`，表示这条边的重要性。
然后，为了实现真正的稀疏化（即 Hard Mask），我根据预设的稀疏化比例（例如保留 Top 10%），利用 `torch.topk` 找到当前 Batch 中重要性得分的阈值 (`thresh`)。接着，我将所有得分大于等于该阈值的边标记为 1，其余为 0，从而生成了一个非 0 即 1 的 `sparsification_mask`。”
### 2. 如果是 Hard Mask，反向传播时你是如何处理梯度截断的？

面试官问这个问题，是因为 `>= thresh` 以及 `.float()` 这种离散化操作在 PyTorch 中是不可微的，梯度在这里会断掉。如果不做特殊处理，被 Mask 掉的边永远没有机会恢复。

**回答话术：**
“为了解决 Hard Mask 带来的梯度截断问题，我采用了**掩码乘法结合辅助 Loss (Masked Soft-Weighting with Auxiliary Loss)** 的策略：

1. **对于前向传播的主任务（保留的边）：**
   在送入图卷积（BATConv）之前，我将可微的软得分与硬掩码相乘（`edge_weights_input = edge_scores * sparsification_mask`）。因为 `sparsification_mask` 只是一个常数（1 或 0），所以对于被保留的边（Mask=1），主任务（如分类预测）的梯度可以顺畅地通过 `edge_scores` 回传给 `EdgeScorer` 网络，指导它更新权重。

2. **对于梯度截断的处理（被丢弃的边）：**
   对于被丢弃的边（Mask=0），它们确实无法从最终的分类 Loss 中获得梯度。为了防止这些边‘一死百了’，我在应用 Mask **之前**，直接基于原始的 `edge_scores` 引入了一个**辅助稀疏化损失 (`sparsification_loss`)**。
   这个辅助 Loss 包含两部分：
   - **L1 稀疏性惩罚**：全局压低所有边的得分。
   - **连通性保持惩罚**：计算每个节点的边权重之和，如果低于最低阈值，则给予惩罚，强迫模型给某些边分配更高的分数以保证图的连通。

通过这种设计，即使一条边当前被 Mask 掉了，它依然能收到来自连通性辅助 Loss 的梯度。如果在后续的 Epoch 中，主任务需要用到这个节点，辅助 Loss 会推高它的 `edge_scores`，一旦超过阈值，它就会在下一次前向传播中被 Top-K 重新‘复活’。这样既实现了物理上的稀疏计算，又保证了全局梯度的流动与动态探索。”


## 训练部分
Focal Loss 是何恺明大神在 2017 年提出的一种专为解决**极端类别不平衡**问题的损失函数

### 1. 基础铺垫：什么是 $p_t$？
在推导之前，先统一定义一个变量 $p_t$（模型预测该样本为**真实类别**的概率）：
*   如果真实标签 $y = 1$（正样本），那么 $p_t = p$（模型预测为正的概率）。
*   如果真实标签 $y = 0$（负样本），那么 $p_t = 1 - p$（模型预测为负的概率）。

简而言之，**$p_t$ 越大，说明模型预测得越准（这个样本越容易分）；$p_t$ 越小，说明模型预测得越差（这个样本越难分）。**

---

### 2. 第一步：标准交叉熵（Cross Entropy, CE）有什么问题？
标准的二分类交叉熵公式是：
$$CE(p_t) = -\log(p_t)$$

**它的痛点在哪里？**
假设你的数据集中有 100 个正样本（休克患者），100,000 个负样本（普通患者）。
对于普通的负样本，模型很容易就能识别，比如输出 $p_t = 0.99$。此时它的 Loss 是 $-\log(0.99) \approx 0.004$。单个易分样本的 Loss 确实很小。
**但是！** 由于负样本数量极其庞大（10万个），$100,000 \times 0.004 = 400$。
而对于难分的正样本，假设 $p_t = 0.1$，Loss 为 $-\log(0.1) \approx 2.3$。100个正样本的总 Loss 只有 $100 \times 2.3 = 230$。

**结论：** 在极端不平衡下，海量“易分负样本”的微小 Loss 累加起来，会彻底淹没“难分正样本”的 Loss。模型梯度的更新方向被大量负样本带偏了。

---

### 3. 第二步：引入 $\gamma$ (Focusing Parameter) 解决“难易”问题
为了让模型不要被海量的“易分样本”分心，Focal Loss 在交叉熵前面乘上了一个**动态缩放因子** $(1 - p_t)^\gamma$：
$$FL(p_t) = -(1 - p_t)^\gamma \log(p_t)$$

**$\gamma$ 是如何大显神威的？（举例说明，面试时可以说出这组数字）**
假设设定 $\gamma = 2$：
*   **对于易分样本（比如 $p_t = 0.9$）**：
    $(1 - 0.9)^2 = 0.01$。这意味着，原本的 Loss 会被**缩小 100 倍**！
*   **对于难分样本（比如 $p_t = 0.1$）**：
    $(1 - 0.1)^2 = 0.81$。原本的 Loss 几乎没怎么变（只缩小了一点点）。

**结论：** $\gamma$ 的作用是**按难度动态分配权重**。模型预测得越准，惩罚力度衰减得越狠。这就强迫模型把有限的注意力（Focus）集中在那些总是预测错的难分样本上。当 $\gamma = 0$ 时，Focal Loss 就退化成了普通的交叉熵。

---

### 4. 第三步：引入 $\alpha$ (Balancing Parameter) 解决“数量”问题
$\gamma$ 解决了“难易不平衡”，但还没有完全解决“正负样本绝对数量不平衡”的问题。所以引入了固定权重 $\alpha_t$：
$$FL(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

*  通常，如果正样本极少，我们会给正样本设置一个较大的 $\alpha$（例如 $\alpha = 0.75$），给负样本设置一个较小的 $1 - \alpha$（例如 $0.25$）。
*   **结论：** $\alpha$ 纯粹是从**物理数量**上平衡正负类；而 $\gamma$ 是从**学习难度**上平衡易分/难分样本。

---

### 💡 结合 GraphCare 项目的面试高分话术

当面试官问到 Focal Loss 时，你可以这样一套连招打出来：

**“在我的 GraphCare ICU 辅助决策项目中，多标签药物推荐和心源性休克风险预测都面临着极其严重的数据长尾分布问题。如果直接使用 Cross Entropy，模型会被海量的、容易预测的常见病和常规用药（易分负样本）主导，导致在罕见药和休克高危患者上的 Recall 非常低。**

**为了解决这个问题，我引入了 Focal Loss。它的核心公式是 $-\alpha_t (1 - p_t)^\gamma \log(p_t)$。**

**这里面有两个非常精妙的参数设计：**
**第一个是 $\gamma$（聚焦参数）。我在项目中通常将其设为 2。它的作用是对 Loss 施加一个动态缩放因子 $(1-p_t)^\gamma$。如果一个常规药模型已经预测得很准了（比如 $p_t=0.9$），这个因子就会把它的 Loss 缩小 100 倍；而对于预测不准的罕见特效药，Loss 几乎不衰减。这就强迫 GNN 在反向传播时，把梯度更新的重心放在攻克‘难分样本’上。**

**第二个是 $\alpha$（平衡参数）。我在项目中通常将其设为 0.25。虽然 $\gamma$ 解决了难易问题，但正负样本的绝对数量差异依然存在。所以我根据数据集中正负标签的先验分布，为少数类分配了更高的 $\alpha$ 权重，从宏观数量层面进一步平衡了 Loss。**

**通过 $\alpha$ 和 $\gamma$ 的双管齐下，模型不再做‘和事佬’去迎合大众数据，从而大幅提升了少数类的识别率，这也直接反映在了最终 Macro-F1 指标的显著提升上。”**



## 验证部分
在多标签分类（比如你的项目中，一个病人可能同时被推荐多种药物）或多分类任务中，我们需要将多个类别的 F1-Score 聚合成一个单一的全局指标来评估模型。Micro-F1 和 Macro-F1 就是最常用的两种聚合方式。

**基础概念回顾**
F1-Score 是精确率（Precision）和召回率（Recall）的调和平均数。
- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)

**Micro-F1 (微平均)**
- **计算方式**：全局统算。它打破类别的界限，将所有类别对应的 True Positives (TP)、False Positives (FP)、False Negatives (FN) 直接累加起来，先算出一个全局的 Precision 和全局的 Recall，最后套用公式求出一个总的 F1。
- **核心特点**：**受多数类（大类）主导**。因为是按总体样本数量累加，样本量越大的类别对最终 Micro-F1 的贡献和影响就越大。
- **适用场景**：整体类别分布相对均衡，或者你只关心总体预测正确的绝对样本数，不在乎模型是否牺牲了少数类。

**Macro-F1 (宏平均)**
- **计算方式**：算术平均。它先独立地计算出每一个类别的 F1-Score，然后把所有类别的 F1 分数直接相加，再除以类别总数求平均值。
- **核心特点**：**众生平等，对少数类高度敏感**。无论某个类别有 10000 个样本还是只有 10 个样本，它们在 Macro-F1 中的权重都是 1/N。如果模型在哪怕一个样本量极小的罕见类上表现糟糕（比如 F1 为 0），都会大幅拉低最终的 Macro-F1 分数。
- **适用场景**：面临**严重的数据不平衡**，且业务场景要求模型必须在**少数类**上也有良好表现。

### 💡 结合 GraphCare 项目的面试高分话术

“在多标签评估中，Micro-F1 和 Macro-F1 的核心区别在于它们对待数据不平衡的态度。Micro-F1 是全局样本统计，受多数类主导；而 Macro-F1 是各类平均，对少数类的表现非常敏感。

在我的 GraphCare ICU 医疗辅助决策项目中，我们面临着**极度严重的长尾分布（数据不平衡）**。比如，β-内酰胺抗菌药，青霉素类(某一类青霉素药) 用得很多，样本量极大；而特效药样本量极小。

如果我们的评估指标只看 Micro-F1，模型只要无脑推荐通用药，分数就会看起来非常高，但这在临床决策中是毫无意义的。**因此，我们在该项目中重点关注 Macro-F1**，因为它能真实反映模型对罕见药物和高危休克风险的联合预测能力。

## 反馈模块
- 这部分我做的是一个用户反馈闭环。第一次推理完成后，医生可以输入自然语言反馈，比如‘患者有严重心衰，需要强化相关风险因素’或者‘患者对某类药物存在禁忌，需要弱化对应推荐’。系统不会直接把这段文本拼进 prompt 就结束，而是把它转换成图模型可执行的结构化操作。
- 具体流程是四步。
	1.  LLM 将反馈拆成 add/remove 两类关键词。
	2. 把这些关键词编码成 embedding。
	3. 用 HNSW 在预先构建好的 cluster embedding 索引中做近似最近邻搜索，快速召回最相关的 cluster 候选；如果有多个关键词，我会把候选集合并，再做一次精确相似度重排。
		- **HNSW 的作用** ：在一大堆向量里， 快速近似地找到和当前查询向量最相近的若干个向量
			- “被搜索的向量”就是 **cluster embedding**
			- “当前查询向量”就是 **关键词的 embedding**
		- 所以它做的事可以表述为：给一个关键词 embedding，快速找到最相近的若干个 cluster embedding, 这反应的是原先的**聚类节点与关键词节点的相关性**
		- **重排的作用**是将每一个候选cluster的embedding与所有关键词embedding的余弦相似度求和, 反应的是**聚类节点与新增描述的相关性**. 
		- 参数
			- max_elements, 根据cluster数量动态决定
			- M=32: 经验值. 每个节点最多保留 32 条近邻边，保证图结构有较好的连通性
			- ef_construction=200: 经验值, 建图时每次插入节点都会在更大的候选范围里挑邻居，换更高质量的索引
			- ef_search=100: 查询时在 100 个候选范围内继续搜索，以换取更好的召回率
	4. 第四步，把最终得到的 cluster index 用于修改患者样本里的 ehr_node_set ，也就是增加或移除某些 cluster 节点，再触发二次图推理。”
- 我在推理时动态改变当前患者子图中被激活的医学概念节点 。HNSW 的作用主要是把原本线性扫描所有 cluster 的高耗时过程，变成低延迟的近似最近邻检索，让这个反馈闭环可以做到交互式响应。”

<<<<<<< HEAD
=======
>>>>>>> f81516b86c4e8635008168039021ce6d584b6c55
>>>>>>> 3c428a2 (chore: 初始化仓库，清理目录结构，合并编译原理作业)
