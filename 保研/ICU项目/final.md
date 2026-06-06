# 第二章 核心作用机制

## 2.1 LLM增强的知识图谱构建

### 2.1.1 基础图谱定义

依据 MIMIC-III 的既有节点库与关系，首先构建基础知识图谱的节点与边集合。令节点集记为 $V=V_{\mathrm{diag}}\cup V_{\mathrm{proc}}$，其中诊断节点 $V_{\mathrm{diag}}=\{d_1,\dots,d_{D_d}\}$、手术节点 $V_{\mathrm{proc}}=\{p_1,\dots,p_{D_p}\}$，$d_k$ 和 $p_k$ 分别表示按固定全局字典排序的第 $k$ 个诊断/手术类别。既有关系类型集记为 $\mathcal{R}_0$（如同次就诊共现、顺序相邻、依从性关系等），初始三元组集为 $E_0\subseteq V\times \mathcal{R}_0\times V$，记为 $(u,r,v)$。为保证语义正确性，引入类型映射 $\mathrm{type}:V\to\{\mathrm{diag},\mathrm{proc}\}$ 以及关系的定义域/值域约束 $\mathrm{dom},\mathrm{cod}:\mathcal{R}_0\to\{\mathrm{diag},\mathrm{proc},\mathrm{any}\}$，仅在满足 $\mathrm{type}(u)\in \mathrm{dom}(r)$、$\mathrm{type}(v)\in \mathrm{cod}(r)$ 时保留边 $(u,r,v)$。

### 2.1.2 LLM生成与融合

传统知识图谱依赖人工标注或固定规则，难以覆盖长尾医学关系。为融合数据事实与医学先验，在上述基础上，为每一个原始节点 $u\in V$ 使用大语言模型自动生成新的关系—尾节点对，从而扩充三元组集。对节点 $u$，模型输出候选集 $\mathcal{G}(u)=\{(r,v,s(u,r,v))\}$，其中 $r\in \mathcal{R}_{\mathrm{LLM}}$、$v\in V$、$s(u,r,v)\in[0,1]$ 为生成的置信度。通过阈值筛选合并，得到由大语言模型生成的三元组集

$$E_{\mathrm{LLM}}=\{(u,r,v)\mid u\in V,\ (r,v,s)\in \mathcal{G}(u),\ s\ge \tau\}$$

并与既有关系并集为 $E=E_0\cup E_{\mathrm{LLM}}$。为后续聚合与可解释性，定义边权

$$w(u,r,v)=\begin{cases}1,& (u,r,v)\in E_0\\s(u,r,v),& (u,r,v)\in E_{\mathrm{LLM}}\end{cases}$$

由于 $entity1$ 固定取自 MIMIC-III 的原始节点库、而 $relation$ 与 $entity2$ 由大语言模型生成并经置信度阈值与类型约束筛选，该图谱构建在保证临床语义一致性的同时，充分融合数据事实与医学先验，为后续时序感知的图神经网络推理提供规模可控、结构可解释的全局知识底座。

## 2.2 语义聚类与超图构建

为解决大规模图谱的计算瓶颈，本文基于节点语义相似度进行聚类压缩，将语义相近的节点聚合为超节点。该策略在保持关键关联结构的同时，降低了子图复杂度，使得推理时仅需激活与患者当前临床情境相关的节点及其邻域，即可构建紧凑的局部推理子图。

### 2.2.1 语义聚类

令语义嵌入映射为 $\varphi:V\to\mathbb{R}^d$，相似度度量（余弦）为

$$\mathrm{sim}(u,v)=\frac{\varphi(u)^\top \varphi(v)}{\|\varphi(u)\|_2\|\varphi(v)\|_2}\in[-1,1]$$

以 $k$-means 为例，求解分区 $\mathcal{C}=\{C_1,\dots,C_K\}$ 与簇中心 $\{\mu_a\}$ 满足

$$\min_{\{C_a\},\{\mu_a\}}\sum_{a=1}^{K}\sum_{u\in C_a}\|\varphi(u)-\mu_a\|_2^2,\quad\mu_a=\frac{1}{|C_a|}\sum_{u\in C_a}\varphi(u)$$

由此构造超节点集合 $V^\star=\{C_1,\dots,C_K\}$，每个超节点为一个语义簇，其代表嵌入取为簇中心 $\mu_a$。

### 2.2.2 超图诱导

在关系层面，依据原图 $G=(V,\mathcal{R},E)$（其中 $\mathcal{R}=\mathcal{R}_0\cup\mathcal{R}_{\mathrm{LLM}}$）诱导超图 $G^\star=(V^\star,\mathcal{R},E^\star)$，其中超边集定义为

$$E^\star=\{(C_a,r,C_b)\mid \exists\, u\in C_a,\ \exists\, v\in C_b:\ (u,r,v)\in E\}$$

为提升可解释性与便于推理加权，超边权通过原始边权聚合得到

$$w^\star(C_a,r,C_b)=\mathrm{Agg}\big(\{w(u,r,v)\mid u\in C_a,\ v\in C_b,\ (u,r,v)\in E\}\big)$$

其中 $\mathrm{Agg}$ 可取计数、最大值或加权平均等。该构造满足关联保持原则：若两个超节点的任意原始节点之间存在关系，则对应超节点之间亦存在关系，从而在压缩节点空间的同时不丢失核心关系语义。

上述过程最终形成全局知识超图 $G^\star=(V^\star,\mathcal{R},E^\star)$，其中 $|V^\star|=K\ll |V|$。在语义紧致与临床关系局部性的前提下，通常有 $|E^\star|\lesssim |E|$，超图的平均度 $\bar{d}^\star=\frac{|E^\star|}{|V^\star|}$ 不高于原图平均度 $\bar{d}=\frac{|E|}{|V|}$，从而显著降低推理时的邻域展开与消息传递成本。

## 2.3 时序感知的图注意力推理

推理层在全局知识超图的基础上，围绕患者当前就诊的症状与手术信息，构建紧凑的个体化子图，并在该子图上实施时序感知的图注意力推理。该过程包含"超节点激活—邻域扩展—时序加权—注意力聚合"四个环节。

### 2.3.1 超节点激活与个体化子图构建

设由知识层得到的全局超图为 $G^\star=(V^\star,\mathcal{R},E^\star)$，其中 $V^\star=\{C_1,\dots,C_K\}$ 为语义聚类形成的超节点集合，$E^\star\subseteq V^\star\times\mathcal{R}\times V^\star$ 为关系三元组集合。令聚类分配函数为 $\kappa:V\rightarrow\{1,\dots,K\}$，将原始诊断/手术节点映射至其所属超节点；诊断输入集合在数据层已剔除心源性休克诊断，记为 $\bar{S}_i^{\text{diag}}(t)\subseteq\mathcal{D}\setminus\{d_{\mathrm{CS}}\}$，手术集合为 $S_i^{\text{proc}}(t)\subseteq\mathcal{P}$。为患者 $i$ 的当前就诊（序号记为 $T_i$）定义基激活超节点集合

$$A_i^{(0)}=\big\{C_{\kappa(u)}\ \big|\ u\in \bar{S}_i^{\text{diag}}(T_i)\cup S_i^{\text{proc}}(T_i)\big\}\subseteq V^\star$$

在超图 $G^\star$ 上定义无权最短路径距离 $\mathrm{dist}_{G^\star}:V^\star\times V^\star\rightarrow\mathbb{N}\cup\{0,\infty\}$。给定半径参数 $k\in\mathbb{N}$，以基激活集为中心进行邻域扩展，得到患者个体化子图的节点集合

$$V_i=\big\{v\in V^\star\ \big|\ \exists\,u\in A_i^{(0)}:\ \mathrm{dist}_{G^\star}(u,v)<k\big\}$$

并取诱导子图 $G_i=G^\star[V_i]$。其邻接关系在 $G^\star$ 中保留，记患者子图上的邻域为

$$\mathcal{N}_i(u)=\big\{v\in V_i\ \big|\ \exists\,r\in\mathcal{R}:\ (u,r,v)\in E^\star\big\}$$

### 2.3.2 时序衰减权重

为体现历史就诊对当前决策的影响并避免标签泄露，本文以时序衰减对节点初始表示进行加权。令超节点的静态语义嵌入为 $\mu_v\in\mathbb{R}^d$。对每个 $v\in V_i$，定义其最近一次被历史就诊激活（含半径 $k$ 的邻域扩展）所在的时间步

$$t_i^\ast(v)=\max\big\{t\in\{1,\dots,T_i\}\ \big|\ v\in B_k(A_i^{(t)})\big\}$$

其中 $B_k(A)=\{w\in V^\star\mid \exists\,u\in A:\ \mathrm{dist}_{G^\star}(u,w)<k\}$ 为集合 $A$ 的 $k$-邻域。若 $v$ 在历史中从未被激活，则令 $t_i^\ast(v)$ 不存在并相应地赋零权重。设时间衰减函数为 $f(\Delta)=\exp(-\lambda\Delta)$，$\lambda>0$，并以就诊序号差 $\Delta_i(v)=T_i-t_i^\ast(v)$ 近似时间间隔。节点的时序加权初始表示定义为

$$h_v^{(0)}=f(\Delta_i(v))\cdot \mu_v$$

或在需要累积多次激活贡献时取

$$s_i(v)=\sum_{t=1}^{T_i} f(T_i-t)\cdot \mathbf{1}_{\{v\in B_k(A_i^{(t)})\}},\quad h_v^{(0)}=s_i(v)\cdot \mu_v$$

其中 $\mathbf{1}_{\{\cdot\}}$ 为指示函数。上述设计使得近期被激活的节点获得更高的初始表示权重，而长期未出现的节点则趋近零权重。

### 2.3.3 时序增强的图注意力机制

在患者子图 $G_i$ 上采用图注意力网络（GAT）进行多层消息传递与特征聚合。第 $l$ 层的原始注意力打分定义为

$$e_{uv}^{(l)}=\mathrm{LeakyReLU}\Big(\mathbf{a}^{(l)\top}\big[\mathbf{W}^{(l)}h_u^{(l)}\ \Vert\ \mathbf{W}^{(l)}h_v^{(l)}\big]\Big),\quad u\in V_i,\ v\in\mathcal{N}_i(u)$$

其中 $\mathbf{W}^{(l)}$ 为线性变换，$\mathbf{a}^{(l)}$ 为注意力向量，$\Vert$ 表示向量拼接。将时间衰减作为临床近期性约束融入注意力归一化，令

$$\alpha_{uv}^{(l)}=\mathrm{softmax}_{v\in \mathcal{N}_i(u)}\Big(e_{uv}^{(l)}\cdot f(\Delta_i(v))\Big)$$

得到更新方程

$$h_u^{(l+1)}=\sigma\left(\sum_{v\in \mathcal{N}_i(u)} \alpha_{uv}^{(l)}\ \mathbf{W}^{(l)}h_v^{(l)}\right)$$

其中 $\sigma(\cdot)$ 为非线性激活。对多头注意力，令 $m=1,\dots,M$，则有

$$h_u^{(l+1)}=\big\Vert_{m=1}^{M}\ \sigma\left(\sum_{v\in \mathcal{N}_i(u)} \alpha_{uv}^{(l,m)}\ \mathbf{W}^{(l,m)}h_v^{(l)}\right)$$

末层可对各头取均值或拼接后经线性投影以形成最终表示。若超边存在来源权重（如由原始节点关系聚合得到的 $w^\star$），可进一步将结构置信度作为先验因子并入注意力打分，例如 $e_{uv}^{(l)}\leftarrow e_{uv}^{(l)}\cdot g(w^\star(u,r,v))$，以提升关系强度大的边在聚合中的影响。

上述推理层在不引入目标诊断至输入侧的前提下，通过"当前就诊驱动的超节点激活＋半径 $k$ 的邻域扩展"构建患者个体化子图，并以指数型时间衰减 $f(\Delta)=\exp(-\lambda\Delta)$ 强化近期事件的权重，将"近期信息更重要"的临床认知融入注意力归一化过程。该设计既保证了实时推理的计算可控性，又在多层消息传递中充分利用图结构与时序上下文，为后续各任务的判别与生成（如药物推荐与心源性休克风险估计）提供高质量的节点与子图表示。

## 2.4 交互式图谱更新

交互层旨在实现以自然语言为入口的人机协同决策，通过术语识别、语义对齐与动态图谱更新，将临床人员提供的补充信息规范化并与知识超图高效融合，进而在患者全序列样本的基础上触发个体化子图的再构建与时序感知推理。

### 2.4.1 术语识别与语义对齐

用户交互输入记为一段中文文本 $U_i$（对应患者 $i$ 的当前就诊情境）。经领域适配的中文术语识别，提取候选术语集合 $T_i=\{\tau_1,\dots,\tau_M\}$，并为每个术语计算嵌入 $\varphi(\tau_j)\in\mathbb{R}^d$。将术语与知识图谱实体进行语义对齐，定义对齐算子

$$\mathrm{align}(\tau_j)=\arg\max_{v\in V}\mathrm{sim}(\varphi(\tau_j),\varphi(v))$$

并以阈值判定其有效性，令 $s(\tau_j)=\mathrm{sim}(\varphi(\tau_j),\varphi(\mathrm{align}(\tau_j)))$，仅当 $s(\tau_j)\geq\tau_{\mathrm{align}}$ 时保留该对齐结果。

### 2.4.2 动态图谱更新

交互层的动态图谱更新在患者级子图上实施，不改变全局聚类结构。交互层将经对齐的术语集合映射至超节点，增量地更新当前就诊的激活集合：

$$A_i^{(T_i)}\leftarrow A_i^{(T_i)}\cup\big\{\pi(\mathrm{align}(\tau_j))\ \big|\ \tau_j\in T_i,\ s(\tau_j)\geq\tau_{\mathrm{align}}\big\}$$

在超图 $G^\star$ 中以无权最短路径距离 $\mathrm{dist}_{G^\star}$ 和半径参数 $k$ 进行邻域扩展，得到更新后的患者子图节点集合 $V_i=\bigcup_{t=1}^{T_i}B_k(A_i^{(t)})$ 并取子图 $G_i=G^\star[V_i]$。该过程使得用户新增的术语即时纳入当前就诊的激活范围，并通过邻域扩展影响患者子图的结构。

### 2.4.3 时序整合

在模型层，交互术语在当前就诊 $T_i$ 被纳入激活，因而其贡献项以 $f(0)=1$ 的最大权重加入时间加权激活强度，从而在后续注意力归一化与末层池化中对近期信息赋予更高影响。

## 2.5 关键性下沉分配机制

为增强交互可解释性，本文对末层节点表示的关键性进行评估。然而，超节点的可解释性难以直接映射至临床可读的原始诊断/手术节点，因此本文提出一种"关键性下沉分配"机制：在保持模型末层关键性度量不变的前提下，将超节点级关键性沿聚类成员关系与患者时序激活轨迹，严格、可控地分配至原始节点，从而支持原始节点级的可视化与路径溯源。

### 2.5.1 末层关键性度量

定义超节点的末层关键性度量为

$$\gamma(V_a)=\|h_{V_a}^{(L)}\|_2\geq 0$$

其中 $h_{V_a}^{(L)}$ 为图注意力网络在患者子图 $G_i$ 上第 $L$ 层的节点表示。根据 $\gamma(V_a)$ 对图中超节点进行排序，选取 Top-$k$ 关键超节点。

### 2.5.2 三类责任权重

定义基于三类证据的归一化"责任权重"并保持总量守恒：

**时序存在性权重**。采用指数衰减 $f(\Delta)=\exp(-\lambda\Delta)$，$\lambda>0$，以就诊序号差近似时间间隔。原始节点 $u$ 的时序存在性权重定义为

$$p_i(u)=\sum_{t=1}^{T_i}f(T_i-t)\cdot \mathbf{1}_{\{u\in \bar{S}_i^{\text{diag}}(t)\cup S_i^{\text{proc}}(t)\}}$$

其中 $\mathbf{1}_{\{\cdot\}}$ 为指示函数。若 $u$ 未在患者序列中出现，则 $p_i(u)=0$。

**结构桥接权重**。记 $E_i$ 为在患者子图 $G_i$ 中实际参与到超边诱导的原始边集合（即在 $E$ 中使 $(V_a,r,V_b)\in E^\star$ 成立的原始边），则定义

$$q_i(u)=\sum_{(u,r,v)\in E_i}1$$

以衡量 $u$ 在患者子图跨簇连接中的桥接参与度。该项提升临床上结构性关键节点的可见度。

**语义贴合权重**。令原始节点嵌入为 $\varphi(u)\in\mathbb{R}^d$，簇中心为 $\mu_a\in\mathbb{R}^d$，则语义贴合度取非负余弦相似

$$s(u)=\max\left(0,\frac{\varphi(u)^\top \mu_a}{\|\varphi(u)\|_2\|\mu_a\|_2}\right)$$

### 2.5.3 加权融合与下沉

将三类证据按超参数加权融合，并在簇内归一化：

$$\tilde{\rho}(u\mid V_a)=\alpha_p\frac{p_i(u)}{\sum_{w\in U(V_a)}p_i(w)}+\alpha_q\frac{q_i(u)}{\sum_{w\in U(V_a)}q_i(w)}+\alpha_s\frac{s(u)}{\sum_{w\in U(V_a)}s(w)}$$

其中 $\alpha_p,\alpha_q,\alpha_s\geq 0$，$\alpha_p+\alpha_q+\alpha_s=1$。进一步归一化：

$$\rho(u\mid V_a)=\frac{\tilde{\rho}(u\mid V_a)}{\sum_{w\in U(V_a)}\tilde{\rho}(w\mid V_a)}\in[0,1]$$

据此，原始节点级关键性下沉分配为

$$\gamma_i(u)=\gamma(V_a)\cdot\rho(u\mid V_a),\quad u\in U(V_a)$$

该定义满足以下性质：

- （保守性）$\sum_{u\in U(V_a)}\gamma_i(u)=\gamma(V_a)$，即关键性总量守恒；
- （非负性）$\gamma_i(u)\geq 0$；
- （时序敏感性）近期出现且结构桥接强的原始节点将获得更高的分配权重。

最终，在选择的关键超节点内按 $\gamma_i(u)$ 全局排序，选取 Top-$K$ 原始节点，保留其原始边信息进行可视化，实现临床可解释的推理路径溯源。
