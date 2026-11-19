**原文**（两组、三篇）：

1. Sander Abnar, Willem Zuidema. *Quantifying Attention Flow in Transformers*. ACL 2020, pp. 4190–4197. DOI `10.18653/v1/2020.acl-main.385`，arXiv `2005.00928`。Zotero key `8YGJR7BM`。
2. Goro Kobayashi et al. *Attention is Not Only a Weight: Analyzing Transformers with Vector Norms*. EMNLP 2020. DOI `10.18653/v1/2020.emnlp-main.583`，arXiv `2004.13307`。Zotero key `VXASASC9`。
3. Goro Kobayashi, Tatsuki Kuribayashi, Sho Yokoi, Kentaro Inui. *Incorporating Residual and Normalization Layers into Analysis of Masked Language Models*. EMNLP 2021, pp. 4547–4568. DOI `10.18653/v1/2021.emnlp-main.373`，arXiv `2109.07152`。Zotero key `Z4A435NT`。

配套术语见 [[注意力可解释性术语详解|术语详解]]；与之同批复现的视觉侧工作见 [[Chefer-Transformer可解释性-相关性传播|Chefer 相关性传播]]。

## 一、原文做的是什么工作

### 1.1 两篇论文在吵同一件事

出发点都是同一句话：**"注意力权重 $\alpha_{ij}$ 到底能不能当作 token $j$ 对 token $i$ 的贡献？"**

- Abnar & Zuidema 说：**单层看不行**，因为信息要经过很多层才走到目标位置，只看最后一层的注意力会漏掉底层的路径。要"把整条路径上的流量加起来"。
- Kobayashi 说：**换个维度也不行**，因为注意力权重只是一个标量系数，真正进入残差流的是 $\alpha_{ij}\cdot f(x_j)$ 这个**向量**。向量长度差很多的话，权重高不等于影响大。

两者是互补的两条修正路线：一条修"跨层聚合"，一条修"权重 vs 向量范数"。本次复现把两条放在同一组句子上对撞。

### 1.2 Abnar & Zuidema：Attention Rollout 与 Attention Flow

**先把每层注意力补上残差**：Transformer 每层是 $h \leftarrow h + \text{Attn}(h)$，所以"留在原地"的那部分也要算进容量。定义

$$\tilde{A}^{(l)} = 0.5\,A^{(l)} + 0.5\,I$$

行和仍为 1，对角线上的 0.5 就是残差通道。

**方法一：Attention Rollout（递归连乘）**

$$R^{(1)}=\tilde{A}^{(1)},\qquad R^{(l)} = \tilde{A}^{(l)} R^{(l-1)}$$

取 $R^{(L)}$ 的第 $i$ 行，就是"目标位置 $i$ 最终看到的信息里，有多大比例来自各个输入 token"。本质是矩阵连乘，把所有路径的乘积全部累加（含绕路、含回头路）。便宜，$O(Ln^3)$。

**方法二：Attention Flow（最大流）**

把整个 Transformer 展开成一张有向图：$L+1$ 层节点，每层 $n$ 个节点，节点 $(l,k)$ 表示"第 $l$ 层位置 $k$ 的表示"。层 $l$ 到层 $l-1$ 的边容量就是 $\tilde{A}^{(l)}$ 的元素。然后问：**从顶层的查询节点到某个输入节点 $v$，这张容量网络最多能送过去多少流量？** 这就是最大流 $F(v)$。

与 rollout 的区别：最大流是"瓶颈约束"下的量，它不会把互相抵消的路径加起来，而是看最窄的那条通道有多宽。原文主张这更符合"信息量"的直觉，代价是要解 $n$ 次最大流，慢得多。

原文用合成任务和语言理解任务展示 flow 比 raw attention 更贴合人工设计的注意力模式。**原文没有报告过 flow 在真实 BERT 上会退化成均匀分布的情况**——这正是本次复现发现的东西。

### 1.3 Kobayashi 2020：范数才是贡献

对每个头 $h$，源 token $s$ 经 value 投影和输出投影后的向量是

$$f^{(l,h)}(x_s) = W_O^{(h)}\big(W_V^{(h)}x_s + b_V^{(h)}\big)$$

于是 token $s$ 对目标 $t$ 的实际贡献大小应该用

$$\big\|\alpha_{ts}\,f^{(l,h)}(x_s)\big\|_2 = \alpha_{ts}\,\big\|f^{(l,h)}(x_s)\big\|_2$$

来衡量，而不是 $\alpha_{ts}$ 本身。原文据此发现了**排名反转**：有些 token 权重很高但范数很小（白看了），有些权重很低但范数很大（被忽略的主力）。原文的实证重点之一是 `[SEP]` 这类特殊 token 往往属于前者。

**实现要点**：$W_O$ 必须**按头切列**（`Wo[:, h*dh:(h+1)*dh]`）。HF 的 `W_O` 是拼在一起的，直接乘会得到混合了所有头的错误向量——这是最容易写错的地方，本次用暴力双重循环校验过。

### 1.4 Kobayashi 2021：把残差和 LayerNorm 也算进去

2020 版还有一个漏洞：$f(x_s)$ 加进残差流之后，还要过 LayerNorm，而 LN 会重新缩放整个向量。所以 2021 版定义

$$g_t(x_s)=\mathrm{LN}\big(h_{l-1}[t] + b_O + \alpha_{ts} f(x_s)\big),\qquad \text{score}(s)=\|g_t(x_s)\|-\|g_t(0)\|$$

即"把 $s$ 的贡献放进去后，目标位置表示的范数**变化量**"。这个量**可正可负**：负值意味着这个 token 加进来之后反而把目标表示压缩了。原文据此指出，计入残差与归一化后，"注意力权重高 = 贡献大"甚至可能完全反向。

## 二、本次复现验证什么

| 编号  | 待验证命题                                                       | 来源                       |
| --- | ----------------------------------------------------------- | ------------------------ |
| H1  | 在 Winograd 最小对比对上，归因方向会随形容词翻转（`tired`→animal，`wide`→street） | 归因方法的基本要求，两篇原文都没在这个探针上测过 |
| H2  | rollout / flow 比 raw attention 更可靠                          | Abnar 原文的核心主张            |
| H3  | 存在权重排名与范数排名反转的 token，且集中在特殊 token 上                         | Kobayashi 2020           |
| H4  | 计入残差 + LN 后，2021 score 与 $\alpha$ 的相关性显著变差甚至转负              | Kobayashi 2021           |

探针句子（只改最后一个形容词）：

- `The animal didn't cross the street because it was too tired` → gold：`it = animal`
- `The animal didn't cross the street because it was too wide` → gold：`it = street`

模型：`bert-base-uncased` 与 `roberta-base`（均 12 层 12 头）。

## 三、怎么判断成功（判据设计）

这一路的判据比视觉那一路（[[Chefer-Transformer可解释性-相关性传播|Chefer 文档]]）更严格，因为结论是**否定性**的，必须先把"是不是自己写错了"排除干净。设计了四道关卡：

### 关卡 0：先确认模型自己会做这道题（cloze 探针）

把 `it` 换成 `[MASK]`，看模型给 `animal` 和 `street` 的概率：

$$\text{ratio} = \frac{p(\text{gold})}{p(\text{other})}$$

**为什么要这一步**：如果模型本身都分不清指代，那后面所有归因方法"失败"都是废话——不是方法的锅。只有先证明模型会做题，归因失败才有意义。

**判定**：ratio ≫ 1（数量级差异）即认为模型解出了指代。

### 关卡 1：归因的 gold/distractor 比值

在**不带 mask 的原句**上，取源 token `it` 的注意力行，比较四种方法给两个候选的分数：

$$\text{ratio} = \frac{\text{score}(\text{gold})}{\text{score}(\text{distractor})}$$

**判定**：ratio > 1 方向正确，< 1 方向错误，≈ 1 无区分度。要求**两句都正确**才算方法可靠——只对一句正确很可能是碰巧。

### 关卡 2：sanity check（排除实现错误）

结论是"方法失败"时，最大的嫌疑永远是自己代码写错。所以每一项实现都配了独立的数值校验：

| 校验项 | 手段 | 通过阈值 |
| --- | --- | --- |
| Kobayashi 范数定义 | 暴力双重循环重算 $f^{(h)}$ | 完全一致 |
| 分解式能重构真实输出 | 验证 $\sum_h \alpha^{(h)} f^{(h)}(x) + b_O$ = HF 的 attention 输出 | float32 噪声量级（~1e-6） |
| 手工重算 encoder | 从 embedding 逐层重算，与 HF `output_hidden_states` 对比 | 相对误差 ~1e-7 |
| rollout 公式 | 手算的 2 层 $2\times2$ 闭式例子 | 完全一致 |
| $L=1$ 时 flow 的闭式解 | 单层时 flow 应严格等于行归一化的 $(W+I)$ | 完全一致 |
| 两套最大流实现 | networkx（Edmonds–Karp, float）vs scipy（Boykov–Kolmogorov, int） | ~1e-7 |
| 残差写法等价性 | `add_residual=True, w=0.5` vs 直接折叠 $\tilde A=0.5A+0.5I$ | 0.0 |

**判定逻辑**：全部通过后，"flow 输出像均匀分布"就只能解释为**方法本身的性质**，而不是 bug。

### 关卡 3：退化诊断（定位失败机理）

发现 flow 退化后，不能只说"它不行"，要说清**为什么不行**。设计了三个诊断：

1. **未归一化 flow**：归一化会掩盖信息，直接看最大流原值 $F(v)$。若全部顶在源节点出容量上，即为**饱和**。
2. **min-cut 分析**：最大流 = 最小割。查割的容量卡在哪一侧（源侧 / 中间 / 汇侧），以及 $|S|$（割集大小）。
3. **深度扫描 + 逐头**：把源节点从第 12 层往下挪，看 flow 从第几层开始饱和；再对 12 个头单独跑，排除"头平均把稀疏性抹平了"这个替代解释。

**判定**：若单头也饱和，则退化与头平均无关，是容量结构本身的问题。

## 四、实验怎么做的

### 4.1 环境

| 项 | 取值 |
| --- | --- |
| conda 环境 | `D:\Env\minConda\envs\attn-repro`（Python 3.10.22） |
| 关键库 | torch 2.13.0+cu130、transformers 4.57.1、networkx 3.4.2、scipy 1.15.3 |
| 硬件 | RTX 5060 Laptop 8GB，但**全程强制 CPU**（`CUDA_VISIBLE_DEVICES=""`，`torch.set_num_threads(8)`） |
| 权重缓存 | `HF_HOME=D:\checkpoints\hf` |

选 CPU 的原因：最大流求解需要完全确定的数值（GPU 的 SDPA kernel 与 cuDNN 归约顺序会带来微小差异），而且要与手工重算逐位对齐。代价是慢（每句约 46–48 s，其中 $n\times n$ 全量 flow 矩阵占 32–34 s）。

### 4.2 两套 flow 实现并存互验

- `attention_graph_util.py`：官方 attention_flow 仓库文件的移植版。去掉 tensorflow 依赖，`nx.from_numpy_matrix` → `nx.from_numpy_array`（networkx 3.x 已移除前者），`get_adjmat` 增加 `add_residual` / `residual_weight` 开关，绘图改纯 matplotlib。节点编号 $idx(l,k)=l\cdot n+k$ 与官方一致。
- `attn_methods.py`：独立重写版。`build_flow_graph` 用 networkx + Edmonds–Karp；`flow_relevance_from_capacities` 用 `scipy.sparse.csgraph.maximum_flow`（整数容量，缩放 $10^6$）。

两份代码独立写、结果对撞，是关卡 2 的一部分。

### 4.3 执行顺序

在 `D:\code\attn-repro\B\` 下：

| 步骤 | 命令 | 作用 |
| --- | --- | --- |
| 1 | `python test_methods.py` | 6 项纯数值自检，不需要模型。输出 `all self-tests passed` |
| 2 | `python run_winograd.py --model bert-base-uncased --tag bert` | BERT 上四方法末层对比表 + 逐层曲线 + 图 |
| 3 | `python run_winograd.py ... --quick` | 同上但跳过 $n\times n$ 全量 flow（计时探针，用于确认 `--quick` 不影响对比表） |
| 4 | `python task1_flow_rollout.py` | BERT + RoBERTa 的 cloze 探针、raw/rollout/flow 归因、flow 图可视化 |
| 5 | `python task2_kobayashi.py` | 手工重算 encoder + 一致性校验 + 2020 范数 / 2021 score / 排名反转 |
| 6 | `python diag_impl.py` | 对比 `default / eager / sdpa` 三种 `attn_implementation` 与手工重算的偏差 |
| 7 | `python diag_task2.py` | 逐步骤（embedding → 第 1 层 attention → FFN）定位偏差来源 |
| 8 | `python diag_flow.py` | flow 退化诊断：4 种容量变体、min-cut、深度扫描、逐头 flow |

步骤 6–8 是**先看到异常结果、再回头加的诊断**，不是一开始就设计好的。

### 4.4 三种测量方式的分工

| 方式 | 做法 | 用途 |
| --- | --- | --- |
| cloze 探针 A | 把 `it` 换成 `[MASK]`/`<mask>`，读 $p(\text{animal})$、$p(\text{street})$ | 关卡 0，确认模型会做题 |
| cloze 探针 B | 把先行词名词 mask 掉，看 top-5 补全 | 旁证模型的语义常识 |
| 归因 | 在**不带 mask 的原句**上取 `it` 的行 | 关卡 1，四方法对比 |
| Kobayashi 分析 | 目标位置 $t$ 取 `[MASK]` 本身（12 层 × 12 头平均） | 关卡 H3/H4 |

注意最后两行的目标位置不同：归因用原句的 `it`，Kobayashi 用 cloze 句的 `[MASK]`。这是因为 Kobayashi 分析要的是"模型在决定填什么词时看了哪里"，用 mask 位置更贴合。

## 五、结果

### 5.1 关卡 0 通过：模型确实解出了指代

| 模型 | 句子 | mask 处 argmax | $p(\text{gold})$ | $p(\text{other})$ | ratio |
| --- | --- | --- | --- | --- | --- |
| BERT | too tired（gold=animal） | `it` (0.8497) | 2.66e-06 | 4.42e-08 | **60.13** |
| BERT | too wide（gold=street） | `it` (0.9680) | 1.71e-05 | 6.03e-07 | **28.46** |
| RoBERTa | too tired（gold=animal） | `it` (0.8301) | 9.22e-07 | 1.97e-09 | **469.20** |
| RoBERTa | too wide（gold=street） | `it` (0.9987) | 1.56e-06 | 3.70e-10 | **4210.89** |

两个模型都把 gold 先行词压过干扰项 1～3 个数量级。**后面归因方法的失败不能推给"模型没学会"。**

（argmax 都是 `it` 而不是名词——这符合预期，因为 MLM 是填 mask 位置的词，模型倾向于还原原词；ratio 才是关键指标。探针 B 也正常：`tired` 句 BERT top-1 是 `car`(0.2501)、RoBERTa 是 `dog`(0.4320)；`wide` 句两者都是 `road`(0.1241 / 0.2784)，语义合理。）

### 5.2 关卡 1：四种方法没有一个两句都对

BERT 最后一层（L12），源 token = `it`(idx 10)，分数已对全部 15 个 token 行归一化。

**句 1 `... too tired`，gold = animal(idx 2)，distractor = street(idx 8)**

| 方法 | gold 分数 | gold 排名 | dist 分数 | dist 排名 | **ratio** | top-1 token |
| --- | --- | --- | --- | --- | --- | --- |
| Raw attention | 0.02820 | 6 | 0.00933 | 15 | **3.022** ✓ | `'` |
| Attention rollout | 0.02883 | 11 | 0.03068 | 10 | 0.939 ✗ | `[CLS]` |
| Attention flow | 0.07116 | 1 | 0.07116 | 1 | **1.000** — | `[CLS]` |
| Vector norm (2020) | 0.11665 | 2 | 0.03080 | 15 | **3.788** ✓ | `it` |

**句 2 `... too wide`，gold = street(idx 8)，distractor = animal(idx 2)**

| 方法 | gold 分数 | gold 排名 | dist 分数 | dist 排名 | **ratio** | top-1 token |
| --- | --- | --- | --- | --- | --- | --- |
| Raw attention | 0.01262 | 11 | 0.02794 | 5 | **0.452** ✗ | `'` |
| Attention rollout | 0.03175 | 8 | 0.02922 | 11 | **1.087** ✓ | `[CLS]` |
| Attention flow | 0.07127 | 1 | 0.07127 | 1 | **1.000** — | `[CLS]` |
| Vector norm (2020) | 0.04404 | 12 | 0.12063 | 2 | **0.365** ✗ | `it` |

汇总一下（✓ = 方向正确）：

| 方法 | tired 句 | wide 句 | 两句都对？ |
| --- | --- | --- | --- |
| Raw attention | ✓ 3.022 | ✗ 0.452 | 否 |
| Rollout | ✗ 0.939 | ✓ 1.087 | 否 |
| Flow | — 1.000 | — 1.000 | 否（无区分度） |
| Vector norm | ✓ 3.788 | ✗ 0.365 | 否 |

rollout 在 wide 句上的"正确"只有 8.7% 优势，且同一行的 top-1 仍是 `[CLS]`，在 tired 句上又反向（0.939）。**不能据此认为 rollout 真的解出了 Winograd 对。**

一个更直白的看法：`animal` 在四个模型/句子里的份额（animal/(animal+street)）几乎不随形容词变化：

| 模型 | raw（tired / wide） | rollout（tired / wide） | flow（tired / wide） |
| --- | --- | --- | --- |
| BERT | 0.751 / 0.689 | 0.484 / 0.479 | 0.500 / 0.500 |
| RoBERTa | 0.824 / 0.532 | 0.659 / 0.655 | 0.487 / 0.480 |

如果方法真的读出了"太累 → 是动物"和"太宽 → 是街道"，份额应该从 >0.5 翻到 <0.5。RoBERTa 的 raw/rollout 完全没翻（0.82→0.53 是 raw，rollout 反而 0.66→0.66 不动）。

![[B-rollout-winograd.png]]

*图 1：Rollout 从 `it` 到各 token 的归因柱状图，左 tired、右 wide。两图中 `animal` 与 `street` 的高低关系几乎不随形容词翻转。*

![[B-raw-vs-flow.png]]

*图 2：raw / rollout / flow 三种方法在 `animal` 与 `street` 上的并排对比。只有 raw attention 在 tired 句上给出了明显分离。*

### 5.3 关卡 3：flow 为什么退化成均匀分布

**现象**：flow 的归一化行几乎处处等于 $1/n$（$n=15$，均值 0.06667，最大值 0.07116），`animal` 与 `street` 并列该行最大值，ratio 恰好 1.000。

**未归一化 flow 一看就明白了**：

| 模型 | 句子 | flow(animal) | flow(street) | 是否饱和 |
| --- | --- | --- | --- | --- |
| BERT | tired | **1.000000** | **1.000000** | 完全饱和 |
| BERT | wide | **1.000000** | **1.000000** | 完全饱和 |
| RoBERTa | tired | 0.920686 | 0.970822 | 未饱和 |
| RoBERTa | wide | 0.922666 | **1.000000** | street 恰好饱和 |

BERT 上两个候选的最大流都顶到了 1.0，也就是源节点的出容量上限。**这就是饱和**：瓶颈在源侧，不管汇是谁，能送过去的流量都一样多，于是 flow 完全丧失区分度。

**min-cut 分析确认瓶颈位置**（BERT tired 句）：

| 汇 | mincut | $\lvert S\rvert$ | 源出容量 | 汇入容量（第 1 层列和） | binding |
| --- | --- | --- | --- | --- | --- |
| `animal` | 1.000000 | 15/195 | 1.000000 | 1.016909 | **middle** |
| `street` | 1.000000 | — | 1.000000 | 1.015013 | middle |
| `it`（查询自身） | 0.900930 | 194/195 | — | — | SINK side |

源出容量 1.0 ≤ 汇入容量 1.017，所以割被卡在源侧。第 1 层容量列和 min = 0.8402、max = 1.5607（`[CLS]`）、mean = 1.0000——**只有 `[CLS]` 明显高于 1**，其余节点的入容量都 ≥ 源出容量，于是流全部顶满。

**深度扫描：从第 5 层起完全饱和**

| 源层 | 1 | 2 | 3 | 4 | 5 | 6–12 |
| --- | --- | --- | --- | --- | --- | --- |
| flow(animal) | 0.027523 | 0.226178 | 0.625700 | 0.851749 | **1.000000** | 1.000000 |
| flow(street) | 0.039370 | 0.247831 | 0.585363 | 0.856706 | **1.000000** | 1.000000 |
| gold 份额 | 0.4115 | 0.4772 | 0.5167 | 0.4985 | 0.5000 | 0.5000 |

底层还有区分度（第 1 层份额 0.4115），层数一多就被抹平。这解释了为什么"从顶层查询节点求流"必然退化——路径太长，瓶颈一定在源侧。

**排除替代解释一：不是头平均的锅。** 对 12 个头单独构造 $\tilde A$ 求 flow，gold 份额落在 $[0.4793,\ 0.5454]$，均值 0.5083。单头注意力比头平均稀疏得多，但**绝大多数头仍然饱和**。所以"挑出好头"救不了。

**排除替代解释二：不是残差权重的锅。** 四种容量变体对比：

| 容量变体 | 源出容量 | flow min | flow max | flow std | gold 份额 |
| --- | --- | --- | --- | --- | --- |
| $\tilde A=0.5A+0.5I$（论文/官方） | 1.000000 | 0.840171 | 1.000000 | 6.62e-02 | **0.5000** |
| $A+I$（残差容量 1，出容量 2） | 2.000000 | 1.680342 | 2.000000 | 1.32e-01 | 0.5000 |
| $\tilde A$ 行稀疏化 top-3 | 1.000000 | 0.143265 | 1.000000 | 2.81e-01 | **0.7877** |
| $\tilde A$ 行稀疏化 top-1 | 1.000000 | 0.000000 | 1.000000 | 2.49e-01 | nan（两候选均为 0） |

把残差容量从 0.5 改成 1.0 只是整体缩放，份额不变（0.5000）。**真正能打破饱和的是行稀疏化**：每行只保留 top-3 容量后，gold 份额升到 0.7877（方向正确）；top-1 则连通性直接断裂。这说明退化根源是"稠密 + 行和为 1"这个容量结构，而不是某个具体超参。

![[B-flow-graph.png]]

*图 3：BERT tired 句的 flow 图。列为 L0（输入，带词标签）到 L12；红色节点 = 顶层源 `it`，金色节点 = L0 的两个汇 `animal`/`street`；蓝色边 = 容量 $\tilde A$（仅画第 88 百分位以上），红色箭头 = 最大流实际经过的边。*

![[B-attnflow-winograd.png]]

*图 4：flow 在同一对句子上的归因。两图形状几乎相同，两个候选都取到该行最大值 0.0712。*

### 5.4 Kobayashi：H3、H4 都成立

目标位置 $t$ = `[MASK]`（BERT）/ `<mask>`（RoBERTa），12 层 × 12 头平均。

**BERT**，句子 `The animal didn't cross the street because [MASK] was too tired`：

| token | $\bar\alpha$ | $r_\alpha$ | $\overline{\lVert\alpha f\rVert}$ | $r_{norm}$ | **rank gap** | 2021 score | $r_{2021}$ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `[SEP]` | 0.3644 | **1** | 0.1862 | **8** | **−7** | −0.0039 | 9 |
| `[CLS]` | 0.0980 | 2 | 0.1948 | 6 | **−4** | +0.0026 | 1 |
| `was` | 0.0911 | 3 | 0.3684 | 2 | +1 | −0.0828 | 15 |
| `because` | 0.0867 | 4 | 0.3851 | **1** | **+3** | −0.0668 | 14 |
| `tired` | 0.0663 | 5 | 0.3282 | 4 | +1 | −0.0087 | 12 |
| `animal` | 0.0585 | **6** | 0.3539 | **3** | **+3** | −0.0114 | 13 |
| `[MASK]` | 0.0559 | 7 | 0.1183 | 9 | −2 | −0.0052 | 11 |
| `street` | 0.0176 | **11** | 0.0965 | **11** | 0 | −0.0008 | 6 |

**H3 成立**：`[SEP]` 拿走 36% 的注意力却只排第 8 的范数（gap −7），是典型的"高权重低贡献"；`[CLS]` 同类（gap −4）。反向的"低权重高贡献"有 `because`（4→1）、`animal`（6→3）、`the`（8→5）、`too`（9→7）。RoBERTa 上 `<s>` gap = −5、`</s>` gap = −3，模式一致。**反转确实集中在特殊 token 上**，与 Kobayashi 2020 的论点吻合。

**H4 成立**：Spearman 相关

| 相关对 | BERT | RoBERTa |
| --- | --- | --- |
| $\rho(\alpha,\ \lVert\alpha f\rVert)$（2020） | +0.8143 | +0.8901 |
| $\rho(\alpha,\ \text{2021 residual+LN score})$ | **−0.4286** | **−0.8681** |
| $\rho(\text{2020 norm},\ \text{2021 score})$ | **−0.5821** | **−0.9033** |

计入残差与 LayerNorm 后，相关性从 +0.81 直接翻到 −0.43（RoBERTa 更极端：+0.89 → −0.87）。**"注意力权重高 = 贡献大"在 2021 度量下反向成立。**

**但有个重要的细微差别**：2020 范数的**整体排序**仍与权重高度一致（+0.81/+0.89），反转只发生在少数 token 上。对候选对而言：

| 度量 | 模型 | animal | street | animal 份额 |
| --- | --- | --- | --- | --- |
| attention weight | BERT | 0.05853 | 0.01764 | 0.768 |
| $\lVert\alpha f\rVert$ (2020) | BERT | 0.35390 | 0.09653 | **0.786** |
| residual+LN score (2021) | BERT | −0.01140 | −0.00077 | 0.937 |
| attention weight | RoBERTa | 0.05708 | 0.01122 | 0.836 |
| $\lVert\alpha f\rVert$ (2020) | RoBERTa | 0.32469 | 0.06051 | 0.843 |
| residual+LN score (2021) | RoBERTa | −0.00349 | +0.00048 | 1.161 |

权重份额 0.768 与范数份额 0.786 几乎相同——**在这个具体探针上 2020 范数并没有推翻注意力权重的结论**，只是把 `[SEP]` 之类剔掉了。而 2021 score 给出的排序完全不同（BERT 上 `animal` 排 13、`street` 排 6），且两者都是负值（意味着加进去反而压缩了目标表示）。RoBERTa 的 street score 甚至是正的而 animal 是负的。

一个值得注意的反例：RoBERTa 的 `Ġstreet` 权重排名 14（倒数第一）、范数排名也 14，但 **2021 score = +0.0005，排名第 1**。三个度量给出三种截然不同的答案。

![[B-kobayashi-norm-heatmap.png]]

*图 5：BERT 上三种度量的 层 × token 热图 —— (a) $\alpha_{t,s}$、(b) $\lVert\alpha_{t,s}f(x_s)\rVert$、(c) 2021 residual+LN score；红框标出两个候选。(a)(b) 图案相近，(c) 几乎完全反转。*

![[B-kobayashi-weight-vs-norm.png]]

*图 6：权重与范数的分歧 —— (a) 排名 vs 排名散点（偏离对角线即反转，颜色为 |gap|，Spearman = 0.814）；(b) 数值 vs 数值散点；(c) 每 token 权重与范数并排条形，`[SEP]` 的"权重最高、范数中游"一目了然。*

### 5.5 关卡 2 结果：所有实现校验通过

| 检查项 | BERT | RoBERTa |
| --- | --- | --- |
| 分解式重构 attention 输出的最大绝对误差 | 2.03e-06（tired）/ 2.38e-06（wide） | — |
| networkx vs scipy 最大流最大绝对差 | 1.39e-07 / 2.10e-07 | — |
| 手工重算 hidden states 最大 L2（绝对 / 相对） | 1.45e-05 / 8.27e-07 | 1.57e-05 / 6.38e-07 |
| 手工重算注意力权重最大绝对误差 | 1.88e-06 | 2.24e-06 |
| 逐头 $W_OW_V$ 分解最大绝对误差 | 9.54e-07 | 2.86e-06 |
| flow 图邻接反解误差 / 残差等价写法误差 | 0.0 / 0.0 | 0.0 / 0.0 |

全部在 float32 噪声量级。`test_methods.py` 的 6 项自检（含 $L=1$ 闭式解）全部通过。**"方法失败"的结论不是实现 bug 造成的。**

## 六、结论

| 命题                         | 判定      | 依据                                                                             |
| -------------------------- | ------- | ------------------------------------------------------------------------------ |
| H1 归因方向随形容词翻转              | **不成立** | 四种方法没有一个两句都对；份额几乎不随形容词变化                                                       |
| H2 rollout/flow 比 raw 更可靠  | **不成立** | flow 在 BERT 上完全饱和（ratio ≡ 1.000）；rollout 只在 wide 句微弱正确（1.087）且 top-1 是 `[CLS]` |
| H3 存在权重/范数排名反转，集中在特殊 token | **成立**  | `[SEP]` gap −7、`[CLS]` −4、`<s>` −5、`</s>` −3                                   |
| H4 计入残差 + LN 后相关性转负        | **成立**  | $\rho(\alpha, \text{2021})$ = −0.4286 / −0.8681                                |

**一句话**：Kobayashi 的批评是对的（权重不等于贡献，尤其计入残差与 LN 后可能反向），但 Abnar 的两种聚合方案在这个探针上并没有把问题解决——rollout 会被 `[CLS]` 主导，flow 会因源侧容量饱和退化成均匀分布。而模型本身其实是会做这道题的（cloze ratio 60/28/469/4211）。

### 与原文的差异（需要说清的三点）

1. **原文没有报告 flow 退化**。Abnar & Zuidema 在合成任务上展示 flow 优于 raw attention，但没有讨论"在真实 BERT 的长句上、从顶层节点求流时，flow 会因源侧容量饱和而退化成均匀分布"。本次给出了机理（min-cut 卡在源出容量、第 5 层起饱和）和打破方式（行稀疏化 top-3 → 份额 0.7877）。
2. **RoBERTa 的 flow 未饱和，但方向也不对**。此前一版报告写"RoBERTa 上 flow 方向与 gold 一致"，核对数据后更正：**只有 wide 句一致**（street 1.000000 > animal 0.922666 ✓），tired 句仍偏向 street（0.970822 > 0.920686，份额 0.487 ✗）。"未退化"成立，"方向正确"只对一半句子成立。
3. **2020 范数在本探针上没有推翻权重**。$\rho(\alpha,\lVert\alpha f\rVert)$ 高达 +0.81/+0.89，候选份额 0.768 vs 0.786。反转集中在特殊 token，而非全部内容词。这与原文"范数与权重经常分歧"的整体印象需要加上限定条件。

### 未验证的推测

BERT 与 RoBERTa 的 flow 表现不同，可能原因：(i) `[CLS]`/`<s>` 汇聚注意力的强度不同（BERT 的 `[CLS]` 第 1 层容量列和 1.5607，远高于均值 1.0000）；(ii) 训练目标差异（NSP vs 无 NSP）使残差流范数分布不同；(iii) 分词长度与 BPE 前缀造成的图规模差异。**这些都只是候选解释，本次没有做控制变量验证。**

### 对主线研究的意义

[[2026-09-07-实验状态全面同步|此前 LLaVA 实验]]用的是单层 raw attention 的区域均值。本次结果给出两条警示：

- raw attention 会被特殊 token（`[CLS]`/`[SEP]`）吸走大量权重，直接对区域求均值可能低估真实贡献；
- 但换成 rollout/flow 也不一定更好——rollout 仍被 `[CLS]` 主导，flow 可能直接退化。

结合 [[Chefer-Transformer可解释性-相关性传播|Chefer 文档]]的结论（梯度修正的相关性传播能按 query 语义定位），**带梯度信号的方法比纯注意力聚合更值得优先尝试**。

## 七、复现中踩的坑

| 问题 | 原因 | 解决 |
| --- | --- | --- |
| HF 报警 `BertSdpaSelfAttention ... does not support output_attentions=True ... Falling back to the manual attention implementation` | transformers 4.57.1 的 SDPA 路径不返回注意力矩阵 | 写 `diag_impl.py` 对比 `default/eager/sdpa` 三种实现与手工重算的偏差，确认回退不改变数值（hidden L2 1.45e-05） |
| flow 结果"看起来像均匀分布" | 源侧容量饱和（非 bug） | 排查路径：① `test_methods.py` 验证 $L=1$ 闭式解（通过）② networkx vs scipy 互验（1.39e-07）③ `diag_flow.py` 输出**未归一化** flow → 立刻看到全顶在 1.000000 → ④ min-cut + 深度扫描定位机理 |
| 残差容量两种写法混淆 | 官方 `get_adjmat(add_residual=True, residual_weight=1.0)` 得 $A+I$（行和 2），论文写法是 $0.5A+0.5I$（行和 1） | 归一化后 flow 相同（`equiv_err=0.0`），未归一化差一倍。统一采用论文写法，$A+I$ 仅作对照（BERT 上极差仅 1.14e-02，同样退化） |
| `nx.from_numpy_matrix` 报错 | networkx ≥ 3.0 已移除该 API | 改为 `nx.from_numpy_array`；同时 `make_graph` 需显式 `remove_edges_from(nx.selfloop_edges(G))`，否则对角线上的残差容量变成自环干扰最大流 |
| scipy 最大流报错 | `scipy.sparse.csgraph.maximum_flow` 要求 CSR 图**结构对称**，且容量须为整数 | 每条容量边 $(u,v)$ 补一条显式零容量反向边 $(v,u)$；容量乘 $10^6$ 取整 |
| RoBERTa 第 1 层就对不上 | 位置编码需从 2 开始（`padding_idx=1`） | 修正起始索引 |
| `Some weights of RobertaModel were not initialized: ['pooler.dense.bias','pooler.dense.weight']` | 官方 checkpoint 不含 pooler | 实验只用 encoder 与 MLM head，无影响 |
| RoBERTa 候选词定位失败 | token 带 `Ġ` 前缀，字符串相等匹配不到 | `task1_flow_rollout.py` 改用 `offset_mapping` 的字符重叠定位 |
| CPU 上每句 46–48 s | $n\times n$ 全量 flow 需 $15\times12=180$ 次 Edmonds–Karp，占 32–34 s | 加 `--quick` 开关跳过全量矩阵；对比两份 json 确认不影响末层对比表 |
| 头平均可能掩盖单头行为 | 12 头平均是很强的简化 | 保留逐头 share；`diag_flow.py` 的逐头 flow 显示单头份额仍在 0.5 附近（0.4793–0.5454），退化不能靠挑头解决 |

## 八、产物

| 类型 | 路径 |
| --- | --- |
| 详细报告 | `D:\code\attn-repro\B\REPORT.md` |
| 方法实现 | `D:\code\attn-repro\B\{attn_methods,attention_graph_util}.py` |
| 实验入口 | `D:\code\attn-repro\B\{run_winograd,task1_flow_rollout,task2_kobayashi}.py` |
| 自检与诊断 | `D:\code\attn-repro\B\{test_methods,diag_impl,diag_task2,diag_flow}.py` |
| 运行日志 | `D:\code\attn-repro\B\{log_winograd,log_task1,log_task2,log_diagflow}.txt`（`diag_impl.py` 未留存日志） |
| 数值 JSON | `results_task1.json`（2446 行）、`results_task2.json`（2144 行）、`D:\checkpoints\outputs\B\{results-bert,results-bert-quickprobe,diag-flow}.json` |
| 交付图 | `assets/attn-repro/B-*.png`（6 张） |
