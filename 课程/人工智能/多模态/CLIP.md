```
# image_encoder - ResNet or Vision Transformer 
# text_encoder - CBOW or Text Transformer 
# I[n, h, w, c] - minibatch of aligned images 
# T[n, l] - minibatch of aligned texts 
# W_i[d_i, d_e] - learned proj of image to embed 
# W_t[d_t, d_e] - learned proj of text to embed 
# t - learned temperature parameter  

# extract feature representations of each modality 
I_f = image_encoder(I) #[n, d_i] 
T_f = text_encoder(T) #[n, d_t]  

# joint multimodal embedding [n, d_e] 
I_e = l2_normalize(np.dot(I_f, W_i), axis=1) 
T_e = l2_normalize(np.dot(T_f, W_t), axis=1)  

# scaled pairwise cosine similarities [n, n] 
logits = np.dot(I_e, T_e.T) * np.exp(t)  

# symmetric loss function 
labels = np.arange(n) 
loss_i = cross_entropy_loss(logits, labels, axis=0) 
loss_t = cross_entropy_loss(logits, labels, axis=1) 
loss = (loss_i + loss_t)/2

```


这段是 CLIP 论文 Figure 3 的伪代码，也就是**对比学习的核心训练逻辑**，全部内容可以概括成一句话：在一个 batch 内做 $n$ 路分类——对每张图，从 $n$ 段文本中挑出正确的那一段（以及反方向同样做一次）。

## 符号与形状

|符号|形状|含义|
|---|---|---|
|`I[n, h, w, c]`|$n\times h\times w\times c$|batch 内 $n$ 张图像，形状与编码器有关（`c` 为通道数）|
|`T[n, l]`|$n\times l$|batch 内 $n$ 段文本，`l` 为 token 长度（CBOW 时可为 1）|
|`W_i[d_i, d_e]`|$d_i\times d_e$|图像特征维度 $d_i$ → 共享嵌入维度 $d_e$ 的可学习投影矩阵|
|`W_t[d_t, d_e]`|$d_t\times d_e$|文本特征维度 $d_t$ → $d_e$ 的可学习投影矩阵|
|`t`|标量|可学习的温度参数（注意写在 log 空间）|

“aligned（对齐的）”是关键前提：**第 $i$ 张图与第 $i$ 段文本本来就是一对**，正样本天然落在矩阵对角线上。

## 逐块拆解

### 1. 各自编码，得到单模态特征

```
I_f = image_encoder(I)   # [n, d_i]
T_f = text_encoder(T)    # [n, d_t]
```

两条独立支路，互不通信，只在最后的相似度矩阵处相遇（这就是“双塔”的含义）。

### 2. 投影到共享空间并归一化

```
I_e = l2_normalize(np.dot(I_f, W_i), axis=1)   # [n, d_e]
T_e = l2_normalize(np.dot(T_f, W_t), axis=1)   # [n, d_e]
```

- `np.dot(I_f, W_i)`：$[n,d_i]\times[d_i,d_e]\to[n,d_e]$，为每个模态学一个线性映射，把两种模态的表示拉到同一个 $d_e$ 维空间。
- `axis=1`：**按行**归一化，即每个样本得到一个单位向量。这样后续点积就等于余弦相似度，取值范围 $[-1,1]$。归一化还顺带消除了特征模长的影响，避免模型靠“放大向量长度”来降低损失。

### 3. 相似度矩阵 + 温度缩放

```
logits = np.dot(I_e, T_e.T) * np.exp(t)   # [n, n]
```

- `np.dot(I_e, T_e.T)`：$[n,d_e]\times[d_e,n]\to[n,n]$，$s_{ij}=\cos(I_i,T_j)$。
- 矩阵含义：第 $i$ 行是“图像 $i$ 与所有文本的相似度”，第 $i$ 列是“文本 $i$ 与所有图像的相似度”，**对角线是正样本，其余 $n^2-n$ 个全是负样本**。
- `* np.exp(t)`：温度缩放。因为 `t` 在 log 空间学习，`exp(t)` 恒为正，等价于用温度 $\tau=1/\exp(t)$ 做缩放：

$$ \text{logits}_{ij}=\frac{\cos(I_i,T_j)}{\tau},\qquad \tau=\frac{1}{e^{t}} $$

温度越小（`exp(t)` 越大），softmax 越尖锐，模型对“搞错”的惩罚越强。CLIP 论文中这一项初始化为 $\tau=0.07$（即缩放约 14.3），并限制缩放不超过 100，防止训练早期 logits 过大导致梯度爆炸。用 `exp(t)` 而不是直接学缩放系数，是为了保证缩放始终为正且优化更稳定。

### 4. 对称损失：两个方向各做一次 $n$ 路分类

```
labels = np.arange(n)                                  # [0, 1, ..., n-1]
loss_i = cross_entropy_loss(logits, labels, axis=0)
loss_t = cross_entropy_loss(logits, labels, axis=1)
loss = (loss_i + loss_t) / 2
```

- `labels = np.arange(n)`：因为正样本在对角线，第 $i$ 行/列的正确答案就是类别 $i$，所以标签就是“自己的下标”。
- `axis` 决定**沿哪一维做 softmax**：
    - `axis=1`：对每一行归一化（在 $n$ 段文本上竞争）→ 图像找文本；
    - `axis=0`：对每一列归一化（在 $n$ 张图像上竞争）→ 文本找图像。
- 合起来就是双向对称的 InfoNCE：

$$ \mathcal{L}=\frac{1}{2}\left(\underbrace{\frac{1}{n}\sum_{i=1}^{n}-\log\frac{e^{s_{ii}}}{\sum_{j}e^{s_{ij}}}}_{\text{图}\to\text{文}}+\underbrace{\frac{1}{n}\sum_{j=1}^{n}-\log\frac{e^{s_{jj}}}{\sum_{i}e^{s_{ij}}}}_{\text{文}\to\text{图}}\right) $$

- 除以 2 只是让两个方向取平均，使梯度尺度与 batch 大小、方向数解耦（代码里两个变量名的下标写法略有歧义，本质就是“两个方向各一次”）。

## 两个容易忽略的工程含义

1. **batch size 就是负样本数量**：每个样本的“错选项”只有同一个 batch 里的其他样本，$n=32$ 的负样本质量远不如 $n=32768$。CLIP 用超大 batch（并配梯度累积/跨卡通信）正是为了让 $n^2$ 个负样本足够多样，这也解释了后来 SigLIP 改用 sigmoid 损失的原因——它不再需要全局 softmax，从而绕开跨设备同步的瓶颈。
2. **归一化 + 点积 + 温度**三步合起来，等价于“把图文对齐问题化成一个温度可调的 softmax 分类”，代码短但所有设计选择（L2 归一化、log 空间温度、对称损失、对角线标签）都在为训练稳定性服务。