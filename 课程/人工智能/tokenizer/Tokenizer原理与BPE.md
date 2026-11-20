Tokenizer 的职责是把人类可读的文本转换为模型可以处理的离散整数序列；生成模型完成预测后，再把整数序列转换回文本。它是文本与神经网络之间的接口，也是影响上下文长度、训练数据表示方式和推理成本的基础组件。

## 1. Tokenizer 的完整流程

一个典型的文本 tokenizer 可以概括为四步：

1. **规范化（normalization）**
   - 处理 Unicode 的某些等价表示，例如是否进行大小写转换、Unicode 规范化或空白字符整理。
   - 具体模型不一定执行全部规范化。过度规范化可能丢失大小写、空格或格式信息，因此现代大语言模型通常会保留更多原始文本。

2. **预分词（pre-tokenization）**
   - 按规则识别单词、数字、标点、空格和换行等片段。
   - 这一步通常不是简单地按空格 `split`。英文中的标点、数字串和前导空格可能被单独处理，中文也可能进入不同的切分路径。
   - 预分词决定了 BPE 的初始边界，但最终 token 仍由后续合并规则决定。

3. **子词切分与编码**
   - 将片段转换为初始符号序列，再按照训练好的 BPE `merges` 规则逐步合并。
   - 每个最终符号在 `vocab` 中查到一个整数 ID，得到模型的输入序列。

4. **特殊 token 处理**
   - 在序列中加入模型需要的控制符号，例如开始、结束、填充、未知词或消息角色标记。
   - 特殊 token 的 ID 也来自词表，但它们通常不是普通文本通过 BPE 合并产生的。

因此，编码过程可以写成：

$$
\text{文本} \xrightarrow{\text{规范化/预分词}} \text{片段}
\xrightarrow{\text{BPE}} \text{token}
\xrightarrow{\text{vocab 查表}} \text{token ID}
$$

解码时执行相反方向的操作：token ID 查回 token，拼接并还原为字节或文本，再处理特殊 token。

## 2. Unicode、code point 与 bytes

### 2.1 字符、code point 和编码不是一回事

- **字符**是人类感知的文本单位，例如字母、汉字或表情符号。
- **Unicode code point** 是 Unicode 为字符分配的编号，写作 `U+` 加十六进制数。例如：
  - `A` 是 `U+0041`；
  - `中` 是 `U+4E2D`；
  - `😀` 是 `U+1F600`。
- **编码（encoding）**规定如何把 code point 表示为字节。UTF-8、UTF-16 都是编码方式。
- **byte（字节）**是存储和传输的基本单位，通常包含 8 个 bit。

code point 不是字符在内存中的固定字节，也不等于 token。一个字符可能由一个或多个 code point 组成，一个 code point 在 UTF-8 中也可能占用 1 到 4 个字节。

### 2.2 UTF-8 的基本规律

UTF-8 用变长字节表示 Unicode code point：

| code point 范围 | UTF-8 字节数 |
| --- | ---: |
| `U+0000` 到 `U+007F` | 1 |
| `U+0080` 到 `U+07FF` | 2 |
| `U+0800` 到 `U+FFFF` | 3 |
| `U+10000` 到 `U+10FFFF` | 4 |

例如：

```text
A   -> code point U+0041 -> UTF-8: 0x41
中  -> code point U+4E2D -> UTF-8: 0xE4 0xB8 0xAD
😀  -> code point U+1F600 -> UTF-8: 0xF0 0x9F 0x98 0x80
```

这解释了为什么中文或表情符号不一定对应一个 token：tokenizer 可能在字符层、子词层或字节层对它们进行切分。

## 3. BPE：用更大的词表换取更短的序列

### 3.1 核心动机

如果只使用字符作为 token，词表较小，但一句话会被切成很长的序列；如果把每个完整单词都作为 token，序列较短，却会遇到词表巨大、未登录词多、形态变化难以覆盖等问题。

BPE（Byte Pair Encoding）在两者之间取折中：从较小的基本符号开始，反复合并训练语料中高频的相邻符号，使常见片段成为一个 token。其核心目标是：

> 通过增大 vocabulary，减少 sequence length，同时保留对未见过文本的组合能力。

词表变大通常意味着平均每个 token 承载更多文本，序列长度下降；但词表过大也会增加 embedding 参数、softmax 输出规模和存储成本，因此需要在词表规模与序列效率之间折中。

### 3.2 BPE 的训练过程

以字符或字节作为初始符号，训练过程大致如下：

1. 统计语料中所有相邻符号对的频率；
2. 找到频率最高的相邻对；
3. 将这一对合并为新符号，并加入词表；
4. 更新语料中的相邻关系；
5. 重复执行，直到达到目标词表大小或合并次数。

假设初始序列为：

```text
l o w e r
```

若 `l` 和 `o` 高频共现，先合并为 `lo`；之后可能合并为 `low`，最终得到：

```text
low er
```

这里的 `er` 也可以是一个 token。BPE 不要求 token 必须是完整单词，常见词根、后缀、标点和空格组合都可能成为 token。

### 3.3 BPE 的编码过程

实际编码时不会重新统计当前句子的频率。tokenizer 使用训练阶段固定下来的 `merges` 优先级：

1. 将输入片段拆为初始符号；
2. 找出当前序列中出现在 `merges` 中的相邻对；
3. 按优先级最高的规则合并；
4. 继续合并，直到没有可用规则；
5. 用 `vocab` 将最终 token 映射为整数 ID。

因此，`merges` 决定“哪些相邻符号可以合并以及先合并谁”，`vocab` 决定“最终符号对应哪个整数 ID”。二者缺一不可。

## 4. Byte-level BPE

普通 BPE 可以从字符或预分词后的符号开始，但字符集合可能很大，还会遇到未登录字符。Byte-level BPE 先把文本编码为 UTF-8 字节，再以有限的字节集合作为基础符号，之后执行 BPE 合并。

它的优点是：

- 基础字节集合足以覆盖任意 UTF-8 文本，基本不需要真正的未知词；
- 拼写错误、新词、代码、混合语言和罕见字符都能被拆成字节序列；
- 词表规模可以保持在可控范围。

代价是：一个 Unicode 字符可能被拆成多个字节 token，尤其是中文、表情符号和罕见字符可能比英文消耗更多 token。训练阶段常会把高频字节组合重新合并，所以常见文本仍然可以获得较短表示。

### 4.1 GPT-2 的空格表示

GPT-2 使用 byte-level BPE。为了区分单词内部的片段与单词前的空格，词表中的可见表示经常使用类似 `Ġ` 的标记表示“前面带空格”。例如，某些工具中可能看到：

```text
Hello -> "Hello"
 world -> "Ġworld"
```

`Ġ` 不是原文中的 Unicode 空格，而是 tokenizer 为显示或内部映射引入的标记。真正解码时它会还原为空格。

### 4.2 GPT-4 及新 tokenizer 的空格与缩进

GPT-4 系列常见的 `tiktoken` tokenizer 使用基于字节的 BPE，但具体词表、预分词正则和特殊 token 取决于模型版本，不能把 GPT-2 的 `Ġ` 形式直接当作所有模型的显示规则。

空格通常会与后面的英文片段共同形成 token，也可能单独成为 token；连续空格、换行和制表符则可能根据训练频率被合并成一个或多个 token。缩进同理：

```text
    def f():
```

开头的四个空格可能对应一个表示四空格缩进的 token，也可能拆成多个空格 token，实际结果由具体 tokenizer 的 `vocab` 和 `merges` 决定。代码中的缩进会消耗上下文长度，并且缩进差异会改变 token 序列，不能假设“一个空格永远就是一个 token”。

最可靠的做法是使用目标模型对应的 tokenizer 实际调用 `encode` 检查 token ID、token 字节和解码结果；不同模型之间不能直接复用 token ID。

### 4.3 Byte fallback

Byte fallback 是 Byte-level BPE 的保底机制：为每个可能的字节保留基础 token，常见形式是 `<0x00>` 到 `<0xFF>`。当一段 Unicode 文本无法由普通子词词表覆盖时，tokenizer 先将其编码为 UTF-8 字节，再将每个字节映射到 byte token。

例如“猫”的 UTF-8 字节为 `E7 8C AB`，若词表没有“猫”或包含它的子词，可能退化为：

```text
猫 -> <0xE7> <0x8C> <0xAB>
```

这与按 Unicode 字符逐个回退不同。一个汉字通常对应多个 UTF-8 字节，一个 emoji 可能对应四个或更多字节。

#### 编码和还原

```python
def encode_piece(piece):
    if piece in vocab:
        return [vocab[piece]]

    raw = piece.encode("utf-8")
    return [byte_to_id[b] for b in raw]

def decode_ids(ids):
    buffer = bytearray()
    output = []
    for token_id in ids:
        if token_id in byte_id_to_value:
            buffer.append(byte_id_to_value[token_id])
        else:
            output.append(buffer.decode("utf-8", errors="strict"))
            buffer.clear()
            output.append(vocab_decode(token_id))
    output.append(buffer.decode("utf-8", errors="strict"))
    return "".join(output)
```

不能在每个 byte token 后立即单独解码 UTF-8，因为多字节字符可能尚未收集完整。实际实现还需要处理普通 token 与 byte token 交错、非法字节序列以及流式输出。

#### 训练与推理

Byte fallback 不是只在训练或只在推理中存在的后处理技巧，而是贯穿词表设计、训练数据编码和生成解码的端到端约定：

- **词表构建时**：预留 byte token，使理论上所有字节都有表示。
- **训练编码时**：遇到词表未覆盖的片段，将其编码为 byte token，模型因此能学习这些 token 的上下文。
- **生成推理时**：模型可能生成 byte token，解码器负责将其还原成原始字节和文本。

某个模型虽然词表有 byte token，但若训练数据从未使用它们，生成质量仍可能很差。

#### 边界与代价

Byte fallback 提供的是“可表示性”，不等于“可理解性”：

- 字节 token 数量可能明显增加，导致序列更长、推理成本更高。
- 模型需要从多个 byte token 组合出字符、词和语义，学习难度高于直接使用常见子词。
- 词表、编码方式和解码器必须一致；UTF-8 字节不能随意按其他字符集解释。
- 任意二进制数据可以被表示，但模型不一定理解压缩文件、加密数据或随机字节。
- 字节序列若被截断或污染，可能无法组成合法 Unicode；解码器需要明确的错误策略。

因此，Byte fallback 是覆盖长尾输入的保底层，常见词和常见形态仍应优先由高效的子词 token 表示。

## 5. `vocab`、`merges` 与 token ID

### 5.1 `vocab`

`vocab` 是词表，建立 token 与整数 ID 的双向映射：

```text
token  -> token ID
token ID -> token
```

模型的输入不是字符串，而是整数 ID 序列，例如：

```text
[15496, 995, 0]
```

整数本身没有语言含义，只是词表中的索引。不同 tokenizer 即使处理同一句话，也可能产生不同的 token 数量和 ID。

### 5.2 `merges`

`merges` 是 BPE 的合并规则及其优先级。它不直接等同于词表，也不负责把 token 映射成 ID。一个合理的 tokenizer 需要保证编码得到的最终符号能够在 `vocab` 中查到。

可以把二者理解为：

- `merges`：构造 token 的规则；
- `vocab`：保存 token 并给出编号；
- 编码器：按照规则构造 token，再查表得到 ID；
- 解码器：按照 ID 查表，再恢复字节和文本。

## 6. Tokenizer 与 embedding model 的区别

Tokenizer 是离散的符号处理组件，embedding model 是把离散符号映射为连续向量的神经网络组件。二者的输入输出完全不同：

| 组件 | 输入 | 输出 | 是否通过训练学习 |
| --- | --- | --- | --- |
| Tokenizer | 文本字符串 | token ID 序列 | 规则和词表在 tokenizer 训练后固定 |
| Embedding 层 | token ID | 向量序列 | 向量参数随模型训练更新 |
| Embedding model | 文本或 token ID | 一个或多个语义向量 | 通过预训练或对比学习优化 |

在大语言模型中，常见路径是：

$$
\text{文本}
\xrightarrow{\text{Tokenizer}}
\text{token IDs}
\xrightarrow{\text{Embedding lookup}}
\text{向量}
\xrightarrow{\text{Transformer}}
\text{上下文表示与预测}
$$

Embedding 层通常可以看作一个矩阵 $E\in\mathbb{R}^{V\times d}$ 的查表操作，其中 $V$ 是词表大小，$d$ 是向量维度。对于 token ID $i$，输入向量是矩阵的第 $i$ 行：

$$
\mathbf{x}_i=E[i]
$$

Tokenizer 不理解 token 的语义，也不产生向量；它只负责稳定地建立文本与离散 ID 之间的映射。Embedding 也不负责决定如何切分文本，它只接收 tokenizer 已经生成的 ID。

需要区分“Embedding 层”和“Embedding model”：

- **Embedding 层**是语言模型内部的参数矩阵，服务于后续的 [[../注意力机制/自注意力机制 (Self-Attention)|注意力机制]] 和 Transformer 计算；
- **Embedding model**通常指专门把文本编码为句向量、段落向量或文档向量的模型，常用于检索、聚类和语义相似度；
- 二者都依赖 tokenizer，但输出目标不同。语言模型的 embedding 主要服务于逐 token 预测，专用 embedding model 更强调整体语义表示。

## 7. 与语言模型训练和推理的关系

Tokenizer 决定输入文本如何占用上下文窗口。相同字符数不代表相同 token 数量：英文常见词可能由一个或少数 token 表示，中文、代码、数学表达式和罕见 Unicode 字符可能需要更多 token。

在[[../微调/预训练/1_预训练的目标、损失与训练范式|预训练]]阶段，训练语料先被 tokenizer 转成 token ID，再输入 Transformer。模型学习的是 token 序列上的条件概率：

$$
P(x_t\mid x_1,x_2,\ldots,x_{t-1})
$$

这里的 $x_t$ 是 token ID，而不是直接的汉字、单词或字节。模型输出下一个 token 的概率分布后，解码器再把 token ID 序列还原成文本。

## 8. 常见误区

- **“一个汉字就是一个 token”**：不成立。汉字可能是一个 token，也可能拆成多个字节相关 token。
- **“一个 token 就是一个单词”**：不成立。token 可以是字符、词根、后缀、标点、空格组合或字节片段。
- **“BPE 编码时会重新找最高频相邻对”**：通常不成立。编码阶段使用训练好的 `merges` 优先级，而不是重新统计当前句子。
- **“token ID 越大语义越强”**：不成立。ID 只是索引，没有可比较的大小关系。
- **“Tokenizer 和 embedding 是同一个模型”**：不成立。前者完成离散化，后者完成向量化。
- **“所有 GPT 模型的 token 规则相同”**：不成立。GPT-2、GPT-4 及不同版本的 tokenizer 可能使用不同词表、合并规则和特殊 token。

判断一个 tokenizer 的真实行为，应同时查看模型名称、词表、合并规则和对应库的编码结果，而不能只根据界面显示的 token 文本推断内部字节表示。
