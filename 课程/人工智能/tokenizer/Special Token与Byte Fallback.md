特殊 token 是词表中具有预先约定语义的符号。它们不是普通文本片段，通常不会由语料中的字符频率或 [[Tokenizer原理与BPE#BPE：用更大的词表换取更短的序列|BPE]] merge 规则自然产生。普通 token 负责表达用户输入的内容，特殊 token 负责标记输入边界、控制模型行为或承载无法按普通文本解释的状态。

## 1. 特殊 token 的类别与作用

### 1.1 结构边界 token

这类 token 用来表达序列或对话的结构：

- `BOS` / `<s>`：序列开始。
- `EOS` / `</s>`：序列结束。
- `PAD`：批处理时补齐长度，通常配合 attention mask 使用。
- `SEP`：分隔句子、段落或不同字段。
- `CLS`：在序列开头提供分类任务的聚合位置。

边界 token 不等于原文中的标点。例如输入文本没有 `</s>`，但编码器仍可能在末尾插入它。`PAD` 更不是需要预测的自然语言内容，训练时一般通过 mask 排除其损失。

### 1.2 控制与角色 token

对话模型常将角色、工具调用和消息边界编码为特殊 token，例如：

```text
<|user|> 请查询天气 <|assistant|>
```

这些 token 让模型区分“谁在说话”和“内容是什么”。工具调用、图像占位符、思考区段或文档分隔符，也可以采用相同机制。它们的专门目的，是把业务协议显式地放进模型可处理的离散序列中，而不是表示某个常见词。

### 1.3 未知与保底 token

`UNK` 或 `<unk>` 表示 tokenizer 无法将输入映射到可用词表时的兜底结果。它属于特殊 token，但不应与 byte fallback 混淆：`UNK` 丢弃了具体字符信息，byte fallback 则尝试保留输入的字节信息。

### 1.4 特殊 token 与普通 token 的边界

特殊 token 可能由原始文本显式写入，也可能由 tokenizer 的模板自动插入。它不要求是原文或 BPE merge 的结果，关键在于词表和 tokenizer 配置是否为它分配了稳定的 token ID，以及编码器是否在特定上下文中识别它。

如果普通文本中出现了看起来像 `<|user|>` 的字符串，是否将其识别为控制 token 取决于配置：有的 tokenizer 只在模板阶段生成，有的 tokenizer 会扫描并识别特殊字符串。若误把用户可控文本解析成控制 token，可能造成提示注入或消息边界混乱，因此特殊 token 的匹配规则必须明确。

## 2. 训练与推理流程

### 2.1 词表训练阶段

训练普通 token 时，tokenizer 从语料统计字符、字节或子词片段，再通过 [[BPE|BPE 合并]]、Unigram 等算法建立词表。特殊 token 通常走独立流程：

1. 预先声明 token 字符串及其属性，例如是否允许出现在输入中。
2. 为每个特殊 token 分配固定 ID，或在训练词表后追加保留 ID。
3. 训练普通词表时禁止其被拆分、合并或删除。
4. 保存 special token 映射、模板和编码选项。

因此，`<s>` 不需要在语料中足够频繁，`<|assistant|>` 也不应被 BPE 当成普通字符片段学习。真正参与模型训练的是最终 token ID 序列；特殊 token 的语义来自训练数据中的位置和模型对该 ID 的学习。

### 2.2 编码阶段

典型编码顺序如下：

```text
输入文本
  -> 识别允许的 special token
  -> 普通文本预分词
  -> 子词切分或 byte fallback
  -> 添加 BOS/EOS 等模板 token
  -> 输出 token ID
```

伪代码：

```python
def encode(text, add_special_tokens=True):
    pieces = split_special_tokens(text, allowed_special)
    ids = []
    for piece in pieces:
        if piece.is_special:
            ids.append(special_to_id[piece.value])
        else:
            ids.extend(subword_encode(piece.text, byte_fallback=True))

    if add_special_tokens:
        ids = [BOS_ID] + ids + [EOS_ID]
    return ids
```

模板插入必须和模型训练格式一致。若训练时使用 `<|user|>`、`<|assistant|>`，推理时只添加普通的 `BOS/EOS`，模型可能无法正确判断角色和生成位置。

### 2.3 解码阶段

解码时，普通 token 先恢复为文本片段，byte token 还要按字节拼接并使用指定编码解码；特殊 token 则根据选项决定：

- 保留为 `<|assistant|>` 等可见标记；
- 转换为结构化消息边界；
- 跳过 `BOS`、`EOS`、`PAD` 等控制符。

`skip_special_tokens=True` 只影响展示结果，不会改变模型实际接收过的 token 序列。

## 3. Byte fallback

Byte fallback 属于 Byte-level BPE 的长尾输入保底机制，完整的原理、编码/解码过程、训练与推理边界以及实现代价见：

[[Tokenizer原理与BPE#4.3 Byte fallback|Tokenizer 中的 Byte fallback]]

## 4. UNK 的缺陷与取舍

使用 `UNK` 时，多个不同输入会映射到同一个 ID：

```text
火星探测器型号 X-9000 -> ... <unk>
另一串完全不同的罕见文本 -> ... <unk>
```

对模型来说，两者在该位置的观测完全相同，具体拼写、数字、语言和实体信息已经丢失。这会造成：

1. 命名实体、网址、代码和拼写错误难以恢复。
2. 输入不可逆，解码结果不能保证与原文一致。
3. 训练信号混杂，不同未知词共享同一个 embedding。
4. 对跨语言、emoji、少数文字和新术语的鲁棒性下降。

`UNK` 仍有适用场景：词表极小、实现需要固定兜底、输入只允许有限字符集，或系统必须拒绝非法输入。对于开放域大语言模型，byte-level 或 byte fallback 通常比单一 `UNK` 更能保留信息，但会以更长序列和更高学习难度为代价。

可以用下面的判据选择策略：

$$
\text{优先级：常见子词} \;>\; \text{byte fallback} \;>\; \text{UNK}
$$

这不是所有 tokenizer 的硬性实现顺序，而是信息保留与效率之间的常见设计取舍：先用紧凑 token 表示高频模式，无法覆盖时保留字节，只有无法安全表示或必须拒绝时才使用 `UNK`。

## 5. 一句话区分

- **Special token**：改变序列结构或控制模型行为的协议符号。
- **Byte fallback**：用字节级 token 覆盖词表之外的文本输入。
- **UNK**：承认无法表示，但用一个占位符替代原始内容。

判断 tokenizer 行为时，应同时查看词表、special token 配置、预分词器、fallback 规则、聊天模板和解码选项；只看 BPE merge 表无法解释完整的编码结果。
