### 文件描述

- ​**​功能​**​: 生成UMLS概念的嵌入向量

### 输入文件

- `/home/pj20/GraphCare/KG_mapping/umls/concept_names.txt` - UMLS概念名称列表

### 主要功能

1. 读取UMLS概念名称列表
2. 使用OpenAI的embedding API（通过 get_emb.py 中的 embedding_retriever 函数）为每个概念生成嵌入向量
3. 定期保存嵌入结果，并在处理完所有概念后保存最终结果

### 输出文件

- `/data/pj20/exp_data/umls_ent_emb_.pkl` - UMLS实体嵌入向量

### 依赖文件

- `get_emb.py` - 嵌入向量生成工具