### 文件描述

- ​**​功能​**​: 生成CCSCM、CCSPROC和ATC3代码的嵌入向量

### 输入文件

- `../../resources/CCSCM.csv` - CCSCM疾病分类代码和名称
- `../../resources/CCSPROC.csv` - CCSPROC医疗过程代码和名称
- `../../resources/ATC.csv` - ATC药物分类代码和名称

### 主要功能

1. 读取CCSCM、CCSPROC和ATC3代码及其名称
2. 使用OpenAI的embedding API为每个代码的名称生成嵌入向量
3. 保存代码ID到嵌入的映射

### 输出文件

- `/data/pj20/exp_data/ccscm_id2emb.pkl` - CCSCM代码到嵌入的映射
- `/data/pj20/exp_data/ccsproc_id2emb.pkl` - CCSPROC代码到嵌入的映射
- `/data/pj20/exp_data/atc3_id2emb.pkl` - ATC3代码到嵌入的映射

### 依赖文件

- `get_emb.py` - 嵌入向量生成工具