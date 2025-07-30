# graph_gen.ipynb - 知识图谱生成脚本

## 文件描述
- **文件类型**: Jupyter Notebook
- **功能**: 使用ChatGPT从医疗编码映射文件生成知识图谱
- **位置**: `graphcare_/graph_generation/graph_gen.ipynb`

## 输入文件
- [[CCSCM.csv]] - 疾病分类映射
- [[CCSPROC.csv]] - 医疗程序映射
- [[ATC.csv]] - 药物分类映射（level 3.0）

## 主要功能
1. 读取映射文件并构建字典
2. 使用ChatGPT API生成医疗实体的关系三元组
3. 为每个医疗代码生成约100个三元组
4. 输出格式：`实体1\t关系\t实体2`

## 输出文件
- `graphs/condition/CCSCM/{code_id}.txt` - 疾病知识图谱文件
- `graphs/procedure/CCSPROC/{code_id}.txt` - 程序知识图谱文件
- `graphs/drug/ATC3/{code_id}.txt` - 药物知识图谱文件
**每个生成的知识图谱文件包含多个三元组，格式为： 实体1\t关系\t实体2**

## 后续处理脚本
- [[cond_emb_ret.ipynb]] - 条件实体嵌入生成
- [[proc_emb_ret.ipynb]] - 程序实体嵌入生成
- [[drug_emb_ret.ipynb]] - 药物实体嵌入生成

## 依赖文件
- [[ChatGPT.py]] - ChatGPT API接口