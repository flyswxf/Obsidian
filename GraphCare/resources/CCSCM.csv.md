# CCSCM.csv - 临床分类系统条件映射文件

## 文件描述
- **文件类型**: CSV映射文件
- **内容**: 临床分类系统（CCS）条件映射，包含疾病代码和名称
- **位置**: `resources/CCSCM.csv`
- **总行数**: 286行

## 数据结构
- `code`: 疾病代码（如1, 10, 100等）
- `name`: 疾病名称（如Tuberculosis, Acute myocardial infarction等）

## 作为输入的脚本
- [[graph_gen.ipynb]] - 知识图谱生成
- [[cond_emb_ret.ipynb]] - 条件实体嵌入生成
- [[data_prepare.py]] - 数据准备和处理

## 生成的输出文件
- [[graphs/condition/CCSCM/{code_id}.txt]] - 每个疾病代码对应的知识图谱文件
- [[graphs/condition/CCSCM/ent2id.json]] - 实体到ID的映射
- [[graphs/condition/CCSCM/entity_embedding.pkl]] - 实体嵌入向量

## 相关文件
- [[ATC.csv]] - 药物分类文件
- [[CCSPROC.csv]] - 医疗程序分类文件
- [[ICD9CM.csv]] - ICD-9-CM映射文件