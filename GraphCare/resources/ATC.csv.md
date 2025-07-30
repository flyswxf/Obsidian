# ATC.csv - 药物解剖学治疗化学分类系统

## 文件描述
- **文件类型**: CSV映射文件
- **内容**: 药物ATC分类系统，包含药物代码、名称、层级和描述
- **位置**: `resources/ATC.csv`
- **总行数**: 15560行

## 数据结构
- `code`: 药物ATC代码
- `parent_code`: 父级代码
- `name`: 药物名称
- `level`: 分类层级（1.0-5.0）
- `description`: 描述
- `indication`: 适应症
- `smiles`: 化学结构
- `drugbank_id`: DrugBank数据库ID

## 作为输入的脚本
- [[graph_gen.ipynb]] - 知识图谱生成（使用level 3.0的药物）
- [[drug_emb_ret.ipynb]] - 药物实体嵌入生成
- [[data_prepare.py]] - 数据准备和处理

## 生成的输出文件
- [[graphs/drug/ATC3/{code_id}.txt]] - 每个药物代码对应的知识图谱文件
- [[graphs/drug/ATC3/ent2id.json]] - 实体到ID的映射
- [[graphs/drug/ATC3/entity_embedding.pkl]] - 实体嵌入向量

## 相关文件
- [[CCSCM.csv]] - 疾病分类文件
- [[CCSPROC.csv]] - 医疗程序分类文件
- [[ATC_to_UMLS.csv]] - ATC到UMLS的映射