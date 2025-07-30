# CCSPROC.csv - 临床分类系统过程映射文件

## 文件描述
- **文件类型**: CSV映射文件
- **内容**: 临床分类系统（CCS）过程映射，包含医疗程序代码和名称
- **位置**: `resources/CCSPROC.csv`

## 数据结构
- `code`: 医疗程序代码
- `name`: 医疗程序名称

## 作为输入的脚本
- [[graph_gen.ipynb]] - 知识图谱生成
- [[proc_emb_ret.ipynb]] - 程序实体嵌入生成
- [[data_prepare.py]] - 数据准备和处理

## 生成的输出文件
- [[graphs/procedure/CCSPROC/{code_id}.txt]] - 每个程序代码对应的知识图谱文件
- [[graphs/procedure/CCSPROC/ent2id.json]] - 实体到ID的映射
- [[graphs/procedure/CCSPROC/entity_embedding.pkl]] - 实体嵌入向量

## 相关文件
- [[CCSCM.csv]] - 疾病分类文件
- [[ATC.csv]] - 药物分类文件
- [[ICD9PROC.csv]] - ICD-9程序映射文件