# cond_emb_ret.ipynb - 条件实体嵌入生成脚本

## 文件描述
- **文件类型**: Jupyter Notebook
- **功能**: 从条件知识图谱生成实体和关系的嵌入向量
- **位置**: `graphcare_/graph_generation/cond_emb_ret.ipynb`

## 输入文件
- [[CCSCM.csv]] - 疾病分类映射
- `graphs/condition/CCSCM/{code_id}.txt` - 由[[graph_gen.ipynb]]生成的知识图谱文件

## 主要功能
1. 解析知识图谱文件，提取实体和关系
2. 创建实体和关系的ID映射
3. 使用预训练模型生成嵌入向量
4. 保存映射文件和嵌入向量

## 输出文件
- [[graphs/condition/CCSCM/id2ent.json]] - ID到实体的映射
- [[graphs/condition/CCSCM/ent2id.json]] - 实体到ID的映射
- [[graphs/condition/CCSCM/id2rel.json]] - ID到关系的映射
- [[graphs/condition/CCSCM/rel2id.json]] - 关系到ID的映射
- [[graphs/condition/CCSCM/entity_embedding.pkl]] - 实体嵌入向量
- [[graphs/condition/CCSCM/relation_embedding.pkl]] - 关系嵌入向量

## 后续使用
- [[data_prepare.py]] - 数据准备和聚类
- [[graphcare.py]] - 模型训练

## 依赖文件
- [[get_emb.py]] - 嵌入向量生成工具