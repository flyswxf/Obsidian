### 文件描述

- ​**​功能​**​: 基于嵌入相似度将CCSCM、CCSPROC和ATC3代码映射到UMLS概念

### 输入文件

- `/home/pj20/GraphCare/KG_mapping/umls/concept_names.txt` - UMLS概念名称列表
- `/data/pj20/exp_data/umls_ent_emb_.pkl` - UMLS实体嵌入向量
- `/data/pj20/exp_data/ccscm_id2emb.pkl` - CCSCM代码到嵌入的映射
- `/data/pj20/exp_data/ccsproc_id2emb.pkl` - CCSPROC代码到嵌入的映射
- `/data/pj20/exp_data/atc3_id2emb.pkl` - ATC3代码到嵌入的映射

### 主要功能

1. 读取UMLS概念名称和ID
2. 读取UMLS实体嵌入和各种医疗代码嵌入
3. 使用余弦相似度计算每个医疗代码嵌入与UMLS实体嵌入的相似度
4. 为每个医疗代码找到最相似的UMLS概念（相似度阈值为0.7）
5. 保存映射结果

### 输出文件

- `/data/pj20/exp_data/ccscm2umls.pkl` - CCSCM代码到UMLS概念的映射
- `/data/pj20/exp_data/ccsproc2umls.pkl` - CCSPROC代码到UMLS概念的映射
- `/data/pj20/exp_data/atc32umls.pkl` - ATC3代码到UMLS概念的映射