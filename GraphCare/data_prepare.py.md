# data_prepare.py - 数据准备和处理脚本

## 文件描述
- **文件类型**: Python脚本
- **功能**: 数据集加载、聚类、图处理和数据集预处理
- **位置**: `data_prepare.py`

## 输入文件
- [[CCSCM.csv]], [[CCSPROC.csv]], [[ATC.csv]] - 原始映射文件
- [[graphs/cond_proc/CCSCM_CCSPROC/entity_embedding.pkl]] - 实体嵌入
- [[graphs/cond_proc/CCSCM_CCSPROC/relation_embedding.pkl]] - 关系嵌入
- MIMIC-III/IV数据集

## 主要功能
1. **数据集加载**: 从MIMIC数据库加载和预处理
2. **聚类**: 对实体和关系进行层次聚类
3. **图处理**: 构建患者特定的子图
4. **数据集处理**: 为模型训练准备数据

## 输出文件
### 聚类结果
- [[clustering/ccscm_ccsproc/clusters_inv_th015.json]] - 实体聚类映射
- [[clustering/ccscm_ccsproc/clusters_inv_rel_th015.json]] - 关系聚类映射
- [[clustering/ccscm_ccsproc/ccscm_id2clus.json]] - CCSCM ID到聚类映射
- [[clustering/ccscm_ccsproc/ccsproc_id2clus.json]] - CCSPROC ID到聚类映射

### 处理后的数据
- `/data/pj20/exp_data/ccscm_ccsproc/sample_dataset_mimic3_drugrec_th015.pkl` - 药物推荐数据集
- `/data/pj20/exp_data/ccscm_ccsproc/graph_mimic3_drugrec_th015.pkl` - 药物推荐图结构
- `/data/pj20/exp_data/ccscm_ccsproc_atc3/sample_dataset_mimic3_mortality_th015.pkl` - 死亡率预测数据集

## 后续使用
- [[graphcare.py]] - 模型训练
- [[graphcare_analysis.py]] - 模型分析