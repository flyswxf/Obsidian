# umls_sampling.py - UMLS知识图谱采样脚本

## 文件描述
- **文件类型**: Python脚本
- **功能**: 从UMLS知识图谱中采样相关子图
- **位置**: `KG_mapping/umls_sampling.py`

## 输入文件
- `/data/pj20/exp_data/ccscm2umls.pkl` - CCSCM到UMLS映射
- `/data/pj20/exp_data/ccsproc2umls.pkl` - CCSPROC到UMLS映射
- `/data/pj20/exp_data/atc32umls.pkl` - ATC3到UMLS映射
- [[KG_mapping/umls/umls.csv]] - UMLS三元组数据
- [[KG_mapping/umls/concepts.txt]] - UMLS概念列表

## 主要功能
1. 为每个医疗编码找到UMLS中的一跳邻居
2. 随机采样二跳邻居（最多5个）
3. 构建医疗编码特定的UMLS子图

## 输出文件
- `graphs/ccscm_umls/first_hop_triples.pkl` - CCSCM一跳三元组
- `graphs/ccscm_umls/second_hop_triples.pkl` - CCSCM二跳三元组
- `graphs/ccsproc_umls/first_hop_triples.pkl` - CCSPROC一跳三元组
- `graphs/atc3_umls/first_hop_triples.pkl` - ATC3一跳三元组

## 相关文件
- [[process.ipynb]] - UMLS映射处理
- [[KG_mapping/umls/umls.csv]] - UMLS原始数据