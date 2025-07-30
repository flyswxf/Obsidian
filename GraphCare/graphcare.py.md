# graphcare.py - GraphCare模型训练脚本

## 文件描述
- **文件类型**: Python脚本
- **功能**: GraphCare模型的训练和评估
- **位置**: `graphcare.py`

## 输入文件
- `sample_dataset_*.pkl` - 由[[data_prepare.py]]生成的处理后数据集
- `graph_*.pkl` - 由[[data_prepare.py]]生成的图结构
- [[graphs/cond_proc/CCSCM_CCSPROC/entity_embedding.pkl]] - 实体嵌入
- [[graphs/cond_proc/CCSCM_CCSPROC/relation_embedding.pkl]] - 关系嵌入
- [[clustering/ccscm_ccsproc/clusters_inv_th015.json]] - 聚类映射

## 主要功能
1. 加载预处理的数据集和图结构
2. 构建GraphCare模型
3. 训练模型并评估性能
4. 生成预测结果

## 输出文件
- `training_logs/{dataset}_{task}_{kg}_{params}.log` - 训练日志
- 模型性能指标和预测结果

## 支持的任务
- 药物推荐 (drugrec)
- 死亡率预测 (mortality)
- 再入院预测 (readmission)
- 住院时长预测 (lenofstay)

## 相关文件
- [[graphcare_analysis.py]] - 模型分析脚本
- [[graphcare_/model.py]] - 模型定义