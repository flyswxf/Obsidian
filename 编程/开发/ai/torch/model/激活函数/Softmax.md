# PyTorch Softmax 激活函数详解

## 核心思想
Softmax 激活函数用于多分类问题，它将一个实数向量映射为一个概率分布。其数学表达式为：
$$ \text{Softmax}(x_i) = \frac{e^{x_i}}{\sum_{j} e^{x_j}} $$
它会将输入的每个分量转化为 $(0, 1)$ 之间，且所有分量之和恒为 1，非常适合作为分类模型的最后一层输出。

## PyTorch 使用

```python
import torch
import torch.nn as nn

# 1. 作为网络层使用 (指定维度 dim)
softmax_layer = nn.Softmax(dim=1)
x = torch.tensor([[1.0, 2.0, 3.0]]) # 模拟一个 batch 中包含 3 个类别的得分
output = softmax_layer(x)

# 2. 函数式 API 调用
import torch.nn.functional as F
output_f = F.softmax(x, dim=1)

print("输入:\n", x)
print("输出概率分布:\n", output)
print("概率和:\n", output.sum(dim=1)) # 恒为 1.0
```

## 关键要点说明
- **数值稳定性**：指数计算极易导致数值溢出（OverFlow）。PyTorch 的实现通常会在分子分母同除以最大值的指数，以确保数值稳定。
- **与交叉熵损失结合**：在多分类任务中，**强烈建议直接使用 `nn.CrossEntropyLoss`**，因为它内部已经包含了 `nn.LogSoftmax` 和 `nn.NLLLoss`。如果在网络最后一层手动加上 `nn.Softmax` 再传给交叉熵损失，会导致重复计算且降低数值稳定性。
- **多维度计算（`dim` 参数）**：`dim` 的选择非常关键。如果是全连接层的输出 `[batch_size, num_classes]`，则应选择 `dim=1`；如果是图像分割任务的输出 `[batch_size, num_classes, H, W]`，通常在 `dim=1`（通道维度）上计算。
