1. 计算[[键#2. ​**​候选键（Candidate Key）​**​]]
2. 从函数依赖集F中选择一个不满足BCNF的函数依赖,α → β 分解关系模式R
	1. R1={α , β}, R2 = R-β
	2. 构造出的R1上, α → β能满足BCNF
3. 在R2上, 计算不含β的新函数依赖集F'
4. F=F', R=R2, 重新执行步骤2

BCNF分解到的分解一定是[[无损分解（Lossless Decomposition）的判定]]的, 但不一定[[保持函数依赖(Dependency Preservation)]]