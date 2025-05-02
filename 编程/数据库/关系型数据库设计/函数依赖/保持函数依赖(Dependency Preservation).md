## 📌 什么是保持函数依赖（Dependency Preservation）？

在数据库规范化过程中，我们经常需要把一个大表（关系）拆分成多个小表（子关系），以消除冗余、避免更新异常、提高一致性。

拆分的原则有很多，比如：

- 保持无损连接性（Lossless-Join）[[无损分解（Lossless Decomposition）的判定]]
    
- 保持函数依赖（Dependency Preservation）
    

其中**保持函数依赖**，就是希望：

> **原来在拆分前表中存在的所有函数依赖，在拆分后的子关系中依然能够被检查和验证，无需在多个子关系之间做笛卡尔积再去验证。**
> 
> **但是可以在多个表中逐个做推导, 只要不做笛卡尔积就行


## 📌 为什么重要？

> 🌱 **测试一个关系内的依赖代价低** 🌱 **测试多个关系（尤其是需要拼接的）代价高**

所以好设计要尽量保证：

- 拆分后依赖还在子关系里能直接检查
    
- 不要为了范式太过追求分解，结果导致依赖验证变得麻烦
    

---

## 📌 计算是否保持函数依赖的算法（Dependency Preservation Test）

我们来看看怎么判断拆分后的子关系是否保持了所有的函数依赖。

### 设定：

- R：原始关系
    
- F：原始关系的依赖集
    
- ρ = {R1, R2, ..., Rn}：分解后的子关系集合
    

### 算法步骤：

**对每个关系依赖f a->b**
1. 将a投影到Ri, 即求a∩Ri
2. 在Ri上, 计算属性的闭包a+
	1. 使用全部函数依赖计算
	2. 计算结果与Ri求∩
3. 令a' = a+ U a, 选择另一个Rj, 重复过程
4. 如果能让a' => a ∪ b, 那么这个分解保持函数依赖

### 例子:
![[Pasted image 20250417193312.png]]


还可查看[BCNF Decomposition | A step by step approach – Data Science Duniya](https://ashutoshtripathi.com/gate/dbms/normalization-normal-forms/procedure-to-decompose-a-given-relation-in-bcnf-bcnf-algorithm/)