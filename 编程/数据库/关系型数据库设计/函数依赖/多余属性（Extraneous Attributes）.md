## 📌 什么是 Extraneous Attributes？

在**函数依赖（Functional Dependency, FD）** 里，某些属性其实是「多余的」，它们**对确定依赖的成立性没有影响**，可以删掉，而不影响依赖的含义。  
这种属性就叫做 **Extraneous Attributes（多余属性）**。

> 📖 **形式化定义：**  
> 设有函数依赖 **X → Y**，如果：

- 在 X 中存在某个属性 A，使得 **(X - {A}) → Y** 也能由原依赖集 F 推导出来  
    
    或者
    
- 在 Y 中存在某个属性 B，使得 **X → (Y - {B})** 也能由 F 推导出来
    

那么这个属性就是 **多余的 Extraneous Attribute**。

---

## 📌 举个例子

### 例1：左边多余属性（在确定性的一方）

函数依赖：

```
F = {A B → C, A → C}
```

这里我们注意到：

- A → C 已经存在了
    
- 那 A B → C 中，B 就是多余的
    

因为即使没有 B，A 也能推出 C  
👉 所以 B 是 **extraneous in the left side**

---

### 例2：右边多余属性（在被确定的一方）

函数依赖：

```
F = {A → B C, A → B}
```

- A → B 已经存在
    
- 那么 A → B C 中的 B 是多余的
    

可以改写成：

```
A → C
```

👉 所以 B 是 **extraneous in the right side**

---

## 📌 怎么判断 Extraneous Attributes？

### 判断左边多余（A in X）

把 A 从 X 去掉，看 (X - {A}) → Y 是否还能从 F 推导出来。 如果可以，就说明 A 是多余的。

### 判断右边多余（B in Y）

把 B 从 Y 去掉，看 X → (Y - {B}) 是否还能从 F 推导出来。 如果可以，说明 B 是多余的。

---

## 📌 为什么要找出来？

在做**最小覆盖（Minimal Cover）**的时候，我们希望：

- 每个依赖的右边只有一个属性
    
- 没有多余属性
    
- 没有多余依赖
    

所以找出并删掉这些 **Extraneous Attributes**，可以让依赖集更简洁、有效。

---

## 📌 小结一下：

|名称|位置|判断方法|
|:--|:--|:--|
|Extraneous on LHS|左边|去掉它，看剩下的是否还能推出右边|
|Extraneous on RHS|右边|去掉它，看左边是否还能推出剩下的|
