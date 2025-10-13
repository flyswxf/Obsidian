# Maude 模块 Modules

## 1. 模块概述
- **定义**：Maude 中规范和编程的基本单元
- **组成**：
  - **签名 (Signature)**：语法声明部分
  - **语句 (Statements)**：断言系统属性

## 2. 签名 (Signature) 组成
签名包含以下声明：

| 声明类型 | 作用 | 说明 |
|---------|------|------|
| **sorts** | 数据类型名称 | 为数据类型命名 |
| **subsorts** | 类型层次结构 | 组织数据类型的层次关系 |
| **kinds** | 错误超类型 | 隐式的，包含正常数据和错误表达式 |
| **operators** | 操作名称 | 定义作用于数据的操作，构建表达式 |

## 3. 模块类型

### 3.1 函数模块 (Functional Modules)
**语法**：
```maude
fmod MODULE-NAME is
  ...
endfm
```

**特点**：
- **语句类型**：方程 (equations) + 成员关系 (memberships)
- **编程视角**：等式风格的函数程序，用户可定义语法
- **规范视角**：等式理论 (Σ, E)，初始代数语义
- **用途**：定义 sorts、元素和函数

### 3.2 系统模块 (System Modules)
**语法**：
```maude
mod MODULE-NAME is
  ...
endm
```

**特点**：
- **语句类型**：方程 + 成员关系 + 规则 (rules)
- **编程视角**：声明式并发程序，用户可定义语法
- **规范视角**：重写理论 (Σ, E ∪ R)，初始模型语义
- **用途**：描述状态间的转换

## 4. 命名约定
- **标识符**：任何 Maude 标识符都可用
- **推荐风格**：
  - 全大写字母
  - 复合名称用连字符连接
  - 示例：`NUMBERS`、`VENDING-MACHINE`

## 5. 模块内容结构
```maude
fmod/mod MODULE-NAME is
  // 子模块导入
  // sorts 声明
  // subsorts 声明  
  // operators 声明
  // variables 声明
  // equations 声明
  // rules 声明 (仅系统模块)
endfm/endm
```

## 6. 理论基础对比

| 模块类型 | 理论基础 | 语义 | 主要用途 |
|---------|----------|------|----------|
| 函数模块 | 等式理论 (Σ, E) | 初始代数语义 | 函数式编程 |
| 系统模块 | 重写理论 (Σ, E ∪ R) | 初始模型语义 | 并发系统建模 |

## 7. 示例

### 7.1 函数模块示例
```maude
fmod NUMBERS is
  sorts Nat .
  op 0 : -> Nat .
  op s : Nat -> Nat .
  op _+_ : Nat Nat -> Nat .
  
  vars M N : Nat .
  eq 0 + N = N .
  eq s(M) + N = s(M + N) .
endfm
```

### 7.2 系统模块示例
```maude
mod VENDING-MACHINE is
  sorts Coin Item State .
  
  ops quarter dime : -> Coin .
  ops apple candy : -> Item .
  
  op [_] : Nat -> State .
  op insert_in_ : Coin State -> State .
  
  vars N : Nat .
  rl [insert quarter in [N]] => [N + 25] .
  rl [insert dime in [N]] => [N + 10] .
endm
```