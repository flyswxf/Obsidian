一部分包在conda上没有(因此在[[micromamba]]上也没有), 此时必须通过pip下载
**但是pip也会下载到当前处在的conda环境中, 所以需要提前切换为正式的环境**
## 下载源配置
下载源配置在`~\AppData\Roaming\pip\pip.ini`

通过此命令查看**config文件位置**和**具体config内容**
```
pip config list
```

```
[global]

index-url = https://pypi.tuna.tsinghua.edu.cn/simple

extra-index-url = https://mirrors.aliyun.com/pypi/simple/
```

只要在任意一个源里找到了匹配的包，就会停止搜索并开始下载。
- `index-url`: 优先在这里找包
- `extra-index-url`: 如果主源里找不到，会去查找

注意: 一旦手动设置了 `index-url`，pip 就会**完全抛弃默认的官方源**

## 下载特定版本

```bash
pip install 'rouge-score>=0.1.2'
```

以下是已解决的问题: 
### 问题描述

在服务器上运行以下命令时没有反应，命令直接结束且没有任何输出：

```bash
pip install rouge-score>=0.1.2
```

### 原因分析

在 bash shell 中，`>` 是**输出重定向运算符**，有特殊含义。

Shell 会将上述命令解析为：
- `pip install rouge-score` → 安装包
- `> 0.1.2` → 将输出重定向到名为 `0.1.2` 的文件

这导致：
1. pip 可能已经安装成功但输出被重定向到文件
2. 或者命令行为异常

### 解决方案

#### 使用引号

```bash
pip install 'rouge-score>=0.1.2'
```

### 验证安装

```bash
pip show rouge-score
```
## 知识扩展：Shell 元字符

以下字符在 Shell 中有特殊含义，当需要作为普通字符使用时需用引号或转义：

| 字符 | 含义 |
|------|------|
| `>` | 输出重定向 |
| `<` | 输入重定向 |
| `\|` | 管道 |
| `&` | 后台运行 |
| `*` | 通配符 |
| `?` | 通配符 |
| `;` | 命令分隔 |

