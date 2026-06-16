## 抽象语法树（AST）简介

抽象语法树（Abstract Syntax Tree，简称 AST）是源代码语法结构的树状表现形式。在代码分析与转换工具中，AST 扮演着核心角色。ESLint 依靠 AST 来理解 JavaScript 代码的语义和结构。

## 解析过程

代码到 AST 的转换通常包含两个阶段：
1. **词法分析**（Lexical Analysis）：将源代码字符串转换为记号（Token）序列。
2. **语法分析**（Syntax Analysis）：根据语言的语法规则，将记号序列组织成树状数据结构。

ESLint 默认使用 Espree 作为解析器。解析后的 AST 遵循 ESTree 规范，每个节点代表一种语法结构（如变量声明、函数调用等）。

```json
// 源码示例: const answer = 42;
// 对应的 AST 结构片段（简化版）
{
  "type": "VariableDeclaration",
  "declarations": [
    {
      "type": "VariableDeclarator",
      "id": {
        "type": "Identifier",
        "name": "answer"
      },
      "init": {
        "type": "Literal",
        "value": 42,
        "raw": "42"
      }
    }
  ],
  "kind": "const"
}
```

## 在 ESLint 中的应用

ESLint 遍历生成的 AST，并在进入或离开特定节点时触发相应的事件。开发者可以编写监听器来捕获这些事件，从而检查代码是否违反了特定的编码规范。例如，监听 `VariableDeclaration` 节点以分析变量的声明方式。
