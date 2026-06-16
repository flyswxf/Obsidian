## JSX 语法基础

JSX 是一种 JavaScript 的语法扩展，允许在 JavaScript 代码中编写类似 HTML 的标记。它是描述 UI 结构的直观方式。在构建过程中，JSX 会被编译为 `React.createElement()` 或现代的 `jsx()` 函数调用，最终转换为虚拟 DOM 节点（即[[Fiber架构解析#核心概念|Fiber 节点]]对应的数据结构）。

### 核心规则与表达式

- **单一根节点**：JSX 结构必须由一个唯一的父元素包裹。若不需要额外的 DOM 节点，可以使用 `<>...</>`（即 `Fragment`）进行包裹。
- **嵌入表达式**：使用大括号 `{}` 可以在 JSX 中嵌入任意合法的 JavaScript 表达式，包括变量、函数调用或计算结果。
- **驼峰命名**：HTML 属性在 JSX 中需要采用小驼峰命名法（如 `className` 替代 `class`，`onClick` 替代 `onclick`）。

### 条件渲染与列表渲染

```javascript
import { useState } from 'react';

export default function JsxDemo() {
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const tasks = [
    { id: 1, title: '学习 React' },
    { id: 2, title: '理解 JSX' }
  ];

  return (
    <div className="container">
      {/* 条件渲染：使用三元运算符或逻辑与 && */}
      <header>
        {isLoggedIn ? (
          <p>欢迎回来！</p>
        ) : (
          <button onClick={() => setIsLoggedIn(true)}>登录</button>
        )}
      </header>

      {/* 列表渲染：使用 map 方法遍历数组，必须提供唯一的 key 属性 */}
      <ul className="task-list">
        {tasks.map(task => (
          <li key={task.id}>{task.title}</li>
        ))}
      </ul>
    </div>
  );
}
```

## 组件渲染原理

现代 React 开发全面拥抱函数组件与[[Hooks原理解析|Hooks 特性]]。组件本质上是一个接收参数（Props）并返回 JSX 的普通 JavaScript 函数。

### 渲染触发时机

组件的渲染主要由以下两种情况触发：
1. **初次挂载（Mount）**：组件首次插入 DOM 树中。
2. **状态更新（Update）**：组件内部状态（如 `useState`、`useReducer`）或传入的属性（Props）发生变化。

### 纯函数原则与副作用

React 强调组件应当表现得像一个**纯函数**：对于相同的输入（Props 和 State），必须始终返回相同的 JSX。
- **渲染过程必须无副作用**：在组件主体内部（即[[Fiber架构解析#Render 阶段|Render 阶段]]）绝不能执行修改外部变量、直接操作 DOM 或发起网络请求等副作用操作。
- **副作用处理**：所有的副作用必须包裹在事件处理函数中，或放置在 `useEffect` 中，交由 React 在[[Fiber架构解析#Commit 阶段|Commit 阶段]]之后安全执行。

```javascript
// 纯函数组件示例
function Profile({ user }) {
  // 错误：在渲染期间产生副作用
  // document.title = user.name; 

  // 正确：将副作用交给 useEffect 处理
  useEffect(() => {
    document.title = user.name;
  }, [user.name]);

  return <section>姓名: {user.name}</section>;
}
```
