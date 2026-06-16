# 架构背景

在 React 15 及更早版本中，组件的更新过程是同步且不可中断的。当组件树结构庞大时，深度遍历的渲染过程会长时间占用主线程，导致页面出现掉帧或卡顿现象。为了解决这一性能瓶颈，React 引入了 Fiber 架构，为后续支持\[\[并发模式|并发渲染特性]]奠定了基础。

## 核心概念

Fiber 本质上是一个普通的 JavaScript 对象，它同时扮演着三个关键角色：

1. **作为架构核心**：标志着 React 从基于调用栈的同步渲染转变为基于链表的异步可中断渲染。
2. **作为静态数据结构**：
   - **与 JSX 的关系**：我们在代码中写的 JSX 元素（如 `<div id="box"></div>`）在编译后会变成纯 JS 对象，它仅仅描述了“页面在这一刻应该长什么样”。而 Fiber 节点是 React 内部用来保存组件状态、对应真实 DOM 节点（如果有）以及各种副作用标记的**实例对象**。React 会根据 JSX 元素来创建或更新对应的 Fiber 节点。
   - **链表结构设计**：为了实现“可中断与恢复”，Fiber 将原本的递归树结构改造成了**单链表树结构**。每个 Fiber 节点包含三个核心指针：
     - `child`：指向第一个子节点。
     - `sibling`：指向右侧的下一个兄弟节点。
     - `return`（或 `parent`）：指向父节点。

**Fiber 链表结构图示：**

![[fiber_linked_list.svg]]

如上图所示，当 React 遍历到 `Header` 节点并发现没有子节点时，它可以通过 `sibling` 指针找到 `Main`，或者通过 `return` 指针回到父级。这种指针结构使得 React 可以随时记住当前执行到了哪个节点，从而安全地中断和恢复渲染任务。

3. **作为动态工作单元**：保存了组件本次更新中的状态改变信息以及需要执行的副作用（如 DOM 操作）。

### 双缓存树机制

React 在内存中同时维护两棵 Fiber 树：

- **current 树**：对应当前屏幕上显示的 DOM 结构。
- **workInProgress 树**：正在内存中构建的树，所有的更新计算均在此树上进行。

当更新操作完成时，React 会直接将 `current` 指针指向 `workInProgress` 树，从而快速完成视图更新。这种机制有效避免了在渲染过程中出现屏幕闪烁。

## 渲染流程

Fiber 的工作流程主要分为两个阶段：

### Render 阶段

此阶段的主要任务是构建 `workInProgress` 树，并计算出需要进行的 DOM 变更。这个过程是纯粹的 JavaScript 计算，并且是**可中断**的。如果浏览器有更高优先级的任务（如用户输入、动画），React 可以暂停当前的构建工作，优先处理紧急任务，之后再恢复。

**核心工作步骤（Diff 与 Effect List）**：
1. **Diff 算法执行**：当组件触发更新时，React 会沿着 Fiber 链表进行深度优先遍历。在遍历到某个节点时，React 会拿**内存中的旧节点**（`current` Fiber）与**最新的 JSX 元素**（React Element）进行对比。
2. **打标记（Flags）**：如果对比发现两者不同（比如类型变了、属性变了，或者有节点需要被删除），React 并不会立刻去修改 DOM，而是给当前这个正在构建的 `workInProgress` Fiber 节点打上一个“副作用标记（Flags）”，例如：
   - `Placement`：需要插入新 DOM。
   - `Update`：DOM 属性或内容需要更新。
   - `Deletion`：需要删除旧 DOM。
3. **构建副作用链（Effect List）**：在早期版本的 Fiber 架构（React 16/17）中，为了避免在 Commit 阶段重新遍历整棵庞大的树去寻找需要更新的节点，React 会在 Render 阶段的回溯过程中，将所有被打上标记的 Fiber 节点用指针串联起来，形成一条单向链表（即 Effect List）。
   > **注**：在 React 18 中，Effect List 被重构为了 `subtreeFlags`（子树副作用标记）机制，但其核心目的依然是为了在 Commit 阶段**快速跳过没有发生变化的子树**，直接定位到需要操作 DOM 的节点。

### Commit 阶段

此阶段负责将 Render 阶段计算出的副作用实际应用到 DOM 上。与 Render 阶段不同，此过程是**同步且不可中断**的，以保证用户看到的界面是一致且完整的。Commit 阶段可以细分为三个子阶段：

1. **Before Mutation**：执行 DOM 操作前，通常用于调用 `getSnapshotBeforeUpdate` 等生命周期方法。
2. **Mutation**：执行实际的 DOM 插入、更新或删除操作。
3. **Layout**：DOM 操作完成后，触发 `useLayoutEffect` 和相关的生命周期钩子，此时可以获取到最新的 DOM 布局信息。

## 调度机制

Fiber 架构允许对不同类型的任务进行优先级划分。通过内部的时间切片（Time Slicing）机制，React 将执行权交还给浏览器，以确保页面的流畅度。这也是实现复杂交互下\[\[状态管理设计#性能优化考量|状态高效更新]]的核心支撑。

## 现代 React 代码实践：感知 Fiber 的执行阶段

虽然 Fiber 架构在底层运行，但开发者可以通过不同的 Hooks 观察到 Render 阶段与 Commit 阶段的区别。

### useEffect 与 useLayoutEffect 的时机差异

在 Commit 阶段中，`useLayoutEffect` 会在 DOM 变更后同步触发，而 `useEffect` 则是异步触发。通过以下代码可以直观感受到这种差异：

```javascript
import { useState, useEffect, useLayoutEffect, useRef } from 'react';

export default function FiberPhaseDemo() {
  const [count, setCount] = useState(0);
  const divRef = useRef(null);

  // 触发于 Commit 阶段的 Layout 子阶段（同步，阻塞浏览器绘制）
  useLayoutEffect(() => {
    if (count === 0) {
      // 可以在此处获取到最新的 DOM 结构，如果在此处触发状态更新，会进行同步重绘
      const rect = divRef.current.getBoundingClientRect();
      console.log('useLayoutEffect: 同步读取 DOM', rect);
    }
  }, [count]);

  // 触发于 Commit 阶段之后（异步，不阻塞浏览器绘制）
  useEffect(() => {
    console.log('useEffect: 异步副作用执行', count);
  }, [count]);

  return (
    <div ref={divRef}>
      <p>当前计数: {count}</p>
      <button onClick={() => setCount(c => c + 1)}>增加</button>
    </div>
  );
}
```

这段代码展示了在\[\[Hooks原理解析#副作用处理：useEffect|副作用处理]]时，应当默认使用 `useEffect`，以避免阻塞主线程。只有在需要同步读取 DOM 并可能触发重绘以避免闪烁时，才使用 `useLayoutEffect`。
