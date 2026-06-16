## 设计理念

Hooks 是 React 16.8 引入的特性，旨在解决类组件中状态逻辑复用困难、生命周期方法中逻辑分散以及 `this` 指向模糊等问题。通过 Hooks，可以在不编写类的形式下使用状态及其他 React 特性。

## 底层数据结构

在执行组件函数时，React 会在对应的[[Fiber架构解析#核心概念|Fiber节点]]上挂载一个 `memoizedState` 属性。对于函数组件，这个属性指向一个单向链表，链表中的每一个节点对应组件中调用的一个 Hook。

### 链表结构特征

- 每一个 Hook（如 `useState`、`useEffect`）在执行时，都会在这个链表上按顺序追加或读取对应的状态节点。
- 这种依赖执行顺序的机制，决定了不能在条件语句、循环或嵌套函数中调用 Hooks。如果调用顺序发生变化，React 将无法正确匹配旧状态，导致难以排查的 Bug。

## 核心 Hook 解析

### 状态管理：useState 与 useReducer

`useState` 是 React 中最基础的[[状态管理设计|状态管理]] Hook，内部实际上是 `useReducer` 的简化语法糖。其参数与返回值规范如下：

- **接收参数 (Initial State)**：
	- **含义**：状态的初始值。
	- **类型**：可以是任何数据类型。
	- **惰性初始化**：若初始状态计算开销较大，也可传入一个函数 `() => initialState`，该函数仅在组件首次渲染时执行一次。
- **返回内容 (State Variables)**：
	- **含义**：返回一个包含两个元素的数组。通常采用 ES6 数组解构的方式提取。
		- **元素一 (Current State)**：当前的状态只读值。组件每次渲染时，该值都会保持最新。
		- **元素二 (Set State Function)**：用于更新状态的调度函数。调用状态更新函数时，React 会创建一个更新任务（Update），将其加入到该 Hook 的更新队列中，并触发组件的重新渲染。在下一次渲染的[[Fiber架构解析#Render 阶段|Render 阶段]]，React 会遍历更新队列，计算出最新的状态值。
  - **更新模式**：
    - **直接更新**：直接传入新值。
    - **函数式更新**：传入一个回调函数，React 会将前一次的最新状态作为参数传递给该回调。当新状态依赖于旧状态时必须使用此模式，如在[[组件间通信#父子组件通信|子传父通信回调]]中常用的更新方式。

### 副作用处理：useEffect

`useEffect` 用于处理与视图渲染无关的副作用逻辑（如数据请求、DOM 操作、订阅事件等）。React 会在组件渲染并提交到屏幕之后，异步执行这些副作用。

#### 核心机制与使用说明

- **执行时机**：默认情况下，`useEffect` 在组件每次渲染（包括首次挂载和后续更新）后都会执行。
- **依赖数组 (Dependencies Array)**：通过传入第二个参数（数组），可以控制副作用的执行频率。React 会通过 `Object.is()` 对比前后两次渲染的依赖数组。
	- **不传数组**：每次渲染后都会执行。
	- **空数组 `[]`**：只在组件首次挂载时执行一次（常用于初始化请求或绑定全局事件）。
	- **有依赖项的数组 `[A, B]`**：只有当 `A` 或 `B` 发生变化时，才会重新执行副作用。如果依赖项为引用类型（如对象或函数），应当注意使用 `useMemo` 或 `useCallback` 避免不必要的重复执行。
- **清理函数 (Cleanup Function)**：`useEffect` 可以返回一个函数。React 会在组件卸载时，或者下一次副作用执行**前**调用它。用于清理定时器、解绑事件等，以防内存泄漏。

## 闭包陷阱与应对策略

由于函数组件每次渲染都会形成独立的闭包，如果异步回调或定时器中捕获了旧渲染作用域中的状态，就会产生“闭包陷阱”。

### 什么是闭包陷阱？

在 JavaScript 中，函数在创建时会“记住”其所在词法作用域中的变量（这就是[[闭包与作用域链|闭包]]）。在 React 中，**每一次渲染都会重新执行一次组件函数**，从而产生一个全新的作用域和全新的状态变量。

**会产生闭包陷阱的错误代码：

```javascript
import { useState, useEffect } from 'react';

export default function BadTimer() {
  const [count, setCount] = useState(0);

  useEffect(() => {
    const timer = setInterval(() => {
      // 错误示范：这里的 count 永远是初次渲染时的 0
      // 因为 useEffect 依赖数组为空 []，回调函数只在挂载时创建了一次，
      // 它闭包内捕获的 count 永远指向第一次渲染时的作用域。
      console.log("定时器执行，当前的 count 是:", count); 
      setCount(count + 1); // 这会导致 count 永远从 0 变成 1，页面卡在 1 不动
    }, 1000);

    return () => clearInterval(timer);
  }, []); // 空依赖数组

  return <div>计时器: {count}</div>;
}
```

**场景推演**：
1. **第 1 次渲染**：`count` 是 `0`。此时在 `useEffect` 中开启了一个 `setInterval`。由于闭包的特性，这个定时器的回调函数永远记住了**这次渲染**的作用域，即 `count` 永远等于 `0`。
2. **点击按钮更新**：我们将 `count` 更新为 `1`。React 触发第 2 次渲染，产生了一个新的作用域，其中新的 `count` 变成了 `1`。
3. **陷阱爆发**：但是，之前那个 `setInterval` 是在第 1 次渲染时创建的！它没有被销毁，它所引用的依然是第 1 次渲染作用域里的那个 `count = 0`。所以，无论你之后怎么更新状态，定时器里打印出来的永远是 `0`。

这就是所谓的“闭包陷阱”：**异步回调中持有了旧渲染周期中的废弃状态**。

### 解决思路

1. **使用函数式更新**：当新的状态依赖于前一个状态时，向 `setXxx` 传递一个回调函数（如 `setCount(c => c + 1)`），确保始终基于最新状态进行计算。
2. **借助 useRef**：利用 `useRef` 跨越渲染周期的特性，将经常变化且不需触发视图更新的值保存在 `ref.current` 中，从而在任何闭包中都能访问到最新值。

### 代码实践：完美避开闭包陷阱

针对上述问题，以下代码展示了**两种经典的解决方案**：

```javascript
import { useState, useEffect, useRef } from 'react';

export default function GoodTimer() {
  const [count, setCount] = useState(0);
  
  // 解决方案 B 的辅助变量：使用 useRef 跨渲染周期保存最新值
  const latestCountRef = useRef(count);
  useEffect(() => {
    latestCountRef.current = count;
  }, [count]);

  useEffect(() => {
    const timer = setInterval(() => {
      // 解决方案 A：使用函数式更新。
      // React 会将前一次的真实最新状态传给回调，不需要依赖外部闭包中的 count。
      setCount(prev => prev + 1);

      // 解决方案 B：通过 ref 读取最新状态。
      // ref.current 的引用是稳定的，不会受闭包作用域限制，适用于不需要触发重渲染的只读逻辑。
      console.log("最新计数值:", latestCountRef.current);
    }, 1000);

    // 清理函数：防止内存泄漏
    return () => clearInterval(timer);
  }, []); // 依然保持空依赖数组，只挂载一次定时器

  return <div>计时器: {count}</div>;
}
```

这两段代码对比展示了闭包陷阱的破坏力，以及如何充分利用 `useRef` 的引用稳定性和 `useState` 的函数式更新机制，完美避开函数组件每次渲染产生独立闭包所带来的问题。

通过深入理解 Hooks 的链表结构，可以更好地编写和封装[[状态管理设计#常见状态管理模式|自定义Hook]]，提升代码的可维护性。
