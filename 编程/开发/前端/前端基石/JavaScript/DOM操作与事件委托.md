## DOM 操作基础

文档对象模型（Document Object Model, DOM）是 HTML 和 XML 文档的编程接口。它将页面表示为一个节点树，开发者可以通过 JavaScript 操作这些节点，从而动态改变文档的结构、样式和内容。

### 获取 DOM 元素

现代开发中，推荐使用通用性更强的选择器方法：

- `document.querySelector(selector)`：返回匹配指定 CSS 选择器的第一个元素。
- `document.querySelectorAll(selector)`：返回一个包含所有匹配元素的静态 NodeList。

### 操作节点

- **创建节点**：`document.createElement(tagName)`。
- **插入节点**：`parentNode.appendChild(child)` 或 `parentNode.insertBefore(newNode, referenceNode)`。现代 API 提供更便捷的 `element.append()` 和 `element.prepend()`。
- **删除节点**：`element.remove()`（现代 API）或 `parentNode.removeChild(child)`。

### 修改属性与类名

- **属性操作**：`element.getAttribute(name)`、`element.setAttribute(name, value)`。
- **类名操作**：推荐使用 `classList` API（`add`、`remove`、`toggle`、`contains`），比直接操作 `className` 字符串更安全便捷。

```javascript
const btn = document.querySelector('.submit-btn');
btn.classList.add('active');
btn.setAttribute('disabled', 'true');
```

## 事件处理

事件是用户或浏览器与页面交互时发生的动作。

### 绑定事件

推荐使用 `addEventListener`，它允许为一个元素绑定多个同类型事件，并可精细控制事件流阶段。

```javascript
const button = document.querySelector('#myBtn');
button.addEventListener('click', (event) => {
  console.log('Button clicked!', event.target);
});
```

## 事件流与事件委托

### 事件流机制

当一个事件发生时，它在 DOM 树中的传播分为三个阶段：
1. **捕获阶段**：事件从 `window` 逐级向下传播到目标元素。
2. **目标阶段**：事件到达触发该事件的最底层元素。
3. **冒泡阶段**：事件从目标元素逐级向上冒泡回 `window`。默认情况下，事件处理程序在冒泡阶段触发。

### 事件委托（Event Delegation）

事件委托利用了事件冒泡的原理。其核心思想是：**不在大量子元素上分别绑定事件监听器，而是将一个共享的监听器绑定在它们共同的父级元素上。**

#### 优势
- **减少内存消耗**：只需绑定一个监听器，大幅减少 DOM 引用和内存占用。
- **动态元素支持**：对于后续通过 JavaScript 动态添加到父元素内部的新子元素，无需重新绑定事件，委托依然有效。

#### 实现方式

在父元素的事件处理函数中，通过检查 `event.target`（实际触发事件的最底层元素）来决定是否执行相应逻辑。可以使用 `element.closest()` 方法来精确定位目标。

```javascript
// HTML 结构
// <ul id="todo-list">
//   <li class="todo-item" data-id="1">任务 1 <button class="delete-btn">删除</button></li>
//   <li class="todo-item" data-id="2">任务 2 <button class="delete-btn">删除</button></li>
// </ul>

const list = document.querySelector('#todo-list');

list.addEventListener('click', (event) => {
  // 检查点击的是否是删除按钮
  if (event.target.classList.contains('delete-btn')) {
    // 向上查找最近的 li 元素
    const listItem = event.target.closest('.todo-item');
    if (listItem) {
      const id = listItem.dataset.id;
      console.log(`准备删除任务 ID: ${id}`);
      listItem.remove(); // 执行删除操作
    }
  }
});
```