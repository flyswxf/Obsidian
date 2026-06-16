JavaScript在浏览器中是单线程执行的，这意味着复杂的计算会阻塞主线程，导致页面卡顿。为了解决这一问题，引入了Web Workers，它允许在后台线程中运行脚本，实现真正的并发执行，而不会干扰用户界面的交互。

## 核心特性

- **独立线程**：Web Worker在一个独立的作用域中运行，拥有自己的[[事件循环与任务队列|事件循环]]。由于它与主线程隔离，Worker线程中无法访问DOM、BOM（如`window`、`document`），但可以使用部分Web API，如`setTimeout`、`XMLHttpRequest`、`fetch`以及`IndexedDB`。

## 通信机制
主线程与Worker线程之间通过消息传递进行通信，核心是`postMessage`方法和`message`事件。
数据在传递时会被序列化和反序列化，这意味着传递的是数据的副本而非引用。对于大量数据，可以使用`SharedArrayBuffer`或Transferable Objects（如`ArrayBuffer`）以零拷贝的方式转移数据所有权，从而大幅提升性能。

```javascript
// 主线程 main.js
const worker = new Worker('worker.js');

// 向 Worker 发送数据
worker.postMessage({ type: 'CALCULATE', payload: 100 });

// 接收 Worker 的处理结果
worker.onmessage = function(event) {
  console.log('来自 Worker 的结果:', event.data);
};

// worker.js
self.onmessage = function(event) {
  if (event.data.type === 'CALCULATE') {
    const result = event.data.payload * 2; // 执行繁重计算
    self.postMessage(result); // 返回结果
  }
};
```

## 常见类型
- **Dedicated Worker**：专用Worker，仅能被创建它的脚本所使用。
- **Shared Worker**：共享Worker，可以被同源下的多个不同上下文（如多个标签页、iframe）共同访问和通信。

## 生命周期与资源管理
Worker的启动和运行会消耗系统资源。当后台任务完成时，应当及时终止Worker。可以在主线程中调用`worker.terminate()`，或在Worker内部调用`close()`来释放资源。