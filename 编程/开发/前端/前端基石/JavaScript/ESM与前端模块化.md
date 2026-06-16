## 前端模块化演进

在早期的 JavaScript 中，并没有原生的模块化机制，所有的代码通常通过 `<script>` 标签引入，容易造成全局变量污染和依赖关系混乱。随着前端工程化的发展，社区先后提出了 CommonJS（主要用于 Node.js）、AMD、CMD 等规范。直到 ES6，JavaScript 才在语言标准层面上引入了原生的模块化方案：ESM（ECMAScript Modules）。

## ESM 核心特性

ESM 是目前现代前端开发的基础标准，具有以下核心特征：

1. **静态解析**：ESM 的 `import` 和 `export` 语句必须位于模块的顶层，不能嵌套在条件语句或函数中。这使得在编译时就能确定模块的依赖关系，从而为构建工具（如 Rollup、esbuild 等）进行静态分析和 Tree Shaking（摇树优化）提供了可能。
2. **异步加载**：在浏览器环境中，使用 `<script type="module">` 引入的模块默认会以 `defer` 的方式异步加载，不会阻塞 HTML 树的解析。
3. **独立作用域**：每个模块都有自己独立的作用域，模块内部声明的变量对外部不可见，除非显式导出。
4. **强绑定与只读**：`import` 导入的模块绑定是只读的引用，不能重新赋值（但如果导出的是对象，可以修改其属性）。

## 基础语法

### 导出 (Export)

可以通过命名导出或默认导出将模块内部的变量、函数或类暴露给外部：

```javascript
// 命名导出（可导出多个）
export const apiEndpoint = "https://api.example.com";
export function fetchData() { /* ... */ }

// 默认导出（每个模块只能有一个）
export default class UserService { /* ... */ }
```

### 导入 (Import)

对应的导入语法：

```javascript
// 导入命名导出的内容（需使用同名，或使用 as 重命名）
import { apiEndpoint, fetchData as fetch } from './api.js';

// 导入默认导出的内容（可以任意命名）
import UserSvc from './user.js';

// 混合导入
import UserSvc, { apiEndpoint } from './module.js';

// 整体导入为一个命名空间对象
import * as ApiModule from './api.js';
```

## 与 CommonJS 的对比

- **语法**：CommonJS 使用 `require` 和 `module.exports`；ESM 使用 `import` 和 `export`。
- **加载方式**：CommonJS 是同步加载模块，主要用于服务器端（本地文件系统读取速度快）；ESM 支持异步加载，原生支持浏览器环境。
- **解析时机**：CommonJS 是运行时加载，依赖关系在代码执行时确定；ESM 是编译时输出接口，静态解析。
- **值的传递**：CommonJS 导出的是值的拷贝（缓存）；ESM 导出的是值的引用（实时绑定）。