## 双引擎架构概述

Vite 采用双引擎架构以平衡开发体验与生产构建性能。在开发环境下使用 esbuild 进行依赖预构建，而在生产环境下采用 Rollup 进行打包。

## 开发环境：esbuild

esbuild 使用 Go 语言编写，编译速度极快。主要负责：
- **依赖预构建**：将 CommonJS 或 UMD 格式的依赖转换为 [[ESM与前端模块化|ESM]] 格式，合并大量内部模块的 [[ESM与前端模块化|ESM]] 依赖，减少浏览器网络请求数量。
- **TypeScript 与 JSX 编译**：仅执行语法转换，不进行类型检查，极大提升了热更新速度。

## 生产环境：Rollup

Rollup 是成熟的 [[ESM与前端模块化|ESM]] 打包工具，具备强大的生态和优化能力。生产环境选择 Rollup 的原因包括：
- **代码分割（Code Splitting）**：提供灵活的代码分割机制，优化首屏加载时间。
- **Tree Shaking**：高效消除未使用的代码。
- **丰富的插件生态**：通过灵活的插件 API 实现复杂的构建需求。为了保证开发和生产环境的一致性，Vite 实现了一套通用插件 API，使得大多数 Rollup 插件可以直接在 Vite 中使用。

```javascript
// vite.config.js 中配置不同引擎的参数示例
import { defineConfig } from 'vite';

export default defineConfig({
  // esbuild 配置（用于开发阶段）
  esbuild: {
    jsxFactory: 'h',
    jsxFragment: 'Fragment'
  },
  // Rollup 配置（用于生产构建阶段）
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['vue', 'vue-router'] // 手动代码分割
        }
      }
    }
  }
});
```
