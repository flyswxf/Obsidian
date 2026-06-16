## Vite 核心概念

Vite 是一个现代化的前端构建工具，主要由两部分组成：基于原生 ES 模块的开发服务器，以及基于 Rollup 的生产环境构建指令。它在开发环境下的快速启动得益于[[ESBuild与Rollup双引擎机制#ESBuild在开发阶段的作用|ESBuild预构建]]。

## 基础配置示例

Vite 的配置文件通常命名为 `vite.config.js` 或 `vite.config.ts`。以下是一个包含基础插件和别名配置的示例：

```typescript
import { defineConfig } from 'vite';
import vue from '@vitejs/plugin-vue';
import path from 'path';

export default defineConfig({
  plugins: [vue()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src')
    }
  },
  server: {
    port: 3000,
    open: true,
    proxy: {
      '/api': {
        target: 'http://localhost:8080',
        changeOrigin: true
      }
    }
  }
});
```

配置好开发服务器后，可以利用[[HMR热更新原理|模块热替换]]极大地提升开发体验。
