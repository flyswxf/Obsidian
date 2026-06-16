## 代码规范协同的意义

ESLint 主要用于代码质量检查（如未使用的变量、潜在的错误），而 Prettier 专注于代码的格式化（如缩进、引号类型）。将二者结合使用可以保证项目代码既符合逻辑规范，又具备统一的视觉风格。若项目有特殊需求，可以进一步进行[[自定义规则编写|ESLint规则定制]]。

## 解决规则冲突

为避免 ESLint 的格式规则与 Prettier 发生冲突，通常需要引入 `eslint-config-prettier` 来关闭 ESLint 中所有与格式相关的规则，并使用 `eslint-plugin-prettier` 将 Prettier 作为 ESLint 的规则运行。

```json
// .eslintrc.json 配置示例
{
  "extends": [
    "eslint:recommended",
    "plugin:@typescript-eslint/recommended",
    "prettier"
  ],
  "plugins": ["@typescript-eslint", "prettier"],
  "rules": {
    "prettier/prettier": "error",
    "no-console": "warn"
  }
}
```

## 自动化格式化流程

在项目中，可以通过配置 Git Hooks（如 Husky 和 lint-staged）在代码提交前自动执行检查和格式化，确保不符合规范的代码不会进入代码库。底层原理与[[AST抽象语法树解析|抽象语法树]]的代码分析机制密切相关。
