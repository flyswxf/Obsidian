## 块级作用域变量声明

ES6 引入了 `let` 和 `const`，取代了传统的 `var` 声明，有效解决了变量提升和作用域泄漏的问题。

- `let`：声明块级作用域的局部变量，允许重新赋值。
- `const`：声明块级作用域的常量，声明时必须初始化，且引用地址不可被修改（但对象内部属性可变）。

```javascript
{
  let a = 10;
  const b = 20;
}
// console.log(a); // ReferenceError
```

## 箭头函数（Arrow Functions）

箭头函数提供了一种更简洁的函数书写方式，并解决了 `this` 指向的困扰。

- 语法简洁：省略 `function` 关键字，单行返回可省略大括号和 `return`。
- `this` 绑定：箭头函数没有自己的 `this`，它会捕获其所在上下文的 `this` 值。

```javascript
const add = (a, b) => a + b;
const doubleArray = arr => arr.map(x => x * 2);
```

## 解构赋值（Destructuring）

允许从数组或对象中提取数据，并赋值给新的变量。

### 对象解构

```javascript
const user = { name: 'Alice', age: 25, city: 'New York' };
const { name, age, country = 'USA' } = user; // 支持默认值
```

### 数组解构

```javascript
const numbers = [1, 2, 3, 4];
const [first, second, ...rest] = numbers; // 结合剩余参数
```

## 扩展运算符与剩余参数（Spread & Rest）

`...` 语法在不同上下文中具有不同的作用。

- **扩展运算符**：展开数组或对象。
- **剩余参数**：收集多余的参数组成数组。

```javascript
// 扩展运算符
const arr1 = [1, 2];
const arr2 = [...arr1, 3, 4]; 
const obj1 = { a: 1 };
const obj2 = { ...obj1, b: 2 };

// 剩余参数
function sum(...args) {
  return args.reduce((total, current) => total + current, 0);
}
```

## 模板字符串（Template Literals）

使用反引号（`` ` ``）标识，支持多行字符串和内嵌表达式。

```javascript
const name = 'Bob';
const greeting = `Hello, ${name}! 
Welcome to the modern JavaScript era.`;
```

## 模块化（ES Modules）

ES6 标准化了 JavaScript 的模块系统，使用 `import` 和 `export` 进行模块的导入和导出。

```javascript
// math.js
export const pi = 3.14;
export function add(a, b) { return a + b; }
export default function multiply(a, b) { return a * b; }

// main.js
import multiply, { pi, add } from './math.js';
```

## 可选链与空值合并（ES2020）

- **可选链（`?.`）**：安全地访问深层嵌套的对象属性，若属性不存在则返回 `undefined`，避免抛出错误。
- **空值合并运算符（`??`）**：仅在左侧操作数为 `null` 或 `undefined` 时返回右侧操作数（不同于 `||` 会在遇到 `0` 或 `''` 时也返回右侧值）。

```javascript
const user = { profile: { email: 'test@test.com' } };
const email = user?.profile?.email;
const age = user?.profile?.age ?? 18;
```