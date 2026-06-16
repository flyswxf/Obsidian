## 层叠上下文 (Stacking Context)

层叠上下文是 HTML 元素的三维概念，这些 HTML 元素在一条假想的相对于面向（电脑屏幕的）视窗的 Z 轴上延伸。HTML 元素依据其自身属性按照一定的优先级顺序占用层叠上下文的空间。

### 触发条件

常见触发层叠上下文的条件包括：
- 根元素 `<html>`
- `position` 值为 `absolute` 或 `relative` 且 `z-index` 值不为 `auto` 的元素
- `position` 值为 `fixed` 或 `sticky` 的元素
- `flex` 容器的子元素，且 `z-index` 值不为 `auto`
- `opacity` 属性值小于 1 的元素
- `transform`、`filter`、`perspective` 等属性值不为 `none` 的元素

### 层叠顺序

在一个层叠上下文中，元素的层叠顺序从底层到顶层依次为：
1. 层叠上下文的背景和边框
2. `z-index` 为负值的子层叠上下文
3. 常规流内的块级元素
4. 浮动元素
5. 常规流内的行内元素
6. `z-index` 为 0 的子元素或属性 `z-index` 为 `auto` 的定位元素
7. `z-index` 为正值的子层叠上下文

## BFC (Block Formatting Context)

块格式化上下文（BFC）是 Web 页面的可视 CSS 渲染的一部分，是块级盒子的布局过程发生的区域，也是浮动元素与其他元素交互的区域。可以把它看作是一个独立的渲染区域，内部元素的布局不会影响到外部元素。

### 触发条件

创建 BFC 的常见方式：
- 根元素 `<html>`
- 浮动元素（`float` 值不为 `none`）
- 绝对定位元素（`position` 值为 `absolute` 或 `fixed`）
- 行内块元素（`display` 值为 `inline-block`）
- 表格单元格（`display` 值为 `table-cell`）
- `overflow` 值不为 `visible` 或 `clip` 的块级元素
- 弹性元素（`display` 为 `flex` 或 `inline-flex` 元素的直接子元素）
- 网格元素（`display` 为 `grid` 或 `inline-grid` 元素的直接子元素）

### BFC 的特性与应用

- **清除浮动**：BFC 区域在计算高度时，会包含内部的浮动元素。常用于解决父元素高度塌陷问题。
- **防止外边距折叠**：属于同一个 BFC 的两个相邻盒子的垂直外边距会发生折叠。若要避免折叠，可以将它们放在不同的 BFC 中。
- **避免文字环绕**：BFC 区域不会与浮动元素的区域重叠，常用于实现多列自适应布局。
