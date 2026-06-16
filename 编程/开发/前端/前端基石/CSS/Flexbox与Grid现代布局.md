## Flexbox 弹性盒子布局

Flexbox（Flexible Box）是一维布局模型，专门用于处理行或列的对齐、空间分配及元素排序。它非常适合组件层级的布局。

### 核心概念

Flexbox 涉及两个主要实体：**容器（Flex Container）** 和 **项目（Flex Items）**。
布局基于两条轴线进行：**主轴（Main Axis）** 和 **交叉轴（Cross Axis）**。

### 常用容器属性

- `display: flex;`：将元素定义为 Flex 容器。
- `flex-direction`：决定主轴的方向（`row`、`column`、`row-reverse`、`column-reverse`）。
- `justify-content`：定义项目在主轴上的对齐方式（`flex-start`、`center`、`space-between`、`space-around` 等）。
- `align-items`：定义项目在交叉轴上的对齐方式（`stretch`、`center`、`flex-start`、`flex-end` 等）。
- `flex-wrap`：控制项目是否换行（`nowrap`、`wrap`、`wrap-reverse`）。

### 常用项目属性

- `flex-grow`：定义项目的放大比例，默认为 `0`。
- `flex-shrink`：定义项目的缩小比例，默认为 `1`。
- `flex-basis`：定义在分配多余空间之前，项目占据的主轴空间。
- 简写属性 `flex`：综合了 `grow`、`shrink` 和 `basis`，如 `flex: 1` 相当于 `flex: 1 1 0%`。

## Grid 网格布局

Grid 是二维布局系统，能够同时处理行和列，适用于构建复杂的页面整体骨架。

### 核心概念

- **网格容器（Grid Container）**：应用了 `display: grid` 的元素。
- **网格轨道（Grid Track）**：相邻网格线之间的空间，即行或列。
- **网格单元（Grid Cell）**：网格中最小的单位，由四条网格线包围。
- **网格区域（Grid Area）**：由多个网格单元组成的矩形区域。

### 常用容器属性

- `grid-template-columns` / `grid-template-rows`：定义列宽和行高。支持绝对单位（`px`）、百分比（`%`）以及弹性单位（`fr`）。
- `gap`（原 `grid-gap`）：定义网格行与列之间的间距。
- `grid-template-areas`：通过命名的网格区域来设计布局，非常直观。

### 常用项目属性

- `grid-column-start` / `grid-column-end`：指定项目横跨的网格线起始和结束位置。
- `grid-area`：可以作为指定网格区域名称的简写，配合 `grid-template-areas` 使用。

```css
/* Flexbox 示例：居中对齐导航栏 */
.nav-bar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0 20px;
}

/* Grid 示例：经典圣杯布局 */
.page-layout {
  display: grid;
  grid-template-columns: 200px 1fr 200px;
  grid-template-rows: 60px 1fr 100px;
  grid-template-areas:
    "header header header"
    "sidebar main rightbar"
    "footer footer footer";
  min-height: 100vh;
}

.header { grid-area: header; }
.sidebar { grid-area: sidebar; }
.main { grid-area: main; }
.rightbar { grid-area: rightbar; }
.footer { grid-area: footer; }
```

## Flexbox 与 Grid 的选择与结合

- 当关注单个维度的排列（通常是局部组件，如按钮组、导航菜单）时，优先选择 **Flexbox**。
- 当需要规划页面的宏观结构，且需要同时控制行和列时，优先选择 **Grid**。
- 现代前端开发通常将两者结合使用：用 Grid 搭建页面的整体骨架，用 Flexbox 填充骨架内部的细节布局。