**完整语法格式：**
```cpp
[capture](parameters) -> return_type { function_body }
```

**例子：**

```cpp
sort(points.begin(),points.end(),[](const vector<int>& a,const vector<int>& b){
    return a[0] < b[0];
});
```
- `[]` - **捕获列表**：空的方括号表示不捕获任何外部变量
- `(const vector<int>& a, const vector<int>& b)` - **参数列表**：接收两个常量引用参数
- `{ return a[0] < b[0]; }` - **函数体**：比较逻辑，返回第一个点的x坐标是否小于第二个点的x坐标

## 等价的传统写法

这个lambda表达式等价于定义一个比较函数：
```cpp
bool compare(const vector<int>& a, const vector<int>& b) {
    return a[0] < b[0];
}
// 然后使用：sort(points.begin(), points.end(), compare);
```

Lambda表达式让代码更简洁，避免了定义额外的函数。
        