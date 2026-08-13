只调整两个文件, 代码行数均100+, 不是很多

# 具体替换规则
## `ms_deform_attn.h`：
- `value.type().is_cuda()` 改成 `value.is_cuda()`
## `ms_deform_attn_cuda.cu`：
- `xxx.type().is_cuda()` 改成 `xxx.is_cuda()`
- `AT_DISPATCH_FLOATING_TYPES(value.type(), ...)` 改成 `AT_DISPATCH_FLOATING_TYPES(value.scalar_type(), ...)`
- `.data<T>()` 改成 `.data_ptr<T>()`

```cpp
// 旧
value.type().is_cuda()
// 新
value.is_cuda()
```

```cpp
// 旧
AT_DISPATCH_FLOATING_TYPES(value.type(), "xxx", ([&] { ... }))
// 新
AT_DISPATCH_FLOATING_TYPES(value.scalar_type(), "xxx", ([&] { ... }))
```

```cpp
// 旧
tensor.data<scalar_t>()
tensor.data<int64_t>()
// 新
tensor.data_ptr<scalar_t>()
tensor.data_ptr<int64_t>()
```