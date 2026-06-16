## Fetch API 简介

Fetch API 提供了一个全局的方法 `fetch()`，用于跨网络异步获取资源。相比于传统的 `XMLHttpRequest`，它使用了 Promise，使得处理异步操作更加简洁。在涉及到[[Web_Workers与多线程|多线程环境]]或[[Service_Worker与离线缓存|Service Worker]]时，Fetch API 是进行网络请求的标准方式。

## 基础用法与代码示例

`fetch()` 默认发起 GET 请求，并返回一个解析为 `Response` 对象的 Promise。

```javascript
// 发起基础 GET 请求
fetch('https://api.example.com/data')
  .then(response => {
    if (!response.ok) {
      throw new Error('网络请求错误');
    }
    return response.json();
  })
  .then(data => console.log(data))
  .catch(error => console.error('请求失败:', error));
```

## 进阶请求配置

可以传递第二个配置对象参数，用于指定请求方法、请求头、请求体等。

```javascript
// 发起 POST 请求
async function postData(url = '', data = {}) {
  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(data)
  });
  return response.json();
}
```
