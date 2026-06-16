Service Worker是一种运行在浏览器后台的独立脚本，充当Web应用程序与网络之间的代理服务器。它是构建渐进式Web应用(PWA)的核心技术，使得应用能够支持离线访问、后台同步以及推送通知。

## 核心特性
- **事件驱动**：不依赖于Web页面，只有在需要时才被唤醒，闲置时会休眠以节省资源。
- **网络拦截**：能够拦截由页面发起的网络请求（`fetch`事件），并根据自定义策略返回[[浏览器缓存机制|缓存]]数据或向网络发起真实请求。
- **HTTPS限制**：出于安全考虑，Service Worker只能在HTTPS环境下运行（本地开发`localhost`除外）。

## 生命周期
Service Worker的生命周期与页面完全独立，包含以下主要阶段：
1. **注册（Register）**：在主线程脚本中注册Service Worker文件。
2. **安装（Install）**：浏览器下载并解析脚本。此时通常会触发`install`事件，开发者可以在此阶段预缓存应用的核心静态资源。
3. **激活（Activate）**：安装成功后进入激活阶段。此时常用于清理旧版本的缓存数据。
4. **控制（Control）**：激活后，Service Worker开始接管其作用域下的所有页面，监听和处理各类事件。

## 缓存策略
借助Cache API，Service Worker可以实现多种灵活的缓存策略：
- **Cache First（缓存优先）**：优先从缓存读取数据，若无缓存则发起网络请求。适用于静态资源。
- **Network First（网络优先）**：优先发起网络请求，失败时回退到缓存。适用于频繁更新的动态数据。
- **Stale-While-Revalidate（缓存更新）**：立即返回缓存数据，同时在后台发起网络请求更新缓存，确保下次访问时数据最新。

```javascript
// sw.js 内部：拦截请求并应用 Cache First 策略
self.addEventListener('fetch', event => {
  event.respondWith(
    caches.match(event.request).then(response => {
      return response || fetch(event.request);
    })
  );
});
```