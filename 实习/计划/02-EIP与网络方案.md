## 瓶颈
当前最关键的物理瓶颈是 EIP 不够。

### 当前ssh封装
当前默认假设测试框架底层是 `paramiko` 一类的 SSH 长连接封装：
  - 用例里先实例化一个 `client`
  - 初始化时传入 IP、用户名、密码、端口等参数
  - 后续通过 `client.exec(cmd)` 执行远程命令
  - 这种模式通常说明底层不是临时拼 `ssh` 命令，而是先建连接，再复用连接
```python
class Client:
    def __init__(self, ip, username, password):
        self.ip = ip
        self.username = username
        self.password = password
        self.conn = SSHConnection(ip, username, password)

    def exec(self, cmd):
        return self.conn.exec(cmd)
```

在这个前提下，跳板机方案需要在代码里显式做中转：
  - 只保留 1 个固定 EIP 给跳板 ECS
  - 文件系统的 4 个节点只保留内网 IP
  - 本地办公机或堡垒机先访问跳板 ECS 的 EIP
  - 再由跳板 ECS 去访问 MDS / Space 节点的内网 IP

实际链路是：
**本地机器 -> 跳板机 EIP -> 节点内网 IP:22**
### 价值
  - 每个文件系统不再单独占用 4 个 EIP
  - 多套文件系统可以共用 1 台跳板机
  - 配置文件里维护的重点从公网 IP 变成内网 IP
  - 资源调度会轻很多
  - 后续自动化申请环境时，不需要频繁拆别的环境上的 EIP

## 预期更改
`paramiko` 下典型的跳板机连接流程：
  1. 创建跳板机 `SSHClient`
  2. 用跳板机 EIP、账号、密码建立连接
  3. 通过跳板机 `transport.open_channel("direct-tcpip", ...)` 打到目标节点内网 IP
  4. 创建目标节点 `SSHClient`
  5. 调用目标节点 `connect(..., sock=channel)`
  6. 后续所有 `exec_command` 都在目标节点这条连接上执行

```python
jump_client.connect(jump_host, username=..., password=...)
# 现在从这条 SSH 连接里拿到底层传输对象
transport = jump_client.get_transport()
channel = transport.open_channel(
    "direct-tcpip",
    dest_addr=(target_private_ip, 22),
    src_addr=("127.0.0.1", 0),
)

target_client.connect(
    hostname=target_private_ip,
    username=...,
    password=...,
    sock=channel,
)
```

可以把它理解成：
  - 以前：本地代码直接连 `目标公网 IP`
  - 现在：本地代码先连 `跳板机公网 IP`，再借这条连接穿到 `目标内网 IP`
  - 所以上层仍然像是在“直接操作目标节点”，只是底层路径换了

## 对配置文件的影响
全局区域：
- 跳板机公网 IP
- 跳板机登录信息
- 是否启用跳板模式
节点区域：
- MDS1 内网 IP
- MDS2 内网 IP
- Space1 内网 IP
- Space2 内网 IP
- 各节点登录信息

底层代码建议支持两种模式：
**直连模式**：
- 直接 `connect(hostname=node_ip, ...)`
**跳板机模式**：
- 先连 jump host
- 再 `open_channel`
- 再通过 `sock=channel` 连目标内网 IP

这样做的好处是：
  - 老环境如果还有 EIP，可以继续用直连模式
  - 新环境可以切到跳板机模式
  - 改造风险更小，便于灰度切换
