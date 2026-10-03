## 适用场景

实验主机无法稳定访问 GitHub，但本地电脑可以通过 SSH 登录实验主机。

代码更新链路改为：

```text
本地工作仓库 → SSH → 实验主机裸仓库 → 实验主机工作目录
```

GitHub 仅作为备份仓库，不再参与日常部署。

## 当前配置

| 项目     | 配置                                                           |
| ------ | ------------------------------------------------------------ |
| SSH 节点 | `yhhao@59.78.189.140:22`                                     |
| 本地仓库   | `Physical-Attention-Attack`                                  |
| 远端裸仓库  | `/public/home/yhhao/wyf/repos/Physical-Attention-Attack.git` |
| 远端工作目录 | `/public/home/yhhao/wyf/Physical-Attention-Attack`           |
| 默认分支   | `main`                                                       |

<br />

本地仓库包含两个 remote：

```text
origin  → GitHub
lab     → 实验主机裸仓库
```

检查配置：

```powershell
git -C "D:\desktop\code\Physical-Attention-Attack" remote -v
```

## 工作原理

裸仓库只保存 Git 提交和分支，不直接编辑文件：

```text
/public/home/yhhao/wyf/repos/Physical-Attention-Attack.git
```

实验代码在工作目录中运行：

```text
/public/home/yhhao/wyf/Physical-Attention-Attack
```

本地将提交推送到裸仓库，实验主机的工作目录再从裸仓库拉取更新。整个过程不经过 GitHub。

## 首次配置

### 1. 在实验主机创建裸仓库

```bash
ssh yhhao@59.78.189.140
mkdir -p /public/home/yhhao/wyf/repos
git init --bare /public/home/yhhao/wyf/repos/Physical-Attention-Attack.git
```

### 2. 为本地仓库添加实验主机 remote

在本地 PowerShell 中执行：

```powershell
git -C "D:\desktop\code\Physical-Attention-Attack" remote add lab "ssh://yhhao@59.78.189.140/public/home/yhhao/wyf/repos/Physical-Attention-Attack.git"
```

如果 `lab` 已存在但地址错误：

```powershell
git -C "D:\desktop\code\Physical-Attention-Attack" remote set-url lab "ssh://yhhao@59.78.189.140/public/home/yhhao/wyf/repos/Physical-Attention-Attack.git"
```

### 3. 首次推送

```powershell
git -C "D:\desktop\code\Physical-Attention-Attack" push lab main
```

### 4. 设置裸仓库默认分支

旧版 Git 创建裸仓库时可能仍以 `master` 为默认分支。首次克隆前执行：

```bash
git --git-dir=/public/home/yhhao/wyf/repos/Physical-Attention-Attack.git \
  symbolic-ref HEAD refs/heads/main
```

### 5. 在实验主机创建工作目录

```bash
git clone /public/home/yhhao/wyf/repos/Physical-Attention-Attack.git \
  /public/home/yhhao/wyf/Physical-Attention-Attack
```

## 日常更新

### 本地提交并推送

```powershell
cd "D:\desktop\code\Physical-Attention-Attack"
git status
git add <需要提交的文件>
git commit -m "说明本次修改"
git push lab main
```

`git push lab main` 推送到实验主机，`git push origin main` 推送到 GitHub。

### 实验主机拉取代码

```bash
ssh yhhao@59.78.189.140
cd /public/home/yhhao/wyf/Physical-Attention-Attack
git status
git pull
```

拉取前先运行 `git status`。如果实验主机上存在未提交修改，先提交、暂存或备份，避免覆盖实验代码。

## 换电脑

Git remote 保存在本地仓库的 `.git/config` 中，不会随普通 Git 提交上传。SSH 配置和密钥也不会上传。

新电脑可以直接从实验主机克隆：

```powershell
git clone "ssh://yhhao@59.78.189.140/public/home/yhhao/wyf/repos/Physical-Attention-Attack.git"
cd Physical-Attention-Attack
git remote rename origin lab
git remote add origin "https://github.com/flyswxf/Physical-Attention-Attack.git"
git remote -v
```

此时：

```text
lab     → 实验主机
origin  → GitHub
```

如果新电脑已经从 GitHub 克隆过仓库，只需补充 `lab`：

```powershell
git remote add lab "ssh://yhhao@59.78.189.140/public/home/yhhao/wyf/repos/Physical-Attention-Attack.git"
git fetch lab
```

## SSH 简化配置

在新电脑的 `~/.ssh/config` 中添加：

```sshconfig
Host 华师大集群
  HostName 59.78.189.140
  Port 22
  User yhhao
```

之后可以简化登录命令：

```bash
ssh 华师大集群
```

Git remote 也可以写成：

```powershell
git remote set-url lab "华师大集群:/public/home/yhhao/wyf/repos/Physical-Attention-Attack.git"
```

建议进一步配置 SSH 公钥登录，避免每次输入密码。私钥和密码不能提交到 Git 仓库。

## 常见问题

### 检查 SSH 端口

```powershell
Test-NetConnection 59.78.189.140 -Port 22
```

若 `TcpTestSucceeded` 为 `False`，检查校园网、VPN、节点状态和防火墙。

### 检查远端地址

```powershell
git remote -v
```

### `remote lab already exists`

说明已经配置过 `lab`，无需重复添加。需要修改地址时使用：

```powershell
git remote set-url lab "ssh://yhhao@59.78.189.140/public/home/yhhao/wyf/repos/Physical-Attention-Attack.git"
```

### 实验主机无法拉取最新提交

依次检查：

```powershell
git push lab main
```

```bash
cd /public/home/yhhao/wyf/Physical-Attention-Attack
git remote -v
git status
git pull
```

### 查看两端提交是否一致

本地：

```powershell
git rev-parse HEAD
```

实验主机：

```bash
cd /public/home/yhhao/wyf/Physical-Attention-Attack
git rev-parse HEAD
```

两个提交哈希一致，说明代码已经同步。
