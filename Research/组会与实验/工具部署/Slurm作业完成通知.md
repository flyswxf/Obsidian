## 需求

长任务提交后只能靠反复 `squeue` 判断进度，GPU 排队时间长时尤其低效。目标是在作业结束时自动收到手机通知，并能区分成功与失败。

## 架构

两条集群事实决定了实现方式：

| 集群事实 | 对方案的约束 |
| --- | --- |
| GPU 计算节点无外网 | 推送不能由作业自身发出，必须由登录节点代发 |
| 登录节点有外网，但 `crontab` 被禁用 | 只能用长驻后台进程，不能用定时任务 |

于是采用「登录节点常驻守护 + 飞书群机器人 Webhook」：

```text
login1 常驻守护进程（setsid，脱离 SSH 会话）
    │ 每 60 秒一轮
    ▼
sacct -u yhhao 取时间窗口内的终态作业
    │ 与已推送 JobID 列表去重
    ▼
curl POST 到飞书群机器人 Webhook
    ▼
手机飞书消息
```

消息内容：

```text
[Slurm] 作业完成
JobID   559598
作业名  notify_e2e_test
分区    cpu
状态    COMPLETED
退出码  0:0
耗时    00:00:25
节点    node106
结束    2026-11-20 13:54:19
```

成功（状态 `COMPLETED` 且退出码 `0:0`）时标题为「作业完成」，其余情况为「作业异常结束」。飞书机器人的安全设置使用「自定义关键词」，关键词填 `Slurm`，消息文本以 `[Slurm]` 开头即可通过校验。

## 部署结构

```text
/public/home/yhhao/wyf/slurm_notify/
├── watch_jobs.sh           守护进程主体
├── notify_feishu.sh        飞书文本推送封装
├── ctl.sh                  管理脚本
├── notify.conf             Webhook 地址，权限 600
├── watch.pid               运行中的进程号
├── state/seen_jobs.txt     已推送的 JobID
├── state/current_jobs.tsv  本轮 sacct 原始结果
└── logs/watch.log          运行日志
```

## 核心逻辑

作业状态查询：

```bash
sacct -u "$USER" \
  -S "$(date -d '-48 hours' '+%Y-%m-%dT%H:%M:%S')" -E now \
  --format=JobID,JobName%64,Partition,State,ExitCode,Elapsed,NodeList,End -P -n
```

过滤规则：

- 跳过 JobID 含 `.` 的行，即 `*.batch`、`*.extern` 等作业步骤行；
- 只保留终态：`COMPLETED`、`FAILED`、`CANCELLED`、`TIMEOUT`、`OUT_OF_MEMORY`、`NODE_FAIL`、`PREEMPTED`、`BOOT_FAIL`、`DEADLINE`、`REVOKED`、`SPECIAL_EXIT`；
- `PENDING`、`RUNNING` 一律忽略。

推送体为飞书文本消息：

```json
{"msg_type":"text","content":{"text":"[Slurm] 作业完成\n..."}}
```

去重与恢复：

- 首次启动把窗口内已有的终态作业直接登记进 `state/seen_jobs.txt`，不补推历史，避免一次性刷屏；
- 每轮结束后按当前窗口裁剪 `state/seen_jobs.txt`，文件不会无限增长；
- 守护进程重启时 `state/seen_jobs.txt` 保留，停机期间结束的作业会在下一轮补推一次。

## 使用

```bash
~/wyf/slurm_notify/ctl.sh status    # 进程、启动时间、已登记作业数、最近日志
~/wyf/slurm_notify/ctl.sh test      # 手动发一条测试消息，验证推送通道
~/wyf/slurm_notify/ctl.sh restart   # 重启
~/wyf/slurm_notify/ctl.sh stop      # 停止
~/wyf/slurm_notify/ctl.sh start     # 启动
```

可调环境变量，在 `ctl.sh start` 前 export 即可：

| 变量 | 默认值 | 作用 |
| --- | --- | --- |
| `SLURM_NOTIFY_INTERVAL` | 60 | 轮询间隔（秒），同时是通知延迟上限 |
| `SLURM_NOTIFY_LOOKBACK_HOURS` | 48 | `sacct` 回看窗口 |

## 边界与维护

- 登录节点重启或管理员清理进程后守护会消失，重新执行 `ctl.sh start` 即可恢复。
- 通知延迟等于轮询间隔，最长 60 秒。
- `notify.conf` 保存了 Webhook 地址，始终保持 600 权限，不要提交进仓库或写入笔记。
- 只在登录节点运行；计算节点无外网，把守护塞进 `sbatch` 不会生效。
