---
状态: 第 1 轮已完成
日期: 2026-11-26
研究线: "[[注意力可视化]]"
基础: "[[注意力可视化审核-生成式VLM新方法]]"
---

## 第 1 轮（2026-11-26）：集群 A100 忠实复现与头过滤方向核验

### 0 摘要

本轮在 centos7-lab 用 1×A100-40GB（Slurm 作业 574061）忠实复现官方仓 `WalBouss/DEX-AR`：装载其 `DexarWrapper`，对 LLaVA-1.5-7B（`llava-hf/llava-1.5-7b-hf`，revision `a272c74`）跑单图 dry-run（官方样例 `assets/cat_and_dog.jpg`）与 8 图批量（PascalVOC2012，8 个类，逐类取一个目标框）。主结果：动态头过滤（filtered）在 8/8 图上使最亮 10% 像素质量占比升高（均值 0.2242 对 0.1157，约 1.9×）、归一化熵降低（0.9801 对 0.9979），方向与原文「头过滤提升定位性」一致 #实测 `/public/home/yhhao/wyf/DEX-AR-runs/20261126-103959/metrics.json`。目标框内质量占比 filtered 亦在 8/8 图高于 unfiltered（0.4397 对 0.3745），但单峰落框命中率反而略低（0.625 对 0.75；n=8，不具统计显著性，仅方向性证据）。开销：纯前向 0.154 s/次，DEX-AR 单图 1.15–1.41 s（热态），约 8× 纯前向，量级与目标句 token 数（8）同阶 #实测 同上。

### 1 目标与范围

- 目标：复现 [[注意力可视化]] 规划表第 36 行 DEX-AR，核验其原文核心结论（动态头过滤提升视觉定位性）在本集群环境是否成立，并量化其相对纯前向注意力的开销 #文献 原文 Zotero key 6U2FFLUA。
- 范围：单图 dry-run + 8 图批量，落盘 filtered／unfiltered 句级热图、逐 token 热图与 `metrics.json`。
- 与规划的两处执行期决定（不回填规划正文）：
  1. 本地为 8 GB 显存，装不下 LLaVA-1.5-7B fp16；改集群 A100（用户确认「集群 A100，忠实复现」）。
  2. 原文定量表（ImageNet／VQAv2 扰动、PascalVOC 分割、filler 过滤）依赖官方评测代码，而官方仓 `WalBouss/DEX-AR` 只含 `dexar/wrapper.py` 与 `playground.py`，无评测脚本 #代码事实 `D:\code\DEX-AR\README.md`、仓内无 eval 目录。故本轮以「头过滤消融 + 目标框定位」为代理口径，不声称复算原文数字。

### 2 被复现方法与评测协议

方法内核（逐生成 token）：(1) 对中间层 logits（logit lens）关于注意力图求梯度；(2) 动态头过滤，按「对图像列 vs 对文本列的关注差」加权 $w_{l,t,i}=(S_{\text{img}}-S_{\text{text}})^{+}$（式 5）；(3) 以视觉相关性权重 $\delta_t$ 聚合为句级热图（式 6）#文献 Zotero key 6U2FFLUA。实现入口为 `dexar/wrapper.py` 的 `compute_dexar` #代码事实 `D:\code\DEX-AR\dexar\wrapper.py`：图像 token 网格 $576=24\times24$（第 57 行），主循环逐 token 取 `grad[:, :, -1, idx_img_tok_start:idx_img_tok_end]`（第 199 行），头过滤 `k_img=1, k_all_text=1`（max 口径，对应原文 Table 3 最优） #代码事实 同上。

评测协议（本轮代理口径）：

- 提示模板固定为 `USER: <image>\nDescribe the image. ASSISTANT:`（官方 playground 的 Describe 变体）；目标句 `This is a photo of a <class>.`。
- 每图产出 filtered（启用头过滤）与 unfiltered（关闭头过滤）两张句级热图，逐像素对比。
- 聚焦度指标：`top10_mass`＝最亮 10% 像素占总质量比例（越高越聚焦）；`entropy_norm`＝归一化香农熵（越低越聚焦） #代码事实 `D:\code\DEX-AR\repro_dexar.py` `focus_metrics`。
- 定位指标（对 VOC 目标框，坐标归一化到 $[0,1]$）：`peak_in_bbox`＝热图峰值是否落框内；`mass_in_bbox`＝框内质量占比 #代码事实 `repro_dexar.py` `bbox_metrics`。
- 开销指标：`profile_forward` 对同一输入关注意力／隐状态做 3 次纯前向计时取均值 #代码事实 `repro_dexar.py`。

选层口径：`layer_index=0`，与官方 `playground.py` 缺省一致 #代码事实 `D:\code\DEX-AR\playground.py`。

### 3 数据与完整性核查

| 项           | 取值                                                                                                                                                                      | 依据                                                                 |
| ----------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| 模型权重        | `llava-hf/llava-1.5-7b-hf` revision `a272c74`，3 分片，fp16                                                                                                                 | #实测 `/public/home/yhhao/wyf/DEX-AR/logs/slurm-574061.err`（加载 3 分片） |
| 批量图像        | PascalVOC2012，8 类各 1 张：dog 2008_000019／cat 2008_000060／bicycle 2008_000036／aeroplane 2008_000033／horse 2008_000008／bus 2008_000132／bottle 2008_000015／chair 2008_000041 | #实测 `/public/home/yhhao/wyf/DEX-AR-runs/voc_manifest.json`         |
| 目标框来源       | 各类 `Annotations/<id>.xml` 中该类首个 object 的 bndbox，除以原图宽高归一化                                                                                                               | #代码事实 `D:\code\DEX-AR\tools\make_voc_manifest.py`                  |
| 图像预处理       | 统一 resize 到 $336\times336$（与官方 playground 一致）                                                                                                                           | #代码事实 `repro_dexar.py`；`playground.py`                             |
| 图像 token 网格 | 576＝$24\times24$；`seq_len=593, first_image_idx=5, wrapper_start=5` 三者边界自洽                                                                                               | #实测 `slurm-574061.out` `[layout]` 行                                |

### 4 实验设置

| 项 | 取值 |
|---|---|
| 分区／资源 | `gpu`，`--gres=gpu:1`，1 节点，4 CPU，48 GB，`-t 04:00:00`；实得 A100-PCIE-40GB | 
| 环境 | conda `dexar`：py 3.10、torch 2.5.1+cu121、transformers 4.45.0（按 requirements.txt 独立新建） | 
| 模型 | LLaVA-1.5-7B fp16，`attn_implementation="eager"`（官方 `wrapper.py` 硬编码） | 
| 固定超参 | `layer_index=0`、`k_img=1, k_all_text=1`（max 头过滤）、`do_sample=False` |
| 自变量 | filtered（头过滤开）对 unfiltered（头过滤关） |
| 复现脚本 | `D:\code\DEX-AR\repro_dexar.py`（含单图/批量/画像三种入口） |
| 作业脚本 | `D:\code\DEX-AR\jobs\repro.sbatch`（`MANIFEST` 或 `IMAGES` 经 `--export` 传入） |

依据：#实测 `slurm-574061.err`（GPU 型号）、`/public/home/yhhao/wyf/DEX-AR/jobs/repro.sbatch`；#代码事实 `dexar/wrapper.py` 第 77–84 行。

### 5 结果

#### 5.1 定性（对照图）

单图 dry-run（cat_and_dog）：filtered 热图集中于主体，unfiltered 明显弥散，方向与批量一致 #实测 `D:\Obsidian\Research\组会与实验\附件\图片\DEX-AR\dexar-dog-filtered.png`、`dexar-dog-unfiltered.png`。

![[dexar-dog-filtered.png]]

![[dexar-dog-unfiltered.png]]

批量 8 图 filtered 句级热图（按类命名，见 8 节目录）：

![[dexar-cat-filtered.png]]

![[dexar-horse-filtered.png]]

![[dexar-bottle-filtered.png]]

![[dexar-chair-filtered.png]]

逐 token 热图（dog，filtered）显示权重集中于内容词：`token_weights` 中 `dog`=3.34、`.`=0.012，其余虚词 `This/is/a/photo/of/a` 全为 0，与式 6 的「按视觉相关性加权、滤掉虚词」设计一致 #实测 `metrics.json` → `records[0].token_weights`。

![[dexar-dog-per-token-filtered.png]]

#### 5.2 定量（n=8）

| 指标 | filtered | unfiltered | 方向 |
|---|---|---|---|
| 均值 `top10_mass` | 0.2242 | 0.1157 | filtered 高（≈1.9×） |
| 均值 `entropy_norm` | 0.9801 | 0.9979 | filtered 低（更聚焦） |
| `top10_mass` 逐图胜出比例 | 8/8 = 1.0 | — | filtered 全胜 |
| 均值 `mass_in_bbox` | 0.4397 | 0.3745 | filtered 高，8/8 图胜出 |
| `peak_in_bbox` 命中率 | 5/8 = 0.625 | 6/8 = 0.75 | unfiltered 略高 |
| 单图计算耗时（热态，均值） | 1.25 s | — | 纯前向 0.154 s，约 8× |

#实测 `/public/home/yhhao/wyf/DEX-AR-runs/20261126-103959/metrics.json`；对照图 `D:\Obsidian\Research\组会与实验\附件\图片\DEX-AR\`。n=8，不具统计显著性，仅作方向性证据。

#### 5.3 与原文结论对照

| 来源 | 结论 | 依据 |
|---|---|---|
| 原文 | 动态头过滤提升视觉定位性（消融胜出） | #文献 Zotero key 6U2FFLUA（本轮未复核其具体图表编号） |
| 本轮 | 头过滤使聚焦度提升（8/8），框内质量占比提升（8/8）；方向一致 | #实测 `metrics.json` |

顶部 5.2 的交叉形态值得记录：头过滤给出的「绝对定位」更强（框内质量更高），但单峰落框率未随之提高——在 bottle（2008_000015）一例，unfiltered 峰值落框而 filtered 峰值移到框外（`mass_in_bbox` filtered 0.2485 仍高于 unfiltered 0.1656）；bicycle（2008_000036）两口径峰值均未落框（该类目标在图中占比极小，`mass_in_bbox`≈0.03）。说明「更聚焦」在当前阈值口径下不必然等价于「argmax 落框」，两者需分别报告 #实测 同上。

### 6 讨论

本轮唯一未能对齐的是原文的定量基准：官方仓不含评测脚本，ImageNet／VQAv2 扰动、PascalVOC 分割、filler 过滤三组数字均无从复算，故 5.2 为代理口径下的方向性证据，不与原文数字直接比较 #文献 Zotero key 6U2FFLUA；#代码事实 `D:\code\DEX-AR\README.md`。

口径层面的两点提示，供迁移 `attention_metrics.py` 时对齐：

1. DEX-AR 需 `attn_implementation="eager"` 且开启反传，实验侧 `cross_attention_raw` 为纯前向读注意力，两者成本不同（本轮测得 DEX-AR≈8× 纯前向）；回答了审核未知项 #实测 5.2 表。
2. 头过滤的收益体现在「质量分布更集中／框内质量更高」，而「峰值落框」是更粗的判据，二者在小样本上可能方向相反（5.3） #实测 `metrics.json`。

### 7 结论

- DEX-AR 在集群 A100 上忠实复现成功：单图与 8 图批量均稳定产出 filtered／unfiltered 句级与逐 token 热图，图像 token 边界自洽 #实测 `metrics.json`、`slurm-574061.out`。
- 其原文核心结论「动态头过滤提升视觉定位性」在本轮代理口径下方向成立：8/8 图聚焦度提升、8/8 图框内质量占比提升 #实测 `metrics.json`。
- 相对纯前向注意力，DEX-AR 单图约 8× 开销（量级随目标句 token 数增长） #实测 5.2 表。
- 依 [[注意力可视化]] 判读条款，本轮判 **pass**：判据（同类性、迁移便利性、契合度）成立，复现无「实现错误」残留。

### 8 产物与复现指引

| 产物 | 路径 |
|---|---|
| 复现脚本 | `D:\code\DEX-AR\repro_dexar.py` |
| 作业脚本 | `D:\code\DEX-AR\jobs\repro.sbatch` |
| VOC manifest 生成器 | `D:\code\DEX-AR\tools\make_voc_manifest.py` |
| 批量结果（集群） | `/public/home/yhhao/wyf/DEX-AR-runs/20261126-103959/`（PNG + .npy + `metrics.json`） |
| 指标 JSON（本地副本） | `D:\Obsidian\Research\组会与实验\附件\图片\DEX-AR\dexar-metrics.json` |
| 对照图（本地） | `D:\Obsidian\Research\组会与实验\附件\图片\DEX-AR\` |

复现命令（集群，环境 `dexar`）：先 `sbatch --export=ALL,MANIFEST=<manifest.json>,OUT=/public/home/yhhao/wyf/DEX-AR-runs jobs/repro.sbatch`；manifest 用 `python tools/make_voc_manifest.py --voc <VOC2012> --pairs dog:2008_000019 ... --out <manifest.json>` 生成。

### 9 未知项与后续轮次

- #猜想 单峰落框率未随头过滤提升，可能因头过滤后的归一化改变了峰位；需更大样本或换「多峰落框」口径验证。
- 原文定量表（扰动 AUC、分割 soft-IoU／EPG、filler 过滤 SNR）未复算，若需对接须按原文自建评测流程，成本待评估 #文献 Zotero key 6U2FFLUA。
- 与实验 `attention_metrics.py` 的 bbox 指标对接尚未实测，DEX-AR 与 `cross_attention_raw` 的逐图相关性未比较。
- 后续轮次候选：换 `layer_index` 扫描（原文缺省 -10 与 playground 的 0 不同）看聚焦度—定位是否更优；换 BakLLaVA-v1 验证跨模型通用性。
