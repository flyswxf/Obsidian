---
状态: 完成
日期: 2026-11-26
研究线: "[[注意力可视化]]"
基础: 论文全文 + 官方代码仓 + LeGrad 复现报告
---

## 审查目标与判据

本审查面向 [[排版攻击]] 的注意力分析实验：在图像中植入对抗性文本贴片，诱导生成式 VLM（LLaVA-1.5-7B）输出攻击者期望答案，需要一套可靠的注意力度量口径来支撑「贴片位置」这一变量。迁移目标实验代码如下：

- `cross_attention_raw`：逐生成步取最后一层注意力中「当前 query token → 图像 patch」分量，头平均、跨生成步平均，得 $24\times24$，即「生成 token 对图像 patch 的交叉注意力」#代码事实 `D:\code\Physical-Attention-Attack\src\models\llava_model.py`（第 129–151 行）。
- `vision_attention_raw`：CLIP ViT 最后一层 CLS token → 图像 patch 自注意力，$24\times24$ #代码事实 `D:\code\Physical-Attention-Attack\src\models\llava_model.py`（第 176–183 行）。
- 指标：给定攻击贴片 bbox，算 patch 网格内／外的注意力质量 #代码事实 `D:\code\Physical-Attention-Attack\src\utils\attention_metrics.py`（第 96–133 行）；批量主循环见 #代码事实 `D:\code\Physical-Attention-Attack\exp1_baseline.py`（第 398–556 行）。

三条核心判据：

1. **同类性**：方法是否面向与实验同一类模型——生成式、产出文本的多模态模型（LLaVA／Qwen-VL 这类 decoder-only 生成式 VLM），而非 CLIP 这类对比／分类模型。
2. **迁移便利性**：能否迁移到 LLaVA 并通用到各类生成文本的多模态模型；需指出改造点（是否需可微标量目标、访问哪一层／哪个注意力张量、是否需重训微调）。
3. **契合度**：与实验两类注意力信号的契合程度。

关键判断纪律：严格区分「论文面向的模型／任务」与「方法本身是否模型无关可迁移」。

## 方法逐一审查

### CLIP AFT（ICLR 2025）

- **面向模型／任务**：CLIP 的**对比／分类**场景。方法全称 Adversarial Fine-Tuning，目标函数建立在「语言-图像相似度」$T_x^{\mathsf T}I_x$ 上（$I_x$、$T_x$ 为单位长度图像／文本嵌入的余弦相似度），是典型的图文对比模型设定 #文献 Zotero key QYEMZ8KF，§3.1 式 (1)。论文自述「改进 CLIP 视觉编码器的**内在可解释性**」，评价载体为 Simple Gradients、Grad-CAM、Rollout、M2IB、Grad-ECLIP 等后验归因图与 network dissection 概念检测器 #文献 Zotero key QYEMZ8KF，§4.1、§4.2。

- **方法要点**：这是**训练时干预**，不是后验归因方法。核心是带范数正则的对抗微调：$\min_\theta \mathbb{E}\max_{\delta_x} \tfrac12(T_x^{\mathsf T}\mathbb{E}_z[f_\theta(x+z+\delta_x)]-T_x^{\mathsf T}I_x)^2-h(\delta_x)$；再以「与文本嵌入无关的上界」替换内层目标，得到只微调图像编码器的无监督目标 $\min_\theta\mathbb{E}\max_{\delta_x}\tfrac12\|\mathbb{E}_z[f_\theta(x+z+\delta_x)]-I_x\|^2-h(\delta_x)$ #文献 Zotero key QYEMZ8KF，§3.1 式 (1)、§3.2 式 (6)(7)。作者证明该 AFT 等价于对输入梯度（SmoothGrad）加 $h^\star$ 正则，并选 Huber 型 $h^\star$ 诱导稀疏显著性 #文献 Zotero key QYEMZ8KF，§3.1 式 (4)(5)。

- **官方代码支持**：训练脚本是完整微调循环。优化器 `AdamW(params, lr, wd)` #代码事实 `D:\code\CLIP_AFT\train\adversarial_training_clip.py`（第 154–158 行）；主循环含 `loss_total.backward(); optimizer.step()` 与 `torch.save(...state_dict(), 'final.pt')` #代码事实 同上（第 298–314、199 行）；README 给出 `--steps 20000 --lr 1e-5 --attack pgd --iterations_adv 10` 的微调命令与三个预训练 checkpoint 下载 #代码事实 `D:\code\CLIP_AFT\README.md`（第 24–43 行）。产出**是一个被改动权重的 CLIP checkpoint**，本身不输出任何注意力图；`interpretation.ipynb` 中生成图的手段仍是普通 Simple Gradients（`torch.autograd.grad(logit[0,0], image)`）与 Grad-CAM（`target_layers=[...resblocks[-2]]`）#代码事实 `D:\code\CLIP_AFT\interpretation.ipynb`（第 246、317 行）。

- **迁移到 LLaVA 生成式 VLM 的可行性：不可行（作为可迁移的归因工具）；作为「改造视觉塔」的路线则等价于重新微调。** 依据：
  - 该方法的收益**完全来自改变视觉编码器权重**。把 LLaVA-1.5 的视觉塔（CLIP ViT-L/14-336）换成 AFT 版本，确是本论文 §4.3 的既有实验：论文称把 LLaVA 的视觉编码器替换为 AFT 版本后，可得到更干净的复合注意力图且「不损害生成质量」#文献 Zotero key QYEMZ8KF，§4.3、图 7。但这正是「**重训视觉塔**」的代价，而非免训练迁移。
  - 需访问与改造的对象是**视觉编码器权重**（对抗微调 + 范数正则），需要 PGD 内层最大化、可微的平滑相似度目标与图像数据集（论文用 ImageNet 训练集微调 2 epoch）#文献 Zotero key QYEMZ8KF，§4；对应实现为 `train/adversarial_training_clip.py` 的整段训练循环 #代码事实 `D:\code\CLIP_AFT\train\adversarial_training_clip.py`。
  - 替换视觉塔后，LLaVA 的投影连接器（connector）与新视觉特征的分布是否对齐论文未给出量化，仅声称与「用 CLIP 嵌入训练的下游 LLM」兼容 #文献 Zotero key QYEMZ8KF，§3.2；对 LLaVA-1.5 而言，可能还需用原训练数据继续对齐 connector，才能保证生成质量与注意力语义不被破坏。
  - **是否存在免重训的等价替代**：没有。方法的收益内生于权重变化（对抗微调诱导的稀疏梯度），不存在同一效果的免训练算子。若目标只是得到「更干净的图像侧显著性」，等价替代是**直接改用后验归因方法**（Grad-CAM、Chefer、LeGrad 等），无需改动模型。

- **对各类生成文本多模态模型的通用性**：弱。论文把框架推广到「任何具有跨模态中间嵌入层的基础模型」#文献 Zotero key QYEMZ8KF，§3.2 与附录 A.4，但**所有实证都限定在 CLIP 视觉编码器及其派生的图文模型**；对 decoder-only 生成式 VLM，它只能作用于视觉编码器子模块，且该子模块须是 CLIP 式（有图像嵌入）。它**不触及**生成侧的交叉注意力，也不提供任何文本生成 token 的可微目标。

- **判定依据**：方法核心是「CLIP 专属」的训练改造，而非模型无关的可迁移归因。判据 1 不满足（面向对比／分类的 CLIP）；判据 2 代价为「重训视觉塔」，非免训练工具；判据 3 仅间接影响 `vision_attention_raw`（改变模型本身），且不产出任何直接可用的注意力度量。

### LeGrad（ICCV 2025）

- **面向模型／任务**：论文自称「specifically designed for ViTs」，用作解释 ViT 图像编码器的 token 归因 #文献 Zotero key 8S8R9TW4，摘要、§1。实验覆盖 ImageNet-S 分割、OpenImagesV7 开放词表定位、ADE20K 声音定位、ImageNet-val 扰动测试；所用模型为 OpenCLIP ViT-B/16、ViT-BigG/14、SigLIP、ImageBind、CoCa #文献 Zotero key 8S8R9TW4，§4。即：**面向视觉编码器（对比／分类型），并借 CLIP 式文本-图像相似度构造标量**，全部实证均非 decoder-only 生成式 VLM。

- **方法要点**：对某一层 $l$，取该层特征经映射 $C$ 得到的标量激活 $s_l=\bar{y}^l_{[\hat c]}$（CLIP 下即文本嵌入与图像特征的点积结果），对**该层自注意力图** $A_l$ 求梯度、ReLU 去负、按头与查询维平均、去掉 [CLS] 列、reshape 并 min-max 归一化，再跨层累加：

$$\hat{E}^{(l)}=\mathrm{ReLU}\!\Big(\tfrac{\partial s_l}{\partial A_l}\Big),\qquad E=\mathrm{norm}\!\Big(\sum_{l}\mathrm{reshape}\big(\overline{\hat{E}^{(l)}}\big)\Big)$$

#文献 Zotero key 8S8R9TW4，§3.2 式 (4)–(9)。关键性质：只需对注意力图的梯度，无需任何训练／微调；层间直接相加、无需逐层归一化（这是相对 CheferCAM 乘性传播的主要优势）。官方实现见 #代码事实 `D:\code\LeGrad\legrad\wrapper.py`（`compute_legrad_clip`，第 144–186 行）：`sim = text_embedding @ img_feat.transpose(-1,-2)` 构造标量（第 167–169 行），`torch.autograd.grad(one_hot, [attn_map], ...)` 求注意力图梯度（第 174–177 行），`image_relevance = grad.mean(dim=1).mean(dim=1)[:, 1:]` 去 [CLS] 列（第 179 行），末段 `min_max` 归一（第 185 行）。

- **官方代码支持**：实现仅做三件事——给 ViT 各层挂 hook 暴露注意力图、把文本-图像相似度设为可微标量、一次反向取注意力梯度。全程冻结参数（`param.requires_grad = False`）#代码事实 `D:\code\LeGrad\legrad\wrapper.py`（第 77–92 行），**不涉及任何权重更新**。

- **本项目已有复现**：LeGrad 在 ImageNet-S 的实现口径已完整复现，定性行为一致（提示可切换定位对象），全量 12,403 张基线 mIoU 46.06／PixAcc 80.73／mAP 71.90（224），与原文 58.66／77.52／82.49 差 12–13 点，且经分辨率、阈值、归一化、聚合、层深五项探针排查均未支持「实现口径差异」#实测 `D:\Obsidian\Research\组会与实验\实验报告\LeGrad-ImageNet-S复现实验报告.md`（§5–§7）。**该「完成」结论的适用边界须明确**：它只证明 LeGrad 在 **CLIP 图像编码器的 CLS 聚合 + 图文相似度**这一场景下可按原文口径跑通并复现定性行为；它**不构成** LeGrad 在「生成式解码器交叉注意力」上有效的证据，因为复现全程没有出现任何生成 token、文本生成目标或 decoder 交叉注意力。

- **迁移到 LLaVA 生成式 VLM 的可行性：需改造（免重训，但须替换目标函数）。** 两条路线：

  1. **视觉塔路线（近直接迁移）**：LLaVA-1.5 的视觉塔即 CLIP ViT-L/14-336，与 LeGrad 原来作用的对象同类。可在该塔的各层自注意力上直接套用 LeGrad，得到一张「图像侧显著性」，即对实验现有 `vision_attention_raw` 的**梯度敏感度加权升级版**。难点：塔内并无成对的文本编码器，$s_l$ 无天然来源，须外接一个 CLIP 文本编码器以图文相似度构造标量（近似复现报告的做法）。此路线代价最小、无重训，但它解释的是**外部 CLIP 相似度**，而非 LLaVA 的实际生成决策。

  2. **端到端生成式路线（须替换目标函数）**：在生成式 VLM 中，LeGrad 赖以工作的标量 $s_l$ 并不存在——decoder 输出的是**下一个 token 的 logits／概率**，没有文本-图像相似度。迁移须把 $s_l$ 换成生成目标的可微标量，典型取目标 token 的对数概率 $s=\log p(w^\ast\mid\cdot)$，再对**交叉注意力图** $A_l^{\text{cross}}$（LLaVA 中即 LLM 自注意力里 query 为文本 token、key 为图像 token 的那部分）求梯度，$\hat E^{(l)}=\mathrm{ReLU}(\partial s/\partial A_l^{\text{cross}})$，按头／查询维平均后取「读出行的图像列」切片，跨层累加。这与本项目 #代码事实 `D:\code\Physical-Attention-Attack\src\models\llava_model.py` 的 `cross_attention_raw` 是同一注意力张量，差别仅在「原始注意力量值」升级为「梯度敏感度量值」。改造点：需可微标量目标（$w^\ast$ 取贪心答案首个实词）、需 `attn_implementation="eager"` 暴露逐层注意力图、需一次反传（`retain_grad`），**无需任何重训**。

  两条路线共同的**局限**：
  - **需可微标量目标**：路线 2 必须锁定一个目标 token，只反映该 token 一步的相关性传播，不覆盖整段生成；目标 token 选择会改变结果。
  - **显存与算子代价**：一次反传 + 逐层注意力图梯度，比实验现有 `torch.no_grad()` 直接读原始注意力更重。
  - **长视觉前缀下的语义对齐风险**：decoder-only + 长视觉前缀易诱发注意力质量向序列最早 token（BOS／最早视觉 token）抽干，使「本应反映区域读取」的读数退化为 sink 行为。LeGrad 用「层间直接相加」而非 CheferCAM 的乘性 rollout，可**部分规避**这种逐层复合的吸收态，但端到端路线仍需以「视觉质量占比」这类闸门监控读数是否被文本前缀抽干。
  - **复现结论不覆盖本场景**：如上，ImageNet-S 的「完成」是 CLIP 场景证据，迁移到生成式交叉注意力属**未验证**区间，须重新建立判据。

- **对各类生成文本多模态模型的通用性**：中等偏可行。只要模型 (a) 暴露逐层注意力图、(b) 存在可微标量目标（任何生成式 VLM 都有 next-token logits），路线 2 即可套用，故对 LLaVA／Qwen-VL 一类生成文本的多模态模型具备通用潜力；但方法原文的公式与实证都未覆盖生成式解码器，通用性需自行论证与验证，不是「拿来即用」。

- **判定依据**：方法核心是「ViT 自注意力梯度敏感度」，机制上与模型类型弱耦合（不依赖分类头，只依赖可微标量 + 注意力图），故**可迁移到生成式 VLM**；但论文面向对比／分类的 CLIP 场景，$s_l$ 在生成式模型中不存在，必须换目标函数，且本项目复现证据不覆盖该场景。

## 小结

- **CLIP AFT（ICLR 2025）**：不适用。判据 1 面向 CLIP 对比／分类场景；判据 2 需重训视觉塔（对抗微调改动权重）而非免训练工具，无免重训等价替代（若只要干净显著性，应改用后验归因方法）；判据 3 仅间接影响 `vision_attention_raw`，且不产出直接可用的注意力度量。
- **LeGrad（ICCV 2025）**：需改造。判据 1 论文场景为 CLIP，但机制弱耦合于模型类型；判据 2 免重训，视觉塔路线近直接迁移，生成式交叉注意力路线须把目标函数由「图文相似度 $s_l$」换为「生成 token logit $\log p(w^\ast\mid\cdot)$」并对交叉注意力图求梯度；判据 3 与本实验两类信号同张量，契合度较高，但复现报告只证明 CLIP 场景，生成式迁移属未验证区间。
- **总判断**：两篇都面向 CLIP 视觉编码器，但性质截然不同——CLIP AFT 是**改造模型训练**（对「已有 VLM 的归因工具」不适用），LeGrad 是**免训练后验归因**（可迁移到生成式 VLM，但须替换目标函数并在新场景重建判据）。作为「排版攻击」注意力度量口径，LeGrad 的生成式改造路线更具落地价值；CLIP AFT 不宜作为归因工具列入迁移清单，其唯一可用形态是「重训视觉塔 + 后验归因」的组合，代价与风险均高于直接使用后验方法。
