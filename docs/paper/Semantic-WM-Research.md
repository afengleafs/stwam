# Semantic-WM 论文调研总结

调研对象: **Reconstruction or Semantics? What Makes a Latent Space Useful for Robotic World Models**

- 项目页: https://hskalin.github.io/semantic-wm/
- arXiv: https://arxiv.org/abs/2605.06388
- PDF: https://arxiv.org/pdf/2605.06388
- 代码: https://github.com/chandar-lab/semantic-wm
- Hugging Face 权重: https://huggingface.co/Nilaksh404/semantic-wm
- 作者: Nilaksh, Saurav Jha, Artem Zholus, Sarath Chandar
- 版本: arXiv v1, 2026-05-07

## 1. 一句话结论

这篇论文回答的问题不是“哪个 world model 生成的视频最清晰”，而是:

> 对机器人 action-conditioned latent diffusion world model 来说，latent space 应该优先保留像素重建细节，还是优先保留动作、任务、物体语义和可规划结构？

论文的核心结论是: **语义表征 latent, 例如 V-JEPA 2.1、Web-DINO、SigLIP 2，通常比 VAE/Cosmos 这类重建型 latent 更适合机器人 world model 的 policy evaluation、planning、action recoverability 和 OOD robustness；像素 fidelity 好并不等价于机器人动态建模好。**

这和本仓库 STWAM/V-TWAM 的本地对照结果方向一致: STWAM 使用 V-JEPA 2.1 + S-VAE 96-d semantic latent，标准 LIBERO 平均 **89.75%**；V-TWAM 把视觉 latent 换成 SD3 VAE 16-d 后降到 **69.25%**。LIBERO-PRO 完整扰动均值也从 STWAM **37.65%** 降到 V-TWAM **25.15%**。

## 2. 背景: 为什么“重建 vs 语义”是关键问题

很多视频 world model 采用 latent diffusion model: 先用一个 encoder 把图像压到 latent，再在 latent 空间里学习未来帧的 transition。传统做法是用 VAE 或类似 autoencoder，因为它们重建像素稳定、decoder 可用、训练视频 diffusion 方便。

但机器人 world model 的目标不是单纯生成好看的视频，而是让 latent transition 能保存:

- action 改变了什么物体状态；
- gripper 是否和目标物产生接触；
- 物体、容器、抽屉、按钮等 affordance 是否被正确绑定；
- 任务是否朝语言目标推进；
- 生成的未来是否能被 planner 或 policy 当作可靠环境。

因此，**像素重建型 latent** 和 **语义表征型 latent** 的差别会直接影响机器人 world model 是否能用于控制。重建型 latent 保留纹理、颜色、边缘和局部外观，但不一定让“动作导致的状态变化”线性或显式。语义 latent 通常更抽象，可能牺牲精细几何和像素细节，但更容易暴露 object layout、task progress 和 action-relevant features。

论文认为，评价机器人 world model 时只看 PSNR/SSIM/FID/FVD 是不够的，因为这些指标会把 decoder 质量和真实动态质量混在一起。一个模型可以渲染出合理画面，但生成的是任务错误状态；另一个模型可能画面没那么锐利，却更准确地保留动作和任务结构。

## 3. 论文做了什么受控实验

论文的设计是固定大部分系统变量，只改变 latent interface。

固定项包括:

- 数据集: Bridge V2；
- transition model: action-conditioned DiT；
- training schedule、optimizer、history 长度、action conditioning；
- rollout/evaluation protocol。

变化项只有:

- encoder `f_phi`；
- optional adapter `alpha_psi`；
- decoder path。

这使得实验主要隔离 “latent space 本身” 的影响。

### 3.1 数据集

主训练数据是 **Bridge V2**:

- 约 60K 条 WidowX 250 real-robot manipulation demonstrations；
- 13 个 task families；
- 每条 episode 包含 RGB observation、7-DoF end-effector action、language instruction。

另外用于 success classifier probe 的数据是 **SOAR**:

- 约 30.5K 条 WidowX success/failure episodes；
- 正负比例约 1:2；
- 用来测试 latent trajectory 是否保留“给定语言任务是否成功”的信息。

### 3.2 被比较的 latent encoder

论文比较两大类 encoder。

重建型 latent:

| Encoder | 维度 | 特点 |
|---|---:|---|
| SD3 VAE | 16 | Stable Diffusion 3 的 VAE latent，偏像素重建 |
| VA-VAE | 32 | 更现代的视觉 autoencoder，对齐视觉基础模型但仍偏重建 |
| Cosmos CI encoder | 16 | NVIDIA Cosmos tokenizer，服务视频/世界模型重建 |

语义型 latent:

| Encoder | native 维度 | adapter 后维度 | 特点 |
|---|---:|---:|---|
| V-JEPA 2.1 ViT-L | 1024 | 96 | 自监督预测式视觉表征，论文中 policy-facing 指标最强之一 |
| Web-DINO | 1024 | 96 | DINOv2/Web-SSL 类语义表征 |
| SigLIP 2 ViT-L | 1152 | 96 | vision-language aligned 表征，任务/语言相关性强 |

对语义 encoder，论文同时评估:

- native high-dimensional latent；
- 通过 S-VAE adapter 压缩到 `d=96` 的 compact latent。

### 3.3 模型结构

论文中的 world model 是 latent diffusion transition model:

```text
image o_t
  -> frozen encoder f_phi
  -> latent z_t in R^{N x D}
  -> optional frozen S-VAE adapter alpha_psi, z_tilde in R^{N x d}
  -> action-conditioned DiT transition
  -> future latent rollout
  -> decoder / pixel decoder
  -> predicted future images
```

关键点:

- encoder 冻结；
- adapter 冻结；
- decoder 冻结或作为 adapter 体系的一部分使用；
- world model 训练时只更新 DiT transition model；
- DiT 每帧接收相同 token 数 `N=256`，因此 high-dimensional semantic latent 不会显著增加 transformer 主干计算量；
- native semantic latent 使用 shallow-wide DDT head 缓解高维输出瓶颈；
- adapter semantic latent 用 S-VAE 压到 96 维，使 diffusion 更容易。

### 3.4 S-VAE adapter

S-VAE adapter 的作用是把高维 frozen semantic feature 压到 compact latent:

```text
z in R^{N x D}
  -> Transformer encoder
  -> Gaussian bottleneck: mu, logvar
  -> z_tilde in R^{N x 96}
  -> Transformer decoder reconstruct feature
```

训练目标包括:

- feature MSE；
- cosine loss，保留语义方向；
- FFT/spectral loss，保留空间高频结构；
- KL regularization，让 latent 更像标准 Gaussian；
- pixel loss，通过 lightweight pixel decoder 从 compact latent 重建 RGB。

论文强调它使用的是 **S-VAE path**: pixel decoder 的 pixel loss 不反传进 adapter/encoder，只训练 decoder，因此 semantic feature adapter 本身主要由 feature-space loss 和 KL 约束。

这点和本仓库 `model/vjepa_encoder.py` 的接入一致: V-JEPA 2.1 backbone 冻结，S-VAE adapter 冻结，输出 canonical semantic-wm layout `[B,T,16,16,96]`。

### 3.5 训练配置

论文主实验配置:

- 输入分辨率: 256x256；
- clip 长度: `T=10`；
- history: `H=2`；
- frame skip: 2；
- future prediction: 8 frames；
- action: 7-D；
- DiT-S/B/L 三种规模；
- DiT-S: hidden size 384, depth 12, heads 6；
- DiT-B: hidden size 768, depth 12, heads 12；
- DiT-L: hidden size 1024, depth 24, heads 16；
- inference autoregressive rollout，用 10-frame sliding context；
- sampler 使用 10 Euler steps。

附录给出的训练资源:

| Run | 训练时长 | 资源 |
|---|---:|---|
| S-VAE + pixel decoder | 约 55 h | 4 x H100 |
| DiT-S world model | 约 6-7 h | 4 x H100 |
| DiT-L world model | 约 34 h | 4 x H100 |

注意: 论文正文描述的是 flow matching objective；本仓库接入 HF `DiT-S_D96.pt` 时，`STWAMConfig` 和 `RUN.md` 里默认按 semantic-wm checkpoint 的 `ddpm` / v-prediction 设置使用。这是本仓库工程接入和论文最终实验叙述之间需要在写作时区分的地方。

## 4. 评价体系

论文提出三条评价轴，不只看视觉指标。

### 4.1 Planning and downstream policy performance

包括:

- CEM latent planning: 给定真实 k-step transition，在 world model 里搜索 action sequence，使预测 latent 最接近目标 latent；
- k=1 和 k=4 两种 horizon；
- OpenVLA-7B policy-in-the-loop: 把 OpenVLA 放进生成的 world model 里 rollout；
- VLM judges: InternVL 3.5 和 Qwen 3.6；
- OOD robustness: distractor-object perturbation 和 OOD-instruction perturbation；
- Borda rank、consensus success rate、interaction quality、instruction following。

CEM 是一个很好的诊断，因为它绕开 pixel decoder，直接问 latent dynamics 是否足够 action-sensitive。

### 4.2 Pixel fidelity and scene geometry

包括:

- SSIM；
- LPIPS；
- FID；
- FVD；
- temporal LPIPS；
- point-track consistency；
- WorldArena 的 perceptual/geometric metrics；
- flow score、depth error、subject consistency 等。

这些指标仍然重要，因为 visual policy 需要看得懂 rollout。但论文核心观点是: **它们不能单独决定 world model 是否适合机器人。**

### 4.3 Latent representation quality

包括:

- IDM action recovery: 从 latent pair/trajectory 恢复 action chunk，用 Pearson r 衡量；
- success classifier probe: 用 SOAR 上的 latent trajectory 和语言指令训练 success classifier，测试生成 rollout 是否保留任务成功信息；
- 比较 encoder latent 上的 ceiling 和 world-model generated latent 上的退化。

这条评价轴直接问 latent 是否保留“动作”和“任务”。

## 5. 关键实验结果

### 5.1 Policy-in-the-loop: 语义 latent 明显更强

DiT-S 下，论文 Table 1 的 consensus VLA success rate:

| Encoder | Consensus SR | Borda rank | ID SR | OOD distractor SR | OOD instruction SR | CEM k=1 | CEM k=4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| VAE | 0.169 | 25 | 0.375 | 0.287 | 0.200 | 0.111 | 0.612 |
| VA-VAE | 0.175 | 23 | 0.350 | 0.250 | 0.200 | 0.097 | 0.543 |
| Cosmos | 0.244 | 16 | 0.425 | 0.362 | 0.275 | 0.112 | 0.661 |
| V-JEPA 2.1 | 0.344 | 6 | 0.600 | 0.575 | 0.400 | 0.084 | 0.424 |
| V-JEPA 2.1-96 | 0.362 | 8 | 0.600 | 0.537 | 0.250 | 0.089 | 0.548 |
| Web-DINO | 0.212 | 21 | 0.550 | 0.512 | 0.250 | 0.090 | 0.474 |
| Web-DINO-96 | 0.300 | 11 | 0.600 | 0.512 | 0.275 | 0.090 | 0.531 |
| SigLIP 2 | 0.325 | 9 | 0.537 | 0.500 | 0.263 | 0.082 | 0.523 |
| SigLIP 2-96 | 0.331 | 15 | 0.625 | 0.588 | 0.312 | 0.086 | 0.537 |

直接读表:

- VAE/VA-VAE 是最弱 policy-in-loop 组；
- Cosmos 比 VAE 好，但仍低于大多数 semantic variants；
- V-JEPA 2.1 和 SigLIP 2 系列整体最强；
- native semantic latent 在 CEM action recovery 上通常更强；
- adapter-96 对 VLA SR 和 visual rollout 有帮助，但可能损伤精细 CEM action geometry。

附录 Table 12 做了 family-level bootstrap:

| 指标 | Semantic family 相对 Reconstruction family |
|---|---:|
| VLA SR | +9.8 percentage points, 95% CI [2.5, 17.7], p=0.0129 |
| OOD SR | +13.6 percentage points, 95% CI [8.8, 18.4], p < 5e-5 |
| CEM k=1 error | -0.0266, 95% CI [0.0122, 0.0412] lower error, p=0.00015 |

这说明论文不是只靠个别表格观察，而是做了任务配对 bootstrap，支持“semantic family policy-facing 指标更好”这个结论。

### 5.2 IDM / success probe: semantic latent 更保留动作和任务结构

论文 Table 2 的 DiT-S latent representation quality:

| Encoder | Enc. IDM k=1 | Enc. IDM k=4 | WM IDM k=1 | WM IDM k=4 | Enc. success acc | WM success acc |
|---|---:|---:|---:|---:|---:|---:|
| VAE | 0.507 | 0.478 | 0.476 | 0.464 | 0.835 | 0.716 |
| VA-VAE | 0.549 | 0.744 | 0.545 | 0.719 | 0.868 | 0.744 |
| Cosmos | 0.626 | 0.673 | 0.581 | 0.651 | 0.851 | 0.723 |
| V-JEPA 2.1 | 0.829 | 0.865 | 0.781 | 0.840 | 0.905 | 0.789 |
| Web-DINO | 0.820 | 0.845 | 0.729 | 0.794 | 0.906 | 0.788 |
| SigLIP 2 | 0.772 | 0.793 | 0.697 | 0.757 | 0.903 | 0.823 |

这里最重要的不是绝对分数，而是趋势:

- V-JEPA/Web-DINO 的 IDM Pearson r 明显高，说明 action-induced latent change 更容易被读出；
- 生成后的 WM latent 仍保留这类优势；
- SigLIP 2 的 WM success classifier accuracy 最高，说明语言/任务成功信息保留得好；
- VAE 虽然可能生成视觉上合理的帧，但 latent 不够 action-explicit。

这解释了为什么 VAE latent world model 可能“看起来会动”，但 policy/planning 用起来弱。

### 5.3 视觉指标: 语义 latent 并没有简单牺牲视觉质量

直觉上会以为 semantic latent 抽象、decoder 弱，因此视觉质量一定差。论文结果更复杂:

- DiT-S 规模下，semantic variants 在 SSIM、LPIPS、FVD、JEPA similarity、subject consistency、depth error、temporal LPIPS 等指标上很有竞争力，甚至经常领先；
- V-JEPA 2.1-96 拿到很强的 FVD；
- SigLIP 2-96 的 SSIM 很强；
- Web-DINO 系列在 JEPA similarity、subject consistency、depth error、t-LPIPS 上很强；
- DiT-L 规模下，VAE 的视觉指标追得很快，甚至在 FID、image quality、aesthetic quality、JEPA similarity、depth error、dynamic degree、FVD 等指标上获得多个第一。

论文对此的解释是:

- 大模型可以弥补重建型 latent 的视觉建模能力；
- 但 policy-facing 和 action-centric 指标并不会完全跟随视觉指标提升；
- visual fidelity 是必要条件，但不是充分条件。

### 5.4 模型规模: 视觉差距会缩小，但 action-centric 差距还在

论文观察到 DiT-S -> DiT-L 时:

- VAE/Cosmos 的 VLA SR 和视觉指标上升明显；
- reconstruction latents 在大模型下更接近 semantic latents；
- 但 CEM action recovery、IDM、success probe 上仍落后。

含义是: 如果只堆更大的 transition model，重建 latent 可以更会“画”，但不一定更会“理解动作导致的任务状态变化”。

### 5.5 Multi-view: 多视角有帮助，但数据不足会伤视觉

论文将 DiT-S 模型 fine-tune 到 BridgeV2 的多视角 episode:

- 多视角能改善 CEM action recovery；
- 但在 limited multi-view data 下，video quality 可能下降；
- semantic encoders 对这种 degradation 更稳。

对机器人来说这点很重要: 多视角能提供几何和遮挡信息，但如果多视角数据规模不足，模型可能牺牲画面一致性。语义 latent 在这种设置下相对更抗退化。

### 5.6 Adapter tradeoff: 96-d 更容易 diffusion，但会损伤精细控制几何

论文的 adapter 结论很细:

- S-VAE adapter 把 high-dimensional semantic feature 压成 96-d，能让 diffusion 和 decoding 更容易；
- adapter-96 通常提升 visual fidelity 和 VLA-in-loop；
- 但 native semantic latent 往往在 CEM action recovery、OOD robustness、PCK coverage 上更强；
- 原因是 compression 可能丢掉一部分精细 action geometry。

Web-DINO adapter dimension ablation 显示 d=96 是一个较好的折中点:

| Web-DINO DiT-S | d=16 | d=96 | D=1024 native |
|---|---:|---:|---:|
| VLA SR | 0.256 | 0.269 | 0.181 |
| SSIM | 0.711 | 0.728 | 0.722 |
| LPIPS | 0.196 | 0.181 | 0.199 |
| FID | 8.37 | 6.00 | 7.63 |

这对本仓库很相关: STWAM 当前用的正是 `V-JEPA 2.1 + S-VAE adapter d=96`，属于“更方便 diffusion 和策略使用”的折中，而不是 native V-JEPA 1024-d。

### 5.7 失败模式: 两类 latent 崩的方式不同

论文的 qualitative finding 很关键:

重建型 latent 的典型失败:

- 生成画面连贯，但任务语义错；
- 容易 hallucinate 错误物体或保持原来的 action pattern；
- 在 OOD instruction 下可能继续执行原任务；
- 结果看起来像真的，但 task state 是错的。

语义型 latent 的典型失败:

- task-level intent 更容易保住；
- 但几何、接触、精细动作程度可能不准；
- 例如 drawer 没开到位、gripper-contact 不稳定、物体位置/形变细节不精确。

这句话可以概括:

> reconstruction latents hallucinate task semantics; semantic latents miss geometry and contact.

这也解释了本地 STWAM 在 LIBERO-PRO 上的表现: Semantic perturbation 很强，但 Position/Swap 和 Task 几乎崩塌。语义 latent 帮了“语言/物体语义”，但没有自动解决精细空间布局、任务重定义和 contact dynamics。

## 6. 论文给出的实践 recipe

论文第 5 节给出一套 semantic latent diffusion robotics world model 的建议:

1. 不要先优化视觉真实感，而要先选一个让 action 和 task progress 显式的 latent space。
2. 默认选择 pretrained semantic encoders，而不是 reconstruction-only VAE。
3. 如果需要视觉 rollout 或 VLA-in-loop，使用 adapter compression，例如 S-VAE d=96。
4. 如果做精细 planning，则要小心 adapter 可能损伤 action geometry，native semantic latent 可能更好。
5. transition model 推荐 spatial-temporal DiT，temporal causal block，spatial non-causal denoising。
6. 对 high-dimensional semantic latent，使用 wide DDT head 和 dimension-aware noise schedule shift。
7. 评价时必须同时看 visual、latent、planning、policy-facing 指标。

## 7. 和当前 STWAM / V-TWAM 仓库的关系

本仓库不是简单复现实验，而是把 semantic-wm 的 semantic latent video expert 改造成一个 world-action policy / WAM 测试台。

### 7.1 直接继承 semantic-wm 的部分

当前仓库已有 vendored semantic-wm 源码:

- `model/_swm/`
- `model/_swm/models/encoders/_vjepa2_src/`

STWAM 使用的关键路径:

- `model/vjepa_encoder.py`
  - V-JEPA 2.1 ViT-L frozen backbone；
  - frozen S-VAE adapter；
  - 输出 `[B,T,16,16,96]`；
  - 双摄像头时在宽度维拼接为 `[B,T,16,32,96]`。

- `model/video_expert.py`
  - semantic-wm DiT 原结构包装；
  - 从 `DiT-S_D96.pt` 初始化；
  - 默认 `in_channels=96, patch_size=1, dim=384, layers=12, heads=6, wide_head=True`。

- `model/config.py`
  - `semantic_dim=96`；
  - `adapter_latent_dim=96`；
  - `freeze_vjepa=True`；
  - `freeze_adapter=True`；
  - `time_dist_shift=2.45`，对应非 VAE latent 的 dimension-aware shift 思路。

因此 STWAM 的视觉 latent 选择和 semantic-wm 论文的主张高度一致: **V-JEPA 2.1 + S-VAE 96-d semantic latent**。

### 7.2 STWAM 相对 semantic-wm 新增/改变的部分

STWAM 不是只做 video world model，而是把 video tower 和 action policy 耦合:

- `model/modeling_stwam.py`
  - semantic-wm video DiT；
  - 轻量 action expert；
  - per-layer zero-init MoT adapter；
  - action expert 读取 clean history frame K/V；
  - video future loss + action flow-matching loss。

关键差异:

- semantic-wm 论文评估的是 world model latent choice；
- STWAM 评估的是 semantic latent WAM policy 在 LIBERO/LIBERO-PRO 上的实际成功率；
- semantic-wm 用 BridgeV2 real robot；
- STWAM 当前用 LIBERO simulation；
- semantic-wm 的 policy-in-loop 是 OpenVLA 在 world model 里 rollout；
- STWAM 是自己的 action expert 直接输出 action chunk。

所以 semantic-wm 可以作为 STWAM 的视觉 latent 设计依据，但不能直接等同于 STWAM 的全部方法贡献。

### 7.3 V-TWAM 是对 semantic-wm 结论的本地受控对照

V-TWAM 的设计:

- `vtwam/vae_encoder.py`: SD3 VAE latent `[B,T,H/8,W/8,16]`；
- `vtwam/config.py`: `in_channels=16, patch_size=2`；
- 保持 WAM/action/MoT 结构，换掉视觉 latent interface。

这相当于把 semantic-wm 的“semantic vs reconstruction latent”问题移植到当前 LIBERO policy setting。

本地结果:

| Model | 标准 LIBERO Avg | LIBERO-PRO Avg |
|---|---:|---:|
| STWAM, V-JEPA/S-VAE semantic latent | 89.75 | 37.65 |
| V-TWAM, SD3 VAE latent | 69.25 | 25.15 |
| Delta | +20.50 | +12.50 |

标准 LIBERO 分项:

| Suite | STWAM | V-TWAM | Delta |
|---|---:|---:|---:|
| Spatial | 88.0 | 81.0 | +7.0 |
| Object | 98.0 | 67.0 | +31.0 |
| Goal | 90.0 | 81.0 | +9.0 |
| Long | 83.0 | 48.0 | +35.0 |

这和 semantic-wm 的主结论一致:

- semantic latent 对 object affordance 和 long-horizon task structure 更有帮助；
- VAE latent 可以学到部分控制，但在 object/long 上明显弱；
- 视觉重建 latent 不足以支撑强 policy-facing 行为。

不过本地结果也暴露 semantic-wm thesis 的边界:

- STWAM 的 LIBERO-PRO Position/Swap = 0；
- Task perturbation 平均只有 4.5；
- V-TWAM 在这些项也弱；
- 这说明 semantic latent 不是万能泛化器，尤其不能自动解决空间重排、任务逻辑重定义和 contact-geometry 泛化。

## 8. 对 STWAM 论文写作的启发

可以借 semantic-wm 支撑的主张:

1. **Semantic latent 是合理设计选择。** V-JEPA/S-VAE 并不是拍脑袋选的；已有受控实验显示 semantic latent 在 action recovery、planning、VLA-in-loop 和 OOD robustness 上优于 VAE/Cosmos。
2. **STWAM vs V-TWAM 是本地独立验证。** 在 LIBERO policy setting 下，换成 SD3 VAE 后成功率大幅下降，和 semantic-wm 的 BridgeV2 结论同向。
3. **不要用视觉质量解释策略能力。** 如果后续补 video rollout 图，必须同时报告 action/policy success，不要只看视频好不好看。
4. **S-VAE 96-d 是折中。** 它提高 diffusion/decoding 易用性，但可能牺牲精细 action geometry；这可以解释 STWAM 的 Position/Task 弱点。
5. **STWAM 的贡献应写成参数效率与受控诊断。** 不是宣称“语义 latent 自动泛化”，而是展示它在哪些维度有效、在哪些维度失败。

建议写法:

> Inspired by controlled evidence that semantic latent spaces preserve action-relevant structure better than reconstruction-aligned VAE latents in robotic world models, STWAM instantiates a compact V-JEPA/S-VAE latent WAM and evaluates its policy-facing effect on LIBERO and LIBERO-PRO. Our V-TWAM ablation confirms the same direction: replacing semantic latents with SD3 VAE latents sharply reduces standard and robustness success rates, especially on object-centric and long-horizon suites.

需要避免的过强表述:

- 不要说 semantic latent 解决了 spatial generalization；
- 不要说 STWAM 的 LIBERO-PRO 泛化强；
- 不要把 V-JEPA/S-VAE 说成无预训练；
- 不要把 semantic-wm 的 BridgeV2 world-model 结论直接当成 LIBERO policy success 结论；
- 不要只用 Object/Semantic perturbation 的好结果掩盖 Position/Task 的崩塌。

## 9. 局限与批判性解读

论文自身局限:

1. **只在 BridgeV2 / WidowX embodiment 上做主要受控实验。** 跨 ALOHA、Franka、RoboCasa、LIBERO 等域是否完全一致仍需验证。
2. **policy-in-loop 用固定 OpenVLA，而不是真实 robot closed-loop。** 它测试的是 world model 作为 policy evaluation environment 的能力，不等于真实部署成功率。
3. **VLM judge 有偏差。** 论文用多个 VLM 和 Borda/consensus 降低偏差，但 success rate 仍受 judge 质量影响。
4. **不同 encoder 的预训练数据和容量不同。** 结果证明 semantic family 有优势，但不能完全拆分“语义目标、模型容量、预训练数据规模”的贡献。
5. **decoder path 仍影响 visual metrics。** 对 semantic latent 使用 adapter pixel decoder，视觉指标不只是 latent transition 的函数。
6. **adapter 有双刃剑。** d=96 方便 diffusion，但也可能丢失精细 action geometry。

对当前仓库的额外提醒:

- 本地 STWAM/V-TWAM 对照非常有价值，因为它把 semantic-wm 的 latent hypothesis 落到 LIBERO policy success；
- 但如果要写论文，需要明确指出数据域、动作头、训练目标和评估协议都不同；
- 当前 STWAM 的优势主要是 Object/Semantic robustness 和参数效率；
- 当前最大短板仍是 Position/Task/Environment generalization。

## 10. 可执行后续实验建议

为了把 semantic-wm 调研转化为 STWAM 下一步实验，可以优先做:

1. **native vs adapter semantic latent 消融。**
   - 当前是 V-JEPA/S-VAE 96-d；
   - 可以小规模测试 V-JEPA native 1024-d 或更高维 adapter，如 192/256；
   - 目标是验证 Position/Task 弱点是否来自 96-d compression 丢失 geometry。

2. **VAE latent 的视频质量 vs policy 成功率对照。**
   - 对 V-TWAM 保存 rollout videos；
   - 检查是否存在“画面合理但任务错”的 reconstruction latent 典型失败。

3. **CEM/IDM 类 latent probe 移植到 LIBERO。**
   - 不只看成功率；
   - 训练 inverse dynamics probe，比较 STWAM latent 和 V-TWAM latent 的 action recoverability；
   - 这会更直接呼应 semantic-wm。

4. **Position/Task failure diagnosis。**
   - 对 Position/Swap 为 0 的任务分析 object pose、slot、目标描述和 policy attention；
   - 判断是视觉 latent 缺几何、action expert 缺位置条件，还是数据覆盖不够。

5. **多视角/深度/pose token 消融。**
   - semantic-wm 显示 multi-view 对 action recovery 有帮助；
   - STWAM 当前双摄像头已拼接宽度维，可以进一步测试明确的 camera embedding、depth/pose token 或 object-centric state。

## 11. 最终理解

这篇论文最有价值的地方是把“world model 好不好”从视频指标中拆出来，变成三个问题:

1. 画面是否合理；
2. latent 是否保留 action 和 task progress；
3. policy/planner 是否能用这个 world model 做出更好的行为判断。

它的答案是: **机器人 world model 的 latent space 应该优先服务 action-relevant semantics，而不是只服务 pixel reconstruction。**

对当前 STWAM 来说，这篇论文是视觉 latent 选择的强背景依据；而本仓库 STWAM/V-TWAM 对照则是对这个依据的本地 policy-level 验证。最诚实的结论是:

> V-JEPA/S-VAE semantic latent 确实显著优于 SD3 VAE latent，尤其体现在 object-centric 和 long-horizon manipulation；但 semantic latent 仍没有解决空间重排、任务重定义和精细接触几何，因此后续改进应集中在 Position/Task diagnostics，而不是继续只追求更高标准 LIBERO 平均分。
