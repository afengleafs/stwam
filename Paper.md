# 2026 VLA / WAM 在 LIBERO 与 LIBERO-PRO 上的结果定位

更新日期: 2026-06-30

本文只纳入两类数字: 1) 论文或官方项目公开报告的 LIBERO / LIBERO-PRO 成功率; 2) 本仓库已落盘的 STWAM 评测 CSV。标准 LIBERO 使用 `Spatial / Object / Goal / Long(libero_10)` 四套各 10 个任务的平均成功率; LIBERO-PRO 使用官方定义的 Object / Position / Semantic / Task / Environment 扰动。STWAM 已补齐五类 LIBERO-PRO 扰动, 每个 suite/condition 为 10 任务 x 10 episodes。

## 1. STWAM 本地实验结果

本地模型是单一联合策略, 同一个 `checkpoint/stwam_libero_ddp/latest.pt` 评测四套 LIBERO。训练日志显示: 4 x RTX 5090, global batch 32, 300k steps, 总训练时间约 145:59:33; 训练数据为 40 个 LIBERO 任务, 无额外 embodied robot pretraining。模型总参数约 444M, 其中 V-JEPA 2.1 ViT-L 编码器冻结约 342.8M; 可训练部分约 101.6M, 从零初始化的 MoT adapter + action expert + proprio 约 28.1M。

| Suite | Success | Trials | Mean steps | 本地来源 |
|---|---:|---:|---:|---|
| LIBERO-Spatial | 88.0% | 100 | 130.61 | `logs/eval_libero_spatial.csv` |
| LIBERO-Object | 98.0% | 100 | 136.07 | `logs/eval_libero_object.csv` |
| LIBERO-Goal | 90.0% | 100 | 127.44 | `logs/eval_libero_goal.csv` |
| LIBERO-Long / libero_10 | 83.0% | 100 | 303.48 | `logs/eval_libero_10.csv` |
| **Average** | **89.75%** | **400** | - | - |

最弱点很集中: Spatial 的 task 5 "on the ramekin" 只有 50%, Long 的双物体/长程组合任务还有明显掉点。这和语义 latent 的优缺点一致: 物体语义足够强, 精细几何和长程 affordance binding 仍弱。

## 2. 标准 LIBERO: 2026 主流 VLA / WAM 对比

下表按论文公开表格摘录。不同论文可能使用不同 action chunk、评测脚本和 checkpoint, 因此 0.x 到 2.x 个点的差异不要过度解读; 但平均水平和强弱结构足够清楚。

| Method | 类型 | 额外 robot data / 规模备注 | Spatial | Object | Goal | Long | Avg | vs STWAM Avg | 来源 |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| **STWAM (ours)** | semantic-latent WAM | 无 embodied PT; 总约 0.44B, 可训约 0.10B | 88.0 | 98.0 | 90.0 | 83.0 | **89.8** | - | 本地 CSV |
| OpenVLA fine-tuned | VLA | 7B, 大规模 robot PT | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 | +13.3 | OpenVLA-OFT 表 1 |
| pi0 fine-tuned | VLA | 约 3B, robot PT | 96.8 | 98.8 | 95.8 | 85.2 | 94.2 | -4.4 | OpenVLA-OFT 表 1 |
| OpenVLA-OFT | VLA | 7B, full fine-tuning recipe | 97.6 | 98.4 | 97.9 | 94.5 | 97.1 | -7.3 | OpenVLA-OFT 表 1 |
| pi0.5 | VLA | pi0 系列升级 | 98.8 | 98.2 | 98.0 | 92.4 | 96.9 | -7.1 | VLA-JEPA 表 1 |
| VLA-JEPA | VLA + JEPA latent | 无 robot PT; JEPA 表征 | 96.2 | 99.6 | 97.2 | 95.8 | 97.2 | -7.5 | VLA-JEPA 表 1 |
| Fast-WAM | WAM | 5B Wan2.2 latent video DiT; 无 embodied PT | 98.8 | 98.8 | 96.8 | 96.0 | 97.6 | -7.8 | Fast-WAM 表 1 |
| Cosmos Policy | WAM | video generative policy | 98.5 | 99.5 | 97.2 | 98.8 | 98.5 | -8.8 | Cosmos Policy 表 1 |
| LingBot-VA | WAM / causal world model | 8B backbone + action expert | 98.5 | 99.6 | ~97.4 | 98.5 | 98.5 | -8.8 | LingBot-VA 表 2 |
| Motus | WAM | 只公开 LIBERO-Long 可比数字 | - | - | - | 97.6 | - | - | Motus 表 1 |

结论比较直接:

1. STWAM 不是标准 LIBERO SOTA。当前平均 89.75%, 比 2026 公开强 WAM / VLA 低约 7-9 个点。
2. STWAM 的 Object 已接近饱和, 98.0% 只比 Fast-WAM / Cosmos / VLA-JEPA 低 0.8-1.6 个点。
3. 主要差距来自 Spatial、Goal 和 Long, 尤其 Long 对 Cosmos Policy / LingBot-VA 差约 15 个点。这说明瓶颈不是"识别物体", 而是空间关系、连续交互和关键 fixture affordance。
4. 和老 OpenVLA fine-tuned baseline 相比, STWAM 平均高 13.3 个点, Long 高 29.3 个点。这个比较能说明小型 WAM 结构有效, 但不能拿来声称超过 2026 SOTA。
5. 参数/算力定位是 STWAM 最有价值的部分: 它用冻结 V-JEPA + 73.5M semantic-wm DiT-S + 轻量 MoT/action expert, 在消费级 5090 上得到接近 pi0 的 LIBERO 水平。但 Fast-WAM 只训 20k steps, STWAM 训 300k steps, 这一点论文里要诚实标注。

## 3. LIBERO-PRO: STWAM 本地鲁棒性结果

LIBERO-PRO 的动机是检查标准 LIBERO 高分是否来自场景/初始状态/指令记忆。官方扰动包括 Object、Position、Semantic、Task、Environment。STWAM 已按同一 checkpoint 补齐五类扰动; 每个 cell 都是 100 trials。

| Suite | Ori | Object | Position / Swap | Semantic | Task | Environment |
|---|---:|---:|---:|---:|---:|---:|
| libero_spatial | 88.0 | 88.0 | 0.0 | 86.0 | 0.0 | 21.0 |
| libero_object | 98.0 | 88.0 | 0.0 | 98.0 | 0.0 | 29.0 |
| libero_goal | 90.0 | 44.0 | 0.0 | 94.0 | 9.0 | 25.0 |
| libero_10 | 83.0 | 41.0 | 0.0 | 78.0 | 9.0 | 43.0 |
| **Mean** | **89.8** | **65.3** | **0.0** | **89.0** | **4.5** | **29.5** |

这组数字的解释应当很克制:

1. **Semantic perturbation 是强项。** 四套平均 89.0%, 相对原始 89.8% 几乎不掉。这支持"V-JEPA 语义 latent + language conditioning 对指令改写鲁棒"的论点。
2. **Object perturbation 是混合结果。** Spatial/Object 两套能保持 88%, 但 Goal/Long 掉到 44%/41%。这说明模型不是纯粹记忆所有外观, 但对关键交互物、fixture 和长程任务中的 affordance 绑定仍然依赖训练分布。
3. **Position / Swap 和 Task 是主要崩塌项。** Position 四套全为 0%, Task 平均 4.5%。这说明模型对初始空间布局和任务逻辑重定义几乎没有可迁移性。
4. **Environment 有少量迁移但不强。** 四套平均 29.5%, 其中 libero_10 environment 为 43.0%, 但 spatial/object/goal 都只有 21-29%。这比 Position/Task 好, 但仍说明场景/工作台分布被强记忆。
5. **完整 LIBERO-PRO 扰动均值为 37.65%。** 这是 Object、Position、Semantic、Task、Environment 五类 x 四套的 20 个 cell 平均值; 它应作为鲁棒性弱点而不是卖点来呈现。

## 4. LIBERO-PRO 官方 leaderboard 与 STWAM 的相对位置

截至本次调研, 主流 WAM 论文如 Fast-WAM、Cosmos Policy、LingBot-VA、Motus 暂未公开完整 LIBERO-PRO 细表。LIBERO-PRO 官方 repo 主要给出 VLA 模型的 leaderboard。下表把官方 0-1 分数换成百分比, 并按四套任务求各扰动均值。

| Model | Object avg | Position avg | Semantic avg | Task avg | Environment avg | Official Total | 来源 |
|---|---:|---:|---:|---:|---:|---:|---|
| OpenVLA | 93.0 | 0.0 | 97.3 | 0.0 | 68.0 | 52.0 | LIBERO-PRO README |
| pi0 | 90.5 | 0.0 | 90.5 | 0.0 | 38.8 | 44.0 | LIBERO-PRO README |
| pi0.5 | 96.0 | 20.8 | 95.8 | 0.8 | 52.8 | 53.0 | LIBERO-PRO README |
| MolmoAct | 76.0 | 1.5 | 85.8 | 1.5 | n/a | 41.0 | LIBERO-PRO README |
| NORA | 70.5 | 0.0 | 86.3 | 0.0 | n/a | 40.0 | LIBERO-PRO README |
| x-VLA | 79.0 | 0.8 | 96.8 | 6.8 | n/a | 46.0 | LIBERO-PRO README |
| **STWAM (ours)** | **65.3** | **0.0** | **89.0** | **4.5** | **29.5** | **37.65** | 本地 `eval_libero_plus/results/libero_pro_matrix.csv` |

直接比较:

1. STWAM 的 Semantic 平均 89.0%, 接近 pi0 的 90.5%, 高于 MolmoAct/NORA, 低于 OpenVLA/pi0.5/x-VLA。语言扰动鲁棒性是可以写进论文的正结果。
2. STWAM 的 Object 平均 65.3%, 低于官方 VLA leaderboard。差距主要不是 Object/Object 或 Spatial/Object, 而是 Goal/Object 和 Long/Object 的崩塌。
3. STWAM 的 Position/Swap 和 OpenVLA、pi0 一样是 0, 明显低于 pi0.5 的 20.8%。Task 平均 4.5%, 只略高于 OpenVLA/pi0 的 0, 低于 x-VLA 的 6.8%。
4. STWAM 的 Environment 平均 29.5%, 低于 OpenVLA/pi0/pi0.5 的 38.8-68.0%。这说明更换 workspace / scene 后, 语义 latent 并没有自动带来强环境泛化。
5. 完整扰动均值 37.65%, 低于官方 VLA leaderboard 的 40-53 区间。标准 LIBERO 89.75% 和 LIBERO-PRO 37.65% 的落差是论文里必须正面呈现的泛化边界。

## 5. 对 STWAM 论文定位的建议写法

最稳妥的主张:

> STWAM demonstrates a parameter-efficient semantic-latent world-action model: it combines Fast-WAM-style training-time video co-training and inference-time direct action prediction with frozen V-JEPA semantic latents, achieving 89.75% average success on standard LIBERO using a 0.44B-parameter model and no embodied pretraining.

可以强调的创新:

1. **语义 latent WAM 闭环。** Fast-WAM 证明 video co-training 对 action policy 有效, VLA-JEPA 证明 JEPA 表征适合控制; STWAM 把这两条线合到一个小型 WAM 里。
2. **参数效率。** 相比 5B-8B 的 WAM / VLA, STWAM 总体约 0.44B, 可训练约 0.10B, 且冻结大视觉编码器。
3. **语言扰动鲁棒。** LIBERO-PRO Semantic 四套平均 89.0%, 几乎保持原始成功率。
4. **鲁棒性诊断清楚。** 完整 LIBERO-PRO 显示失败集中在 Position、Task 和 Environment, 后续改进目标明确。

必须避免的过强表述:

1. 不要说 STWAM 达到 2026 SOTA; 标准 LIBERO 比 Fast-WAM / Cosmos / LingBot-VA 低 7-9 个点。
2. 不要笼统说 "object generalization 很强"; Goal/Object 和 Long/Object 掉到 44%/41%。
3. 不要说 LIBERO-PRO 泛化强; 完整扰动均值只有 37.65%, Position 为 0%, Task 只有 4.5%。
4. 不要把 Spatial/Position 弱点藏起来; 这恰好是后续工作的最清晰方向。

最应该补的实验:

1. Spatial/Position 诊断: 增加 object pose / slot / depth / multi-frame history 的消融。
2. Task 泛化诊断: 分析 task perturbation 中语言目标和 BDDL target-state 的变化是否被策略忽略。
3. 语义 latent vs 像素/VAE latent 的受控消融, 对齐 Fast-WAM 的 "video co-training" ablation。
4. 训练步数和模型尺寸消融: 100k/200k/300k, semantic_dim 96 vs 更高维, num_history 1/2/4。

## 6. 来源

公开论文 / 项目:

- OpenVLA-OFT: https://arxiv.org/html/2502.19645v1
- VLA-JEPA: https://arxiv.org/html/2602.10098v1
- Fast-WAM: https://arxiv.org/html/2603.16666v1
- Cosmos Policy: https://arxiv.org/html/2601.16163v1
- LingBot-VA / Causal World Modeling: https://arxiv.org/html/2601.21998v1
- Motus: https://arxiv.org/html/2512.13030
- LIBERO-PRO paper: https://arxiv.org/abs/2510.03827
- LIBERO-PRO official repo / leaderboard: https://github.com/Zxy-MLlab/LIBERO-PRO

本地实验文件:

- `logs/eval_libero_spatial.csv`
- `logs/eval_libero_object.csv`
- `logs/eval_libero_goal.csv`
- `logs/eval_libero_10.csv`
- `eval_libero_plus/results/libero_pro_matrix.csv`
- `eval_libero_plus/results/libero_pro_summary.md`
- `logs/stwam_libero_ddp_20260623_112007.log`
