# STWAM 实验补全计划 (基于现有结果 + 代码审阅)

更新日期: 2026-07-09

本文档基于对 `Paper.md`(结果台账)、`STWAM_ICRA2027_draft.md`(论文草稿)、`Semantic-WM-Research.md`(理论依据)以及实际模型代码(`model/`、`vtwam/`、`logs/`)的交叉审阅,给出投稿前需要补做的实验清单。所有"裂缝"均有代码/落盘数据出处。

---

## 0. 现有实验的结论强度评估

| 实验 | 结论 | 证据强度 | 判断 |
|---|---|---|---|
| Q1 标准 LIBERO 89.75% | 小模型可用 | 强 | 真值成功率, 可信 |
| Q2 V-TWAM 语义>重建 (−20.5) | latent 因果重要 | 中(有 confound) | 方向对, 但不是干净对照 |
| Q3 LIBERO-PRO 五轴 | 诚实鲁棒性边界 | 强 | 诊断性叙事, 加分项 |
| Q4 k-draws +3.0 | 降方差有小益 | 弱(疑似噪声) | 2k step + 单 seed, 大概率落在 eval 噪声内 |

真正硬的只有 Q1 和 Q3;Q2 是核心卖点但有硬伤;Q4 目前是负担。

---

## 1. 审阅中发现的关键裂缝(草稿未暴露)

### 裂缝 1 — ~~V-TWAM 不是 "only latent interface" 的干净对照~~ 【2026-07-09 已勘误：此判断错误】

> **勘误(2026-07-09):** 下表基于 `model/config.py` / `vtwam/config.py` 的**默认值**,而非实际落盘 checkpoint 的 config。核对训练后的 checkpoint(`checkpoint/stwam_libero_ddp/latest.pt` 与 `vtwam/checkpoint/vtwam_libero_ddp/step_00300000.pt`)后发现:**两模型的 `num_history`、`n_frames`、`chunk_size`、`n_action_steps`、`num_views` 全部一致**,原表所述"重规划/上下文 confound"**不存在**。

实际 checkpoint config 对比:

| 因子 | STWAM 实际 | V-TWAM 实际 | 判断 |
|---|---|---|---|
| `num_history` | 1 | 1 | ✓ 一致(原表误写 2 vs 1) |
| `n_frames` | 9 | 9 | ✓ 一致 |
| `chunk_size` | 32 | 32 | ✓ 一致 |
| `n_action_steps`(重规划) | 32 | 32 | ✓ 一致(原表误写 8 vs 32) |
| `num_views` | 2 | 2 | ✓ 一致 |
| `in_channels`(latent) | 96 | 16 | latent 本身 |
| `patch_size` | 1 | 2 | 保持 token 数相等(16×32) |
| `wide_head` | True | False | 随预训练 DiT-S_D96/D16, latent 相关 |
| `time_dist_shift` | 2.45 | 1.0 | latent-appropriate 噪声 schedule |

**修正后结论:** V-TWAM 其实是一个**相当干净**的对照 —— 训练/推理超参全对齐,剩余差异(`patch_size`/`wide_head`/`time_dist_shift`)几乎都是"选了这个 latent 就必然/应当随之改变"的内生项,而非可自由拉平的 confound。这**削弱了**原先"审稿人一枪毙命"的担忧,反而对论文 Q2 有利。唯一还算非 latent 强制的差异是 `wide_head`,而它来自预训练 checkpoint 家族,拉平需重训(用户已否决)。

**对 P0-2 的影响:** 原 P0-2("把 V-TWAM 从 32 步对齐到 8 步以拆重规划 confound")**前提失效**——两者本就都在 32 步。改为可选的"重规划鲁棒性 sweep"(STWAM 与 V-TWAM 同时在 `n_action_steps=8/16` 重评,验证 −20.5 gap 不随重规划频率消失),待用户确认。

### 裂缝 2 — 全篇没有 "world model 是否有用" 的消融(最致命)

`train_ablation.py` / `model/ablation.py` 支持的 ablation 只有 `k_draws / proprio_dropout / pooled_adaln`; `logs/ablation_eval/` 磁盘上跑过的只有 `kstudy_k1`、`kstudy_k8`。

即: 论文叫 "World-Action Model", 核心主张是 "co-train 视频预测能帮策略"(Fast-WAM recipe), 但**从未做过关掉视频协同训练的对照**。如果去掉 video 塔后成功率不变, 这篇就不是 WAM, 只是一个在 JEPA latent 上做条件的策略。

### 裂缝 3 — 单 seed, 且 eval 噪声已被观测到很大

`eval_libero.py` 中 `seed` 出现 0 次, 所有数字均为单次运行。对比同一 checkpoint 两批评测:

| Suite | 主评测 (Table I) | rollout_ctx1 复评 | 差 |
|---|---:|---:|---:|
| Object | 98.0 | 100.0 | +2 |
| Long | 83.0 | 77.0 | **−6** |

同一模型同一任务 Long 抖 6 个点 ⇒ Q4 的 k-draws +3.0(per-suite +4/+3/−1/+6)完全在噪声内;连 89.75 都需要置信区间。

### 裂缝 4 — 一个已做完却被雪藏的好实验(rollout ctx1 vs ctx9)

`logs/rollout_eval/` 中有一组未进论文的实验, 直接支撑 Sec. III-E "测试时不需要 rollout 视频":

| | Spatial | Object | Goal | Long |
|---|---:|---:|---:|---:|
| ctx1(只读 clean history, =正式推理) | 89 | 100 | 89 | 77 |
| ctx9(喂入 rollout 出来的未来 latent) | 21 | 9 | 27 | 29 |

测试时读 "生成的未来" 成功率断崖式崩塌 ⇒ 把 "act directly" 从 assertion 变成强实验证据。应提拔进正文作图, 现在浪费了。

### 其余已知缺口(草稿 TODO 有, 尚未做)
- 无推理延迟/控制频率(efficiency 论文缺 wall-clock 不可接受);
- "Position 崩塌来自 96-d 压缩" 是纯推测, 无 native 1024-d / d=192 对照;
- 训练步数 confound(300k vs Fast-WAM 20k), 无 100k/200k/300k 曲线;
- 零定性证据, 无 "V-TWAM 画面合理但任务错" 对比图。

---

## 2. 实验清单(按优先级)

### P0 — 不做会被拒

| # | 实验 | 挡掉的攻击 | 成本 | 是否需重训 |
|---|---|---|---|---|
| P0-1 | **World-model 存在性消融**: (a) λ_video=0; (b) freeze video 塔; (c) 去掉 video 塔, action expert 直接读 encoder latent(≈VLA-JEPA 式) | "这根本不是 WAM, 是个 JEPA 策略" | 中(可短训 50–100k 对照) | 是 |
| P0-2 | **拆 V-TWAM confound**: 先对齐 `n_action_steps`(V-TWAM 推理时执行 8 步而非 32, **不用重训**); `num_history`/`wide_head` 若能对齐则重训一版 | "掉分是重规划频率不是 latent" | 对齐重规划 = 几小时(纯 re-eval) | 否(重规划) / 是(其余) |
| P0-3 | **IDM probe**: 从两个 checkpoint 抽 (z_t, z_{t+k}), 训小回归器比 action recoverability(Pearson r) | 把 Q2 从"相关"升级为"机制", 对齐 Semantic-WM Table 2 | 最低, 不动主模型, 几小时 | 否 |
| P0-4 | **推理延迟/控制频率**: 测 prefill-once vs 每步重跑 video 塔的 ms 与 Hz | efficiency 无 wall-clock 站不住 | 几十分钟 | 否 |

### P1 — 让论文从"诚实"变"严谨"

| # | 实验 | 作用 | 成本 |
|---|---|---|---|
| P1-5 | **≥3 seed 或 bootstrap CI**: 至少给 STWAM/V-TWAM 头条数字 + Q2 gap 加置信区间 | 治单 seed; 决定 Q4 生死 | re-eval, 中等 |
| P1-6 | **semantic_dim 消融**: 96 vs 192 / native 1024 | 验证 "Position 崩塌 = 压缩丢几何" | 需小规模重训 |
| P1-7 | **提拔 ctx1/ctx9 rollout 进正文** + 补定性对比图 | 坐实 Sec. III-E 与 Semantic-WM thesis | 数据已在, 仅作图 |

### P2 — 有则更强

| # | 实验 | 作用 |
|---|---|---|
| P2-8 | 100k/200k/300k 训练曲线 | 回应 Fast-WAM 20k 公平性 |
| P2-9 | k-draws 用 seeds 重跑或降至 appendix | 现为噪声级正结果, 要么坐实要么撤 |

---

## 3. 建议执行顺序

先做**成本几乎为零、收益最高**的:

1. **P0-2(对齐重规划频率的 re-eval)** — 直接检验核心卖点(V-TWAM gap)在拆掉 confound 后是否缩水, 纯 re-eval。
2. **P0-4(推理延迟测量)** — 几十分钟, 补 efficiency 硬指标。
3. **P0-3(IDM probe)** — 不动主模型, 把 Q2 升级为机制证据。
4. **P0-1(world-model 消融)** — 需重训, 但决定论文身份, 必须做。
5. 其余 P1 / P2 视时间补齐。

---

## 4. 一句话结论

现有实验能证明 "小语义-latent 策略在 LIBERO 上可用、鲁棒性边界诚实", 但证不住标题里的两个核心词 —— **"World-Action Model"(缺 world-model 消融)** 和 **"semantic latent 因果"(V-TWAM 有重规划/上下文 confound)**。最高杠杆三件事: ① 补 world-model 存在性消融(定身份)、② 对齐 V-TWAM 的 `n_action_steps` 重评(拆 confound, 近零成本)、③ IDM probe(相关升级为机制, 最便宜)。

---

## 5. 相关文件索引

- 结果台账: `Paper.md`
- 论文草稿: `STWAM_ICRA2027_draft.md`
- 理论依据调研: `Semantic-WM-Research.md`
- STWAM 模型: `model/modeling_stwam.py`、`model/mot_adapter.py`、`model/config.py`
- V-TWAM 模型/配置: `vtwam/modeling_vtwam.py`、`vtwam/config.py`
- 消融基建: `train_ablation.py`、`model/ablation.py`(当前仅支持 k_draws / proprio_dropout / pooled_adaln)
- 未报告的 rollout 实验: `logs/rollout_eval/*_rollout_ctx1.csv`、`*_rollout_ctx9.csv`
- k-draws 结果: `logs/ablation_eval/kstudy_k1/`、`logs/ablation_eval/kstudy_k8/`

---

## 6. 实验结果(执行中,更新于 2026-07-09)

> 4 个 P0 实验的落地状态与结果。GPU:P0-1 训练占 cuda:0-3;评测占 cuda:4-7。

### P0-4 推理延迟 ✅ 完成

`bench_latency.py`(仅 WAM 双塔,排除 V-JEPA 编码),RTX 5090,bf16,batch=1,iters=100,flow_steps=10,chunk=32,replan=32:

| 路径 | ms / chunk |
|---|---|
| A) prefill-once(部署路径) | **210.6 ± 4.9** |
| B) naive 每 flow 步重跑 video 塔 | 505.1 ± 13.2 |
| **single-prefill 加速比 A vs B** | **2.40×** |

- 摊薄规划开销 = **6.58 ms / 执行动作**(每 32 步重规划一次),约 **152 action-plans/s**(摊薄)。
- 结论:单次 prefill + 缓存 K/V 相对朴素重跑省 2.4×;动作推理本身很轻(百毫秒级/chunk),efficiency 卖点有 wall-clock 支撑。V-JEPA 编码开销另计(每控制步一次)。
- 产物:`logs/latency/latency.txt`。

### P0-1 world-model 消融 (λ_video=0) ⚠️ 降级为 sanity check(2026-07-11 勘定:不作论文 finding)

**方法学勘定(用户裁定):** 本实验从 300k co-trained checkpoint 出发做 40k 微调,world-model 结构已烧入权重,"移除式"微调消融**无法区分"目标无用"与"目标作用已完成"**,Δ≈0 是注定的,不可信、不进论文。若要回答"视频目标是否必要",唯一有效设计是 from-scratch 对照(入背版)。

保留的 sanity 观察(仅工程参考):λv=0 微调 40k 后 video loss 从 ~0.09 升至 0.555 而 4 套成功率不降(90.75 vs 91.75,噪声内)——说明**已收敛策略不依赖训练期 WM 预测精度来维持性能**,与 ctx1/ctx9 结论自洽。相关 checkpoint 与评测产物已于 2026-07-11 清理删除。

### P0-2 拆 V-TWAM confound ❌ 已退役

见 §1 勘误:STWAM 与 V-TWAM 的 `n_action_steps` 本就同为 32(且 num_history/chunk/n_frames 全对齐),无重规划 confound 可拆。用户确认**丢掉 P0-2**;论文里直接写明"两模型均每 32 步重规划(matched)"即可。`vtwam/eval_libero.py` 已加的 `--n-action-steps` flag 保留(无害,备将来做重规划鲁棒性 sweep)。

### P0-3 IDM probe ✅ 完成

`idm_probe.py`(冻结 encoder,不训主模型):抽 (z_i, z_{i+k}) 空间 mean+std 池化 → 小 MLP 回归两帧间 action 段,test Pearson r(7 维动作均值)。N=7680 对(k=1)/ 4800 对(k=4):

| Encoder | IDM r (k=1) | IDM r (k=4) |
|---|---:|---:|
| STWAM (V-JEPA/S-VAE 96-d) | **0.892** | **0.878** |
| V-TWAM (SD3-VAE 16-d) | 0.799 | 0.817 |
| **Δ (STWAM − V-TWAM)** | **+0.093** | **+0.062** |

- 结论:语义 latent 在两个 horizon 上 action recoverability 都更高 → "V-TWAM 成功率低" 不只是策略学得差,而是**它的 latent 本身就更难读出 action-induced change**。方向与 Semantic-WM Table 2 一致(该文 V-JEPA IDM≈0.83/0.87 > VAE≈0.51/0.48;我们的 VAE 绝对值更高是因池化/probe 不同且 LIBERO 比 BridgeV2 简单,但**gap 符号一致**)。这把 Q2 从"成功率相关"升级为"机制证据"。
- 产物:`logs/idm_probe/idm_results.csv`。

---

## 7. 当前计划:ABC-connector 性能提升 + 消融矩阵(2026-07-11 重定向)

**论文主线(用户裁定):** 主成果 = STWAM vs V-TWAM 受控对比 + IDM probe;下一步 = 借 ABC(arXiv:2606.27375)connector 消融提升 STWAM 性能。ABC 发现:π0.5 式逐层耦合 connector 最差(mean progress 11.7%),**pooled adaLN 最好(61.4%)**;STWAM 当前 MoT 逐层 read path 正是前者模式。仓库 `model/ablation.py` 的 pooled-adaLN 基建(`PooledAdaLNCond`,零初始化)已就绪、从未运行。

### 两阶段协议(遵守"移除式微调消融不可信"标准)

- **B1 探针轮(finetune,仅加法式选型):**
  - `pooled_add`:`--pooled-adaln add`,20k,零初始化新通路叠加在逐层 read 上——涨即真涨;
  - `pooled_only`:`--pooled-adaln only`,40k,read path 置零冻结——移除式,仅方向参考不作结论。
- **B2 评测噪声 CI:** 主 checkpoint 4 套 × 3 重复,mean±std;任何"提升"主张以此为标尺。
- **B3 决胜轮:** 赢家配置 from-scratch 300k(需给 `train_ddp.py` 移植 pooled/k-draws 参数)→ STWAM-v2 头条数字 + 3× CI + 五轴 PRO。

### 消融矩阵(论文 ablation 表)

| 轴 | 状态 |
|---|---|
| Latent 界面(V-TWAM, −20.5 / PRO −12.5) | ✅ 主成果 |
| IDM 机制探针(0.892/0.878 vs 0.799/0.817) | ✅ 主成果配套 |
| Connector(逐层 vs pooled-add vs pooled-only) | 🚀 B1/B3 |
| 测试时 rollout(ctx1 vs ctx9, 89→21) | ✅ 提拔进正文 |
| k-draws(k1 vs k8 @2k, +3.0 噪声级) | ✅ appendix, CI 定生死 |
| 推理路径(prefill-once 2.40×, 210.6ms/chunk) | ✅ Sec. III-E |
| 评测噪声 CI | 🚀 B2 |
| 五轴 LIBERO-PRO(v1 37.65;v2 待补) | ✅/🚀 |

**背版(不占本轮 GPU):** from-scratch λ_video=0(WAM 目标必要性)、num_history 1↔2(草稿 H=2 与实际 H=1 不符,先改草稿)、proprio-dropout、RTC/action-prefix(ABC H.2)。

### 清理记录(2026-07-11,释放 ~10 GB)

删除:`checkpoint/ablation_{r0_control,r_video0,r1_pdrop05,kstudy_k1,kstudy_k8}/`(kstudy 评测 CSV 保留)、`logs/ablation/`、`logs/ablation_eval/{r0_control,r_video0}/`、冒烟杂项(`logs/eval_libero_spatial.log`、`rollout_eval/smoke_*`、`vtwam_eval/smoke/`、根 `__pycache__/`)。
保留:STWAM/V-TWAM 主线 checkpoint(latest+100k/200k/300k)、两份主训练日志(训练时长引用来源)、全部主成果 CSV。

### B1 connector 探针(2026-07-11)✅ 完成

微调探针(init=主 300k checkpoint,bs16,k_draws=8):`pooled_add`(--pooled-adaln add,20k,零初始化叠加)与 `pooled_only`(--pooled-adaln only,40k,逐层 read 置零冻结)。

| Suite | v1 基线(单次) | pooled_add | pooled_only |
|---|---:|---:|---:|
| Spatial | 88.0 | 90.0 | 66.0 |
| Object | 98.0 | 99.0 | 86.0 |
| Goal | 90.0 | 90.0 | 73.0 |
| Long | 83.0 | 88.0 | 56.0 |
| **Avg** | **89.75** | **91.75 (+2.0)** | **70.25 (−19.5)** |

读数:
1. **pooled_only 崩塌(−19.5)**:训练侧 action loss 平台高 3×(0.049 vs 0.017)已预示;8-query 全局池化向量无法承载逐层 token 级 read 的信息。⚠️ 移除式微调仅方向参考(用户方法学标准),但方向足够明确。
2. **pooled_add +2.0,归因未定**:已删的 r0_control(40k,k=8,无 pooled)当时为 90.75 → 相对"同为 k=8 微调"的最近对照,pooled 净增益 ~+1.0,而 Long 单套噪声 ±6。**是否真提升待 B2 CI 判定。**
3. **与 ABC 的对话(候选论文结论)**:ABC 中逐层耦合(11.7%)≪ pooled(61.4%);STWAM 中逐层 read 是 zero-init 门控、video 塔未被梯度污染,ABC 的失败模式不存在,pooled 反而是信息瓶颈。若 CI 证实 add≈基线,结论为"**零初始化门控的逐层 read 已规避 ABC 报告的深耦合失败,且不可被全局池化替代**"——正反都可写。

产物:`checkpoint/ablation_pooled_{add,only}/latest.pt`、`logs/ablation_eval/pooled_{add,only}/libero/*.csv`、`logs/pooled/*_train.log`。

### B2 评测噪声 CI(2026-07-12)✅ 完成 — pooled_add 总体增益未达显著, Long 单套疑似真效应

base 与 pooled_add 各 3 次独立重复(rep1=原始单次评测;flow 采样噪声天然异种子),每 rep 400 trials:

| Suite | base mean±std (3 reps) | pooled_add mean±std | Δ |
|---|---:|---:|---:|
| Spatial | 89.00 ± 1.73 | 91.00 ± 1.00 | +2.00 |
| Object | 98.00 ± 1.00 | 96.67 ± 2.08 | −1.33 |
| Goal | 93.33 ± 3.06 | 92.00 ± 2.00 | −1.33 |
| Long | 78.00 ± 4.36 | **86.33 ± 2.89** | **+8.33** |
| **Avg** | **89.58 ± 0.29** | **91.50 ± 0.66** | **+1.92** |

统计判定(合并 1200 trials/模型):
1. **总体 Δ=+1.92,z=1.60 < 1.96 → 未达 5% 显著**(且 trials 因共享任务/init state 非独立,真实显著性更低)。按预设规则 **B3 不满足直接启动条件**。
2. **Long 单套 Δ=+8.33,z=2.67 > Bonferroni×4 临界 2.50,且三次重复方向一致(+5/+8/+12,pooled 最差 83 = base 最好 83)**——长程任务上的增益是本轮唯一疑似真效应;与"全局池化条件在多阶段任务中补充局部 read"的机制假设相容。
3. **重要副产品:基线校准。** base 三次重复的 Avg 极稳(89.58±0.29),但单套抖动大(Long ±4.4、Goal ±3.1)——论文所有单套结论必须配多次重复;历史上 rollout_ctx1 复评的 Long=77 与本轮 75/76 一致,说明原始 83 偏高。
4. **归因 confound 未拆:** pooled_add 比 base 多 20k 步 k=8 续训。已启动 `k8_ctrl`(20k, k=8, 无 pooled, 同 init)对照,3 reps 评测后与 pooled_add 配对比较,分离 [pooled connector] 与 [k8 续训] 两因子——特别看 Long。
5. 产物:`logs/ablation_eval/ci/{base,pooled_add}/rep{2,3}/`。

**B3 决策(暂缓):** 等 k8_ctrl 判定。若 Long 增益归 pooled → 小规模 from-scratch 验证(或 Long 专项);若归 k8 续训 → 改进路线变为"主 checkpoint 续训"(需诚实标注非 from-scratch),B3 从-scratch 取消。

### B2-附 k8_ctrl 归因对照(2026-07-13)✅ 完成 — 探针轮最终判定:**B3 不启动**

`k8_ctrl`(同 init、20k、k=8、无 pooled)3 次重复,与 base / pooled_add 三方对照:

| Suite | base | k8_ctrl | pooled_add |
|---|---:|---:|---:|
| Spatial | 89.00 ± 1.73 | 89.33 ± 3.06 | 91.00 ± 1.00 |
| Object | 98.00 ± 1.00 | 95.33 ± 1.15 | 96.67 ± 2.08 |
| Goal | 93.33 ± 3.06 | 89.33 ± 1.53 | 92.00 ± 2.00 |
| Long | 78.00 ± 4.36 | 84.00 ± 7.81 | 86.33 ± 2.89 |
| **Avg** | **89.58 ± 0.29** | **89.50 ± 2.50** | **91.50 ± 0.66** |

配对检验(合并 1200 trials/模型):k8_ctrl vs base Avg Δ=−0.08(z=−0.07);pooled_add vs k8_ctrl Avg Δ=+2.00(z=+1.67, p≈0.10);Long 上 base→k8 +6.0(z=1.87)、k8→pooled +2.33(z=0.80)。

**判定:**
1. **续训因子排除:** 20k 额外 k=8 训练对 Avg 增益 = 0(89.50 vs 89.58)——pooled_add 的 +1.9 不是"多训了"造成的;k-draws 续训无平均收益(kstudy +3.0 正式判死,移 appendix)。
2. **pooled 因子未达显著:** vs 正确对照 k8_ctrl,+2.00(z=1.67)差一口气;Long 分解后 pooled 的净增量只剩 +2.33(z=0.80)。**按预注册规则,B3 from-scratch 不启动**(省 6 GPU-天)。
3. **但 pooled_add 是一致的名义最优:** 每个 rep 都 ≥90.75,base 从未超过 89.75(3×3 全序,Mann-Whitney 恰 p=0.05);逐任务看 Long 增益结构健康(task0/task4 每 rep 稳定 +2~3)。0.69M 参数、零推理代价、从未变差 → **可作工程默认,但不作论文"improvement"主张**。
4. **k8_ctrl 自身高方差**(Avg ±2.5,Long ±7.8,单 rep Long 摆 14 点)是判定精度的瓶颈;若未来要坐实 pooled 因子,正路是 from-scratch 配对(各≥2 seed)而非继续堆微调 reps。
5. **connector 消融的论文写法(正式定稿):** ①逐层 zero-init 门控 read 是承重结构——pooled 替代则崩塌(70.25,−19.5);②ABC 的"pooled ≫ 逐层"在 STWAM 不复现,归因于 zero-init 门控使 ABC 的梯度污染失败模式不存在;③在其上叠加 pooled 全局条件至多带来边际增益(+2.0,z=1.67,n.s.)。与 ABC 直接对话,正结果负结果都有位置。

产物:`checkpoint/ablation_k8_ctrl/latest.pt`、`logs/ablation_eval/ci/k8_ctrl/rep{1,2,3}/`。
