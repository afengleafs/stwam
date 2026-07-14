# STWAM — 运行命令（服务器上验证）

项目索引见根目录 [`README.md`](README.md)；论文/调研文档在 [`docs/paper/`](docs/paper/)；本地大文件清单见 [`artifacts/MANIFEST.md`](artifacts/MANIFEST.md)。后台训练脚本在 `scripts/`，根目录 `train_*_bg.sh` 为兼容包装。

仿真环境验证

libero_spatial,libero_object,libero_goal,libero_10


MUJOCO_GL=egl .venv/bin/python eval_libero.py \
    --checkpoint checkpoint/stwam_libero_ddp/latest.pt \
    --suite libero_object --n-episodes 10 --device cuda:1 \
    --output logs/eval_libero_object.csv

MUJOCO_GL=egl .venv/bin/python eval_libero.py \
      --checkpoint checkpoint/stwam_libero_ddp/latest.pt \
      --suite libero_goal --n-episodes 10 --device cuda:1 \
      --output logs/eval_libero_goal.csv

MUJOCO_GL=egl .venv/bin/python eval_libero.py \
      --checkpoint checkpoint/stwam_libero_ddp/latest.pt \
      --suite libero_10 --n-episodes 10 --device cuda:1 \
      --output logs/eval_libero_10.csv


 Full libero_spatial results (latest.pt @ step 300k, 10 tasks × 10 episodes, GPU 1):

  ┌──────────────────────┬─────────────────────────────────────────────────────────────────┐
  │ metric               │ value                                                           │
  ├──────────────────────┼─────────────────────────────────────────────────────────────────┤
  │ Overall              │ 88/100 = 88.0%                                                  │
  ├──────────────────────┼─────────────────────────────────────────────────────────────────┤
  │ Perfect tasks (100%) │ 0, 2, 3, 6, 7                                                   │
  ├──────────────────────┼─────────────────────────────────────────────────────────────────┤
  │ Weakest              │ task 5 "on the ramekin" (50%), task 4 "in the top drawer" (80%) │


=== OVERALL: 98/100 = 98.0% ===
[done] wrote logs/eval_libero_object.csv
sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute

=== OVERALL: 90/100 = 90.0% ===
[done] wrote logs/eval_libero_goal.csv
sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute

=== OVERALL: 83/100 = 83.0% ===
[done] wrote logs/eval_libero_10.csv
sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute




世界-动作模型：semantic-wm DiT-S（加载 `DiT-S_D96.pt`，结构不改）作 video expert
+ 轻量 1D action expert（从零）+ 每层 zero-init MoT adapter 耦合（FastWAM 式：
action 单向 cross-attn 读取干净历史帧，时间对齐 RoPE；训练时历史帧 teacher-forced
t=0，与推理一致；VideoExpert 不直接 action-conditioned，`observation.state` 作为
context token 被 video/action 两路读取；推理 prefill 一次 video 塔后每个 flow step
只跑 action expert）；V-JEPA 2.1 + 冻结 S-VAE adapter 提供 96 维语义 latent；按
lerobot pi05 组织。

## 0. 环境
```bash
cd stwam
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install -r requirements.txt --index-url https://mirrors.aliyun.com/pypi/simple/
```
> 代码已自带 vendored 源码：`model/_swm/`（semantic-wm：DiT/diffusion/adapter/pixel_decoder/encoders）
> 与 `model/_swm/models/encoders/_vjepa2_src/`（facebookresearch/vjepa2 主干）。无需再 clone。

## 1. 下载权重（HF）
```bash
export HF_ENDPOINT=https://hf-mirror.com  
hf download Nilaksh404/semantic-wm \
  vjepa/DiT-S_D96.pt vjepa/adapter_vjepa_image_96.pt \
  --local-dir ./weights

# V-JEPA 2.1 ViT-L 已固定从本地加载，不在训练时联网
ls -lh ./weights/vjepa/vjepa2_1_vitl_dist_vitG_384.pt

# 文本编码器：使用 FLAN-T5-Large（1024维，约 5.8GB PyTorch 权重）
# 如果之前误下了 XXL，可以先删除残留目录。
rm -rf weights/flan_t5_xxl
mkdir -p weights/flan_t5_large

HF_ENDPOINT=https://hf-mirror.com \
.venv/bin/hf download google/flan-t5-large \
  --local-dir weights/flan_t5_large \
  --include "*.json" "*.model" "*.txt" "*.safetensors" \
  --exclude "flax_model*" "tf_model*" "pytorch_model*.bin" \
  --max-workers 4
```

## 2. Checkpoint introspection（反推架构，第一步）
```bash
python -m model.checkpoint ./weights/vjepa/DiT-S_D96.pt
# 期望: in_channels=96, patch_size=1, dim=384, num_layers=12, num_heads=6,
#       wide_head=True, decoder_dim=2048, action_dim=<实测,多半7>
# objective/temporal_mode 无法从权重反推 -> 默认 ddpm(v-pred)/factored
```

## 3. 全链路验证（introspection→加载→编码→前向→采样）
```bash
python -m scripts.verify_server \
  ./weights/vjepa/DiT-S_D96.pt \
  ./weights/vjepa/adapter_vjepa_image_96.pt \
  ./weights/vjepa/vjepa2_1_vitl_dist_vitG_384.pt
# 关注:
#  [3] video DiT load: 0 missing / 0 unexpected   <- 加载即等价原 DiT
#  [4] semantic latent: (1, T, 16, 32, 96)        <- 双路视频宽度维拼接
#  [5] loss=... {loss_video, loss_action}
#  [6] sampled action chunk: (1, chunk_size, 7)
```

## 4. 纯逻辑自检（无需权重，CPU 即可）
```bash
python -m scripts.verify_local
# [2] zero-init no-op: video vs raw DiT max|diff| = 0.00e+00   <- 关键不变量
```

## 关键文件
```
model/config.py            STWAMConfig（含 launch.py 核实的默认值）
model/checkpoint.py        ckpt 解包 + introspection
model/video_expert.py      semantic-wm DiT 包装 + 精确加载（结构不改）
model/action_expert.py     轻量 1D action 专家（adaLN + self-attn + 原生 cross-attn 语言/状态）
model/mot_adapter.py       zero-init MoT adapter（action→历史帧单向 cross-attn + 3D RoPE；text→video）
model/vjepa_encoder.py     V-JEPA + 冻结 S-VAE adapter -> [B,T,16,16,96]
model/modeling_stwam.py    STWAMModel：coupled forward + 训练loss(v-pred视频+flow动作) + sample_actions
policy/stwam_policy.py     STWAMPolicy（pi05 风格：forward/select_action/queue）
model/_swm/, .../_vjepa2_src/   vendored 源码
```

## lerobot/libero 数据集下载

```bash
export HF_ENDPOINT=https://hf-mirror.com

# 下载为本地 LeRobot v3 数据集目录；train.py 用
# LeRobotDatasetMetadata("local/libero", root="libero") 离线读取这里。
mkdir -p libero

.venv/bin/hf download lerobot/libero \
  --repo-type dataset \
  --local-dir ./libero \
  --include "README.md" "meta/**" \
  --max-workers 4

.venv/bin/hf download lerobot/libero \
  --repo-type dataset \
  --local-dir ./libero \
  --include "data/**" "videos/**" \
  --max-workers 8

# 快速检查本地目录能否被当前训练脚本读取
.venv/bin/python - <<'PY'
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
meta = LeRobotDatasetMetadata("local/libero", root="libero")
print(f"fps={meta.fps} episodes={meta.total_episodes} frames={meta.total_frames}")
print(sorted(meta.features))
PY
```



## 训练接入（lerobot）
`train.py` 仿照 LeRobot 示例加载数据：
`LeRobotDatasetMetadata -> delta_timestamps -> LeRobotDataset(video_backend="pyav")`。
`delta_timestamps` 由 `STWAMConfig.observation_delta_indices/action_delta_indices`
生成；默认 observation 为 `[-1,0,1,2,3,4,5,6] / fps`，action 为
`[0..15] / fps`。

```bash
.venv/bin/python train.py \
  --dataset-root libero \
  --hf-endpoint https://hf-mirror.com \
  --batch-size 1 \
  --num-workers 0 \
  --max-steps 2
```

训练 batch：
`observation.images.image[B,T,3,256,256]` + `observation.images.image2[B,T,3,256,256]`
经 V-JEPA2.1/S-VAE 分别编码后在 latent 宽度维拼接为
`semantic_latent[B,T,16,32,96]`；`action[B,chunk,a]`；FLAN-T5-Large 预编码的
`text_embeds[B,L,1024]` + `text_mask[B,L]`；当前帧 `observation.state[B,8]`；
`action_is_pad`/`image_is_pad`。
对齐 FastWAM 默认设计，VideoExpert 不直接 action-conditioned；`observation.state`
会作为 proprio/context token append 到文本上下文，并被 video/action 两路 cross-attn 读取。
首次训练会通过 `--hf-endpoint https://hf-mirror.com` 下载
`google/flan-t5-large` 到 `--text-model-dir`，并把 LIBERO 任务文本缓存到
`--text-cache-path`（默认 `weights/flan_t5_large/libero_text_cache.pt`）；后续训练直接读缓存。

> 本机 `torchcodec` 与当前 PyTorch/FFmpeg 可能不兼容，所以训练脚本显式传
> `video_backend="pyav"` 给 LeRobotDataset。
> 默认 objective=ddpm(v-pred)、temporal_mode=factored、num_history=2、decoder_dim=2048
> —— 若 HF 附带 train config 与此不符，改 `STWAMConfig` 后重跑 introspection 校验。
