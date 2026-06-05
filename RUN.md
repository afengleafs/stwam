# STWAM — 运行命令（服务器上验证）

世界-动作模型：semantic-wm DiT-S（加载 `DiT-S_D96.pt`，结构不改）作 video expert
+ 轻量 1D action expert（从零）+ 每层 zero-init joint MoT adapter 耦合；V-JEPA 2.1
+ 冻结 S-VAE adapter 提供 96 维语义 latent；按 lerobot pi05 组织。

## 0. 环境
```bash
cd stwam
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install -r requirements.txt
```
> 代码已自带 vendored 源码：`model/_swm/`（semantic-wm：DiT/diffusion/adapter/pixel_decoder/encoders）
> 与 `model/_swm/models/encoders/_vjepa2_src/`（facebookresearch/vjepa2 主干）。无需再 clone。

## 1. 下载权重（HF）
```bash
huggingface-cli download Nilaksh404/semantic-wm \
  vjepa/DiT-S_D96.pt vjepa/adapter_vjepa_image_96.pt \
  --local-dir ./weights
# V-JEPA 2.1 主干权重首次运行自动从 dl.fbaipublicfiles.com 拉取
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
python -m scripts.verify_server ./weights/vjepa/DiT-S_D96.pt ./weights/vjepa/adapter_vjepa_image_96.pt
# 关注:
#  [3] video DiT load: 0 missing / 0 unexpected   <- 加载即等价原 DiT
#  [4] semantic latent: (1, T, 16, 16, 96)
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
model/mot_adapter.py       zero-init joint MoT adapter（branch B）
model/vjepa_encoder.py     V-JEPA + 冻结 S-VAE adapter -> [B,T,16,16,96]
model/modeling_stwam.py    STWAMModel：coupled forward + 训练loss(v-pred视频+flow动作) + sample_actions
policy/stwam_policy.py     STWAMPolicy（pi05 风格：forward/select_action/queue）
model/_swm/, .../_vjepa2_src/   vendored 源码
```

## 训练接入（lerobot）
`STWAMPolicy.forward(batch)->(loss,dict)` 期望 batch：
`video[B,3,T,256,256]`（或预计算 `semantic_latent[B,T,16,16,96]`）、`action[B,chunk,a]`、
可选 `video_action[B,T,a]`、`text_embeds[B,L,4096]`+`text_mask`（默认走预计算文本，避免常驻 UMT5-XXL）、
`observation.state`（proprio）、`action_is_pad`/`image_is_pad`。
> 默认 objective=ddpm(v-pred)、temporal_mode=factored、num_history=2、decoder_dim=2048
> —— 若 HF 附带 train config 与此不符，改 `STWAMConfig` 后重跑 introspection 校验。
```
