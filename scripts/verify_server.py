"""Full server-side verification (needs GPU + local V-JEPA/DiT/S-VAE weights).

Usage:
    python -m scripts.verify_server  <DiT-S_D96.pt>  <adapter_vjepa_image_96.pt>  <vjepa2_1_vitl_dist_vitG_384.pt>
"""
import sys

import torch

from model import checkpoint as ck
from model.config import STWAMConfig
from model.modeling_stwam import STWAMModel
from model.vjepa_encoder import VJEPASemanticEncoder


def main() -> None:
    dit_ckpt, adapter_ckpt = sys.argv[1], sys.argv[2]
    vjepa2_ckpt = sys.argv[3] if len(sys.argv) > 3 else "weights/vjepa/vjepa2_1_vitl_dist_vitG_384.pt"
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. introspection
    sd = ck.load_raw_state_dict(dit_ckpt)
    info = ck.introspect(sd)
    print(f"[1] introspect ({len(sd)} tensors): {info}")

    cfg = STWAMConfig(
        adapter_ckpt=adapter_ckpt, video_dit_ckpt=dit_ckpt,
        vjepa2_ckpt=vjepa2_ckpt, device=dev, num_views=2, proprio_dim=8,
    )
    cfg.apply_introspection(info)
    cfg.objective, cfg.temporal_mode = "ddpm", "factored"   # launch.py defaults
    cfg.action_layers = cfg.num_layers                       # keep adapters aligned

    # 2. encoder + model
    enc = VJEPASemanticEncoder(cfg).to(dev)
    model = STWAMModel(cfg, vjepa_encoder=enc).to(dev)

    # 3. load pretrained video DiT (expect exact match)
    missing, unexpected = model.video.load_pretrained(sd, strict=False)
    print(f"[3] video DiT load: {len(missing)} missing / {len(unexpected)} unexpected "
          f"(expect 0 / 0)")

    # 4. encode real-ish two-view frames
    B, T = 1, cfg.n_frames
    video1 = torch.rand(B, 3, T, cfg.vjepa_input_size, cfg.vjepa_input_size, device=dev)
    video2 = torch.rand(B, 3, T, cfg.vjepa_input_size, cfg.vjepa_input_size, device=dev)
    z = torch.cat([enc.encode(video1), enc.encode(video2)], dim=3)
    print(f"[4] semantic latent: {tuple(z.shape)}  (expect (1, {T}, 16, 32, 96))")

    # 5. training forward + 6. action sampling
    action = torch.randn(B, cfg.chunk_size, cfg.action_dim, device=dev)
    ctx = torch.randn(B, 5, cfg.text_dim, device=dev)
    ctx_mask = torch.ones(B, 5, dtype=torch.bool, device=dev)
    state = torch.randn(B, cfg.proprio_dim, device=dev)
    ctx, ctx_mask = model.build_context(ctx, ctx_mask, state)
    loss, parts = model.training_loss(z, action, ctx, ctx_mask)
    print(f"[5] loss={float(loss):.4f} {parts}")
    chunk = model.sample_actions(z[:, : cfg.num_history], ctx, ctx_mask)
    print(f"[6] sampled action chunk: {tuple(chunk.shape)}")
    print("SERVER VERIFY OK")


if __name__ == "__main__":
    main()
