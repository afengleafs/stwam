"""Full server-side verification (needs GPU + the two HF checkpoints; V-JEPA
weights auto-download on first run).

Usage:
    python -m scripts.verify_server  <DiT-S_D96.pt>  <adapter_vjepa_image_96.pt>
"""
import sys

import torch

from model import checkpoint as ck
from model.config import STWAMConfig
from model.modeling_stwam import STWAMModel
from model.vjepa_encoder import VJEPASemanticEncoder


def main() -> None:
    dit_ckpt, adapter_ckpt = sys.argv[1], sys.argv[2]
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. introspection
    sd = ck.load_raw_state_dict(dit_ckpt)
    info = ck.introspect(sd)
    print(f"[1] introspect ({len(sd)} tensors): {info}")

    cfg = STWAMConfig(adapter_ckpt=adapter_ckpt, video_dit_ckpt=dit_ckpt, device=dev)
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

    # 4. encode real-ish frames
    B, T = 1, cfg.n_frames
    video = torch.rand(B, 3, T, cfg.vjepa_input_size, cfg.vjepa_input_size, device=dev)
    z = enc.encode(video)
    print(f"[4] semantic latent: {tuple(z.shape)}  (expect (1, {T}, 16, 16, 96))")

    # 5. training forward + 6. action sampling
    action = torch.randn(B, cfg.chunk_size, cfg.action_dim, device=dev)
    va = torch.zeros(B, T, cfg.action_dim, device=dev)
    loss, parts = model.training_loss(z, action, va)
    print(f"[5] loss={float(loss):.4f} {parts}")
    chunk = model.sample_actions(z[:, : cfg.num_history],
                                 torch.zeros(B, cfg.num_history, cfg.action_dim, device=dev))
    print(f"[6] sampled action chunk: {tuple(chunk.shape)}")
    print("SERVER VERIFY OK")


if __name__ == "__main__":
    main()
