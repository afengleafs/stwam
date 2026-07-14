"""Sanity-check the local V-JEPA 2.1 ViT-L weight file."""
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
p = ROOT / "weights/vjepa/vjepa2_1_vitl_dist_vitG_384.pt"
sd = torch.load(p, map_location="cpu", weights_only=False)
print(type(sd), sd.keys() if isinstance(sd, dict) else None)
assert isinstance(sd, dict)
assert "ema_encoder" in sd or "encoder" in sd
print("V-JEPA2.1 ViT-L checkpoint OK")
