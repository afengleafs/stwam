"""DiT-S_D96 checkpoint introspection.

The HF weights (`Nilaksh404/semantic-wm/vjepa/DiT-S_D96.pt`) ship without config
metadata, so we recover the architecture from weight shapes and record the
ambiguous fields (`objective`, `temporal_mode`) explicitly (they cannot be read
from a state_dict — factored/joint share the same parameter shapes).
"""
from __future__ import annotations

from typing import Any

import torch


def load_raw_state_dict(path: str) -> dict[str, torch.Tensor]:
    """Load and unwrap common checkpoint wrappers (ema / model / state_dict)."""
    obj = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("ema", "model", "state_dict", "module", "net"):
        if isinstance(obj, dict) and key in obj and isinstance(obj[key], dict):
            obj = obj[key]
    if not isinstance(obj, dict):
        raise ValueError(f"Unexpected checkpoint object: {type(obj)}")
    # strip common prefixes
    def strip(sd):
        for pfx in ("module.", "model.", "_orig_mod."):
            if all(k.startswith(pfx) for k in sd):
                sd = {k[len(pfx):]: v for k, v in sd.items()}
        return sd
    return strip(obj)


def introspect(state: dict[str, torch.Tensor], head_dim: int = 64) -> dict[str, Any]:
    """Infer DiT hyper-parameters from weight shapes."""
    info: dict[str, Any] = {}
    if "x_proj.weight" in state:
        w = state["x_proj.weight"]          # [dim, in_ch, p, p]
        info["dim"] = int(w.shape[0])
        info["in_channels"] = int(w.shape[1])
        info["patch_size"] = int(w.shape[2])
    if "action_embedder.weight" in state:
        info["action_dim"] = int(state["action_embedder.weight"].shape[1])
    n_layers = 0
    for k in state:
        if k.startswith("blocks."):
            n_layers = max(n_layers, int(k.split(".")[1]) + 1)
    info["num_layers"] = n_layers
    info["wide_head"] = "s_projector.weight" in state
    if info["wide_head"]:
        info["decoder_dim"] = int(state["s_projector.weight"].shape[0])
    info["num_heads"] = info.get("dim", head_dim * 6) // head_dim
    # ambiguous fields (not in state_dict) -> caller must set from train config / defaults
    info["objective"] = None      # default "ddpm" (v-pred) per launch.py
    info["temporal_mode"] = None  # default "factored" per launch.py
    return info


if __name__ == "__main__":
    import sys, json
    sd = load_raw_state_dict(sys.argv[1])
    print(f"# {len(sd)} tensors")
    print(json.dumps(introspect(sd), indent=2))
