"""Video expert = semantic-wm DiT, used *unmodified* and initialized from
`DiT-S_D96.pt`.  We only wrap it to (a) load the checkpoint and (b) expose the
internal building blocks (`patchify`, `get_cond`, `blocks`, wide head) so the
coupled MoT loop in `modeling_stwam.py` can interleave the action expert and the
joint adapter between the video DiT's own blocks — without touching its weights.
"""
from __future__ import annotations

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from ._swm.models.model import DiT


class VideoExpert(nn.Module):
    def __init__(self, in_channels: int = 96, patch_size: int = 1, dim: int = 384,
                 num_layers: int = 12, num_heads: int = 6, action_dim: int = 7,
                 max_frames: int = 16, wide_head: bool = True, decoder_dim: int = 2048,
                 num_views: int = 1, temporal_mode: str = "factored",
                 action_dropout_prob: float = 0.1) -> None:
        super().__init__()
        self.dit = DiT(
            in_channels=in_channels, patch_size=patch_size, dim=dim,
            num_layers=num_layers, num_heads=num_heads, action_dim=action_dim,
            max_frames=max_frames, wide_head=wide_head, decoder_dim=decoder_dim,
            num_views=num_views, temporal_mode=temporal_mode,
            action_dropout_prob=action_dropout_prob,
        )
        self.dim = dim
        self.num_layers = num_layers

    @property
    def blocks(self) -> nn.ModuleList:
        return self.dit.blocks

    def load_pretrained(self, state: dict[str, torch.Tensor], strict: bool = True):
        return self.dit.load_state_dict(state, strict=strict)

    # --- pieces of DiT.forward, exposed for the coupled loop ---
    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        return self.dit.patchify(x)

    def get_cond(self, t: torch.Tensor, action: torch.Tensor):
        return self.dit.get_cond(t, action)

    def head(self, x: torch.Tensor, time_cond: torch.Tensor) -> torch.Tensor:
        """Wide-head path identical to `DiT.forward` (after the block stack)."""
        dit = self.dit
        T = x.shape[1]
        if dit.wide_head:
            head_cond = x + time_cond.unsqueeze(2).unsqueeze(3)
            head_cond = dit.s_projector(F.silu(head_cond))
            head_c = head_cond.mean(dim=(2, 3))
            x = dit.head_final_layer(head_cond, head_c)
        else:
            # `c` here would need action_cond too; wide_head is the DiT-S_D96 case.
            x = dit.final_layer(x, time_cond)
        x = einops.rearrange(x, "b t h w d -> (b t) h w d")
        x = dit.unpatchify(x)
        x = einops.rearrange(x, "(b t) h w c -> b t h w c", t=T)
        return x
