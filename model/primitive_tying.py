from typing import List

import torch

PRIMITIVES: List[str] = ["REACH", "PRESHAPE", "GRASP", "CARRY", "REORIENT", "PUSH", "RELEASE"]
SLOT_NAMES: List[str] = ["translation", "orientation", "gripper"]
DEFAULT_SLOT_MASK: List[List[int]] = [
    [1, 1, 0],  # REACH
    [0, 0, 1],  # PRESHAPE
    [0, 0, 1],  # GRASP
    [1, 0, 0],  # CARRY
    [0, 1, 0],  # REORIENT
    [1, 0, 0],  # PUSH
    [0, 0, 1],  # RELEASE
]


def frames_to_tokens(x: torch.Tensor, n_tokens: int) -> torch.Tensor:
    """Per-frame labels [B, H] -> per-token labels [B, n_tokens]."""
    if x.dim() != 2:
        raise ValueError(f"expected [B, H], got {tuple(x.shape)}")
    H = x.shape[1]
    centre = ((torch.arange(n_tokens, device=x.device, dtype=torch.float32) + 0.5)
              * H / n_tokens).long().clamp_(0, H - 1)
    return x[:, centre]
