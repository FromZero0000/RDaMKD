# -*- coding: utf-8 -*-
"""Shared utilities for structure-based knowledge distillation."""

from typing import List, Optional, Sequence

import torch
import torch.nn.functional as F


EPS = 1.0e-12


def feature_list(feat) -> List[torch.Tensor]:
    """Return a stable shallow-to-deep feature list plus optional pooled feature."""
    if isinstance(feat, torch.Tensor):
        return [feat]

    if isinstance(feat, (list, tuple)):
        return [x for x in feat if isinstance(x, torch.Tensor)]

    if isinstance(feat, dict):
        out: List[torch.Tensor] = []
        for key in ("feats", "features", "feature"):
            if key in feat:
                value = feat[key]
                if isinstance(value, torch.Tensor):
                    out.append(value)
                elif isinstance(value, (list, tuple)):
                    out.extend([x for x in value if isinstance(x, torch.Tensor)])
                break

        if "pooled_feat" in feat and isinstance(feat["pooled_feat"], torch.Tensor):
            out.append(feat["pooled_feat"])

        if len(out) == 0:
            for value in feat.values():
                if isinstance(value, torch.Tensor):
                    out.append(value)
        return out

    return []


def resolve_layer_ids(num_layers: int, layer_ids: Sequence[int]) -> List[int]:
    if num_layers <= 0:
        return []
    resolved = []
    for idx in layer_ids:
        i = int(idx)
        if i < 0:
            i = num_layers + i
        i = max(0, min(num_layers - 1, i))
        if i not in resolved:
            resolved.append(i)
    return resolved


def normalize_layer_weights(num_layers: int, weights: Sequence[float], device) -> torch.Tensor:
    if num_layers <= 0:
        return torch.empty(0, device=device)
    if weights is None or len(weights) != num_layers:
        return torch.ones(num_layers, device=device) / float(num_layers)
    w = torch.tensor(list(weights), dtype=torch.float32, device=device)
    return w / w.sum().clamp_min(EPS)


def gap(feat: torch.Tensor) -> torch.Tensor:
    if feat.dim() == 4:
        return F.adaptive_avg_pool2d(feat, 1).flatten(1)
    if feat.dim() == 3:
        return feat.mean(dim=-1)
    if feat.dim() == 2:
        return feat
    return feat.flatten(1)


def pairwise_cosine(z: torch.Tensor) -> torch.Tensor:
    z = F.normalize(z, p=2, dim=1, eps=1.0e-8)
    return z @ z.t()


def pairwise_distance(z: torch.Tensor) -> torch.Tensor:
    return 1.0 - pairwise_cosine(z)


def positive_mean_normalize(
    v: torch.Tensor,
    min_value: float = 0.0,
    max_value: Optional[float] = 3.0,
) -> torch.Tensor:
    v = v.detach().float().clamp_min(0.0)
    if v.numel() == 0:
        return v
    v = v / v.mean().clamp_min(EPS)
    if max_value is not None and max_value > 0:
        v = v.clamp(max=float(max_value))
    if min_value > 0:
        v = v.clamp_min(float(min_value))
    return v


@torch.no_grad()
def mst_edges(distance: torch.Tensor):
    """Prim MST on a dense pairwise distance matrix. Returns edge indices [2, E]."""
    n = int(distance.size(0))
    if n <= 1:
        return None

    d = distance.detach().clone()
    inf = torch.tensor(float("inf"), device=d.device, dtype=d.dtype)
    d.fill_diagonal_(inf)

    visited = torch.zeros(n, dtype=torch.bool, device=d.device)
    visited[0] = True
    src, dst = [], []

    for _ in range(n - 1):
        candidate = d.clone()
        candidate[~visited, :] = inf
        candidate[:, visited] = inf
        flat_idx = torch.argmin(candidate)
        best = candidate.reshape(-1)[flat_idx]
        if not torch.isfinite(best):
            break
        i = torch.div(flat_idx, n, rounding_mode="floor")
        j = flat_idx % n
        src.append(i)
        dst.append(j)
        visited[j] = True

    if len(src) == 0:
        return None
    return torch.stack([torch.stack(src), torch.stack(dst)], dim=0).long()
