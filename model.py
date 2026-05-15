from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from skimage.segmentation import slic
    _HAS_SLIC = True
except Exception:
    slic = None
    _HAS_SLIC = False


def build_gaussian_kernel(device: torch.device, channels: int) -> torch.Tensor:
    kernel = torch.tensor(
        [[1, 4, 6, 4, 1],
         [4, 16, 24, 16, 4],
         [6, 24, 36, 24, 6],
         [4, 16, 24, 16, 4],
         [1, 4, 6, 4, 1]],
        dtype=torch.float32,
        device=device,
    )
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, 5, 5).repeat(channels, 1, 1, 1)
    return kernel


def gaussian_blur(x: torch.Tensor) -> torch.Tensor:
    c = x.shape[1]
    kernel = build_gaussian_kernel(x.device, c)
    return F.conv2d(x, kernel, stride=1, padding=2, groups=c)


def laplacian_texture(x: torch.Tensor) -> torch.Tensor:
    g1 = gaussian_blur(x)
    g1_up = F.interpolate(g1, size=x.shape[-2:], mode="bilinear", align_corners=False)
    return x - g1_up


def upsample_to(x: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class DWT_Haar(nn.Module):
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x01 = x[:, :, 0::2, :] / 2.0
        x02 = x[:, :, 1::2, :] / 2.0
        x1 = x01[:, :, :, 0::2]
        x2 = x02[:, :, :, 0::2]
        x3 = x01[:, :, :, 1::2]
        x4 = x02[:, :, :, 1::2]
        ll = x1 + x2 + x3 + x4
        hl = -x1 - x2 + x3 + x4
        lh = -x1 + x2 - x3 + x4
        hh = x1 - x2 - x3 + x4
        return ll, lh, hl, hh


class IWT_Haar(nn.Module):
    def forward(
        self,
        ll: torch.Tensor,
        lh: torch.Tensor,
        hl: torch.Tensor,
        hh: torch.Tensor,
    ) -> torch.Tensor:
        b, c, h, w = ll.shape
        out = torch.zeros(b, c, h * 2, w * 2, device=ll.device, dtype=ll.dtype)
        out[:, :, 0::2, 0::2] = ll - hl - lh + hh
        out[:, :, 0::2, 1::2] = ll - hl + lh - hh
        out[:, :, 1::2, 0::2] = ll + hl - lh - hh
        out[:, :, 1::2, 1::2] = ll + hl + lh + hh
        return out / 2.0


class FrequencyAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(x)


class TinyTransformer(nn.Module):
    def __init__(self, embed_dim: int = 32, num_heads: int = 4, token_stride: int = 4, mlp_ratio: int = 2) -> None:
        super().__init__()
        self.token_stride = token_stride
        self.patch_embed = nn.Conv2d(embed_dim, embed_dim, kernel_size=token_stride, stride=token_stride, bias=False)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(embed_dim * mlp_ratio, embed_dim),
        )
        self.patch_unembed = nn.ConvTranspose2d(
            embed_dim, embed_dim, kernel_size=token_stride, stride=token_stride, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.patch_embed(x)
        b, c, h, w = x.shape
        seq = x.flatten(2).transpose(1, 2)
        seq2 = self.norm1(seq)
        attn_out, _ = self.attn(seq2, seq2, seq2, need_weights=False)
        seq = seq + attn_out
        seq = seq + self.mlp(self.norm2(seq))
        x = seq.transpose(1, 2).reshape(b, c, h, w)
        x = self.patch_unembed(x)
        if x.shape[-2:] != residual.shape[-2:]:
            x = F.interpolate(x, size=residual.shape[-2:], mode="bilinear", align_corners=False)
        return x + residual


class LWTSTF(nn.Module):
    def __init__(
        self,
        in_bands: int = 6,
        embed_dim: int = 32,
        num_heads: int = 4,
        token_stride: int = 4,
    ) -> None:
        super().__init__()
        self.in_bands = in_bands
        self.dwt = DWT_Haar()
        self.iwt = IWT_Haar()

        c_in = in_bands * 5
        c_dwt = c_in * 4

        self.reduce = nn.Conv2d(c_dwt, embed_dim, kernel_size=1, bias=False)
        self.freq_attn = FrequencyAttention(embed_dim)
        self.transformer = TinyTransformer(embed_dim=embed_dim, num_heads=num_heads, token_stride=token_stride)
        self.expand = nn.Conv2d(embed_dim, c_dwt, kernel_size=1, bias=False)
        self.tail = nn.Conv2d(c_in, in_bands, kernel_size=3, padding=1, bias=True)

    def forward_features(
        self,
        t1: torch.Tensor,
        b1: torch.Tensor,
        c2u: torch.Tensor,
        t3: torch.Tensor,
        b3: torch.Tensor,
    ) -> torch.Tensor:
        xin = torch.cat([t1, b1, c2u, t3, b3], dim=1)
        ll, lh, hl, hh = self.dwt(xin)
        xdwt = torch.cat([ll, lh, hl, hh], dim=1)

        x = self.reduce(xdwt)
        x = self.freq_attn(x)
        x = self.transformer(x)
        x = self.expand(x)

        c = xin.shape[1]
        ll_hat = x[:, 0:c]
        lh_hat = x[:, c:2 * c]
        hl_hat = x[:, 2 * c:3 * c]
        hh_hat = x[:, 3 * c:4 * c]

        x_rec = self.iwt(ll_hat, lh_hat, hl_hat, hh_hat)
        delta_h = self.tail(x_rec)
        return delta_h

    def forward(
        self,
        t1: torch.Tensor,
        b1: torch.Tensor,
        c2u: torch.Tensor,
        t3: torch.Tensor,
        b3: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward_features(t1, b1, c2u, t3, b3)


@torch.no_grad()
def prepare_inputs(
    f1: torch.Tensor,
    c1: torch.Tensor,
    c2: torch.Tensor,
    f3: torch.Tensor,
    c3: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    size = f1.shape[-2:]
    c1u = upsample_to(c1, size)
    c2u = upsample_to(c2, size)
    c3u = upsample_to(c3, size)
    t1 = laplacian_texture(f1)
    t3 = laplacian_texture(f3)
    b1 = f1 - c1u
    b3 = f3 - c3u
    return t1, b1, c2u, t3, b3


@torch.no_grad()
def build_targets(f2: torch.Tensor, c2: torch.Tensor) -> torch.Tensor:
    c2u = upsample_to(c2, f2.shape[-2:])
    return f2 - c2u


def build_slic_labels(
    patch: np.ndarray,
    n_segments: int = 80,
    compactness: float = 10.0,
) -> np.ndarray:
    if not _HAS_SLIC:
        h, w = patch.shape[-2:]
        return np.zeros((h, w), dtype=np.int32)

    arr = np.asarray(patch, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4, 6):
        arr = np.moveaxis(arr, 0, -1)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    elif arr.shape[-1] > 3:
        arr = arr[..., :3]
    arr_min = arr.min()
    arr_max = arr.max()
    if arr_max > arr_min:
        arr = (arr - arr_min) / (arr_max - arr_min)
    labels = slic(arr, n_segments=n_segments, compactness=compactness, start_label=0, channel_axis=-1)
    return labels.astype(np.int32)


def object_regularization_loss(
    pred: torch.Tensor,
    slic_labels: torch.Tensor,
    sam_labels: Optional[torch.Tensor] = None,
    sigma: float = 1.0,
) -> torch.Tensor:
    """Object-aware smoothness inside shared SLIC/SAM regions."""
    offsets = [(0, 1), (1, 0), (1, 1), (1, -1)]
    total = pred.new_tensor(0.0)
    count = 0
    for dy, dx in offsets:
        y0 = max(0, dy)
        y1 = pred.shape[-2] + min(0, dy)
        x0 = max(0, dx)
        x1 = pred.shape[-1] + min(0, dx)

        a = pred[:, :, y0:y1, x0:x1]
        b = pred[:, :, y0 - dy:y1 - dy, x0 - dx:x1 - dx]
        same = slic_labels[:, y0:y1, x0:x1] == slic_labels[:, y0 - dy:y1 - dy, x0 - dx:x1 - dx]
        if sam_labels is not None:
            same = same & (sam_labels[:, y0:y1, x0:x1] == sam_labels[:, y0 - dy:y1 - dy, x0 - dx:x1 - dx])

        diff = (a - b).pow(2).sum(dim=1)
        w = math.exp(-float(dx * dx + dy * dy) / (2.0 * sigma * sigma))
        total = total + (diff * same.float() * w).mean()
        count += 1
    return total / max(count, 1)


class ReconstructionWithObjectLoss(nn.Module):
    def __init__(self, lambda_reg: float = 0.1, sigma: float = 1.0) -> None:
        super().__init__()
        self.lambda_reg = lambda_reg
        self.sigma = sigma

    def forward(
        self,
        pred_delta: torch.Tensor,
        target_delta: torch.Tensor,
        slic_labels: torch.Tensor,
        sam_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        rec = F.mse_loss(pred_delta, target_delta)
        reg = object_regularization_loss(pred_delta, slic_labels, sam_labels, sigma=self.sigma)
        total = rec + self.lambda_reg * reg
        return {"loss": total, "rec_loss": rec, "reg_loss": reg}
