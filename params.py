from __future__ import annotations

import time

import torch

from model import LWTSTF

try:
    from thop import profile
    _HAS_THOP = True
except Exception:
    profile = None
    _HAS_THOP = False


def count_params(model: torch.nn.Module) -> float:
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


def compute_fps(model: torch.nn.Module, inputs, warmup: int = 50, runs: int = 200) -> float:
    if not torch.cuda.is_available():
        return -1.0
    model.eval()
    for _ in range(warmup):
        _ = model(*inputs)
    torch.cuda.synchronize()

    start = time.time()
    with torch.no_grad():
        for _ in range(runs):
            _ = model(*inputs)
    torch.cuda.synchronize()
    end = time.time()
    return runs / max(end - start, 1e-12)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LWTSTF(in_bands=6, embed_dim=32, num_heads=4, token_stride=4).to(device)
    model.eval()

    h, w = 800, 800
    t1 = torch.randn(1, 6, h, w, device=device)
    b1 = torch.randn(1, 6, h, w, device=device)
    c2u = torch.randn(1, 6, h, w, device=device)
    t3 = torch.randn(1, 6, h, w, device=device)
    b3 = torch.randn(1, 6, h, w, device=device)
    inputs = (t1, b1, c2u, t3, b3)

    params_m = count_params(model)
    print(f"Params (M): {params_m:.3f}")

    if _HAS_THOP:
        macs, _ = profile(model, inputs=inputs, verbose=False)
        print(f"FLOPs (G): {macs / 1e9:.3f}")
    else:
        print("FLOPs (G): thop not installed")

    fps = compute_fps(model, inputs)
    if fps > 0:
        print(f"FPS: {fps:.2f}")
    else:
        print("FPS: GPU only")
