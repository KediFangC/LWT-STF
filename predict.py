from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from io_utils import read_raster, write_raster
from patch_utils import extract_patches, fold_patches
from model import LWTSTF, prepare_inputs


def load_checkpoint(ckpt_path: str, device: torch.device) -> LWTSTF:
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["config"]
    model = LWTSTF(
        in_bands=cfg["in_bands"],
        embed_dim=cfg["embed_dim"],
        num_heads=cfg["num_heads"],
        token_stride=cfg["token_stride"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def compute_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    diff = pred.astype(np.float32) - target.astype(np.float32)
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae = float(np.mean(np.abs(diff)))
    y = target.reshape(-1)
    x = pred.reshape(-1)
    ss_res = np.sum((y - x) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
    return {"rmse": rmse, "mae": mae, "r2": r2}


@torch.no_grad()
def predict_one(
    model: LWTSTF,
    rec: Dict,
    patch_size: int,
    stride: int,
    device: torch.device,
) -> Tuple[np.ndarray, Dict]:
    f1, ref_meta = read_raster(rec["fine_prev"])
    f3, _ = read_raster(rec["fine_next"])
    c1, _ = read_raster(rec["coarse_prev"])
    c2, _ = read_raster(rec["coarse_tgt"])
    c3, _ = read_raster(rec["coarse_next"])
    f2, _ = read_raster(rec["fine_tgt"])

    h, w = f1.shape[-2:]
    f1_patches, coords = extract_patches(f1, patch_size, stride)
    f3_patches, _ = extract_patches(f3, patch_size, stride)

    c_patch_size = max(patch_size // 20, 8)
    c1_patches, _ = extract_patches(c1, c_patch_size, max(c_patch_size // 2, 1))
    c2_patches, _ = extract_patches(c2, c_patch_size, max(c_patch_size // 2, 1))
    c3_patches, _ = extract_patches(c3, c_patch_size, max(c_patch_size // 2, 1))

    n = min(len(coords), len(c1_patches), len(c2_patches), len(c3_patches))
    coords = coords[:n]
    preds = []

    for i in range(n):
        f1_t = torch.from_numpy(f1_patches[i:i + 1]).to(device)
        f3_t = torch.from_numpy(f3_patches[i:i + 1]).to(device)
        c1_t = torch.from_numpy(c1_patches[i:i + 1]).to(device)
        c2_t = torch.from_numpy(c2_patches[i:i + 1]).to(device)
        c3_t = torch.from_numpy(c3_patches[i:i + 1]).to(device)

        t1, b1, c2u, t3, b3 = prepare_inputs(f1_t, c1_t, c2_t, f3_t, c3_t)
        delta = model(t1, b1, c2u, t3, b3)
        pred = c2u + delta
        preds.append(pred.squeeze(0).cpu().numpy())

    pred_full = fold_patches(np.stack(preds, axis=0), coords, h, w)
    metrics = compute_metrics(pred_full, f2)
    return pred_full, {"meta": ref_meta, **metrics}


def run_predict(
    index_json: str,
    ckpt_path: str,
    out_dir: str,
    patch_size: int = 128,
    stride: int = 64,
) -> List[Dict]:
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint(ckpt_path, device)

    with open(index_json, "r", encoding="utf-8") as f:
        records = json.load(f)

    all_metrics: List[Dict] = []
    for rec in records:
        pred, info = predict_one(model, rec, patch_size, stride, device)
        out_path = Path(out_dir) / f"{rec['id']}_pred.tif"
        write_raster(out_path, pred, reference_meta=info["meta"], dtype="float32")
        rec_metrics = {"id": rec["id"], "rmse": info["rmse"], "mae": info["mae"], "r2": info["r2"]}
        all_metrics.append(rec_metrics)
        print(f"{rec['id']} | rmse {info['rmse']:.6f} | r2 {info['r2']:.4f}")

    with open(Path(out_dir) / "predict_metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    return all_metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--index_json", required=True)
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--patch_size", type=int, default=128)
    parser.add_argument("--stride", type=int, default=64)
    args = parser.parse_args()

    run_predict(
        index_json=args.index_json,
        ckpt_path=args.ckpt_path,
        out_dir=args.out_dir,
        patch_size=args.patch_size,
        stride=args.stride,
    )
