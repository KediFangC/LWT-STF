from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from io_utils import read_raster
from model import (
    LWTSTF,
    ReconstructionWithObjectLoss,
    build_slic_labels,
    build_targets,
    prepare_inputs,
)


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def random_crop(arr: np.ndarray, patch_size: int) -> np.ndarray:
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    _, h, w = arr.shape
    if h <= patch_size or w <= patch_size:
        out = np.zeros((arr.shape[0], patch_size, patch_size), dtype=np.float32)
        out[:, :h, :w] = arr
        return out
    y = np.random.randint(0, h - patch_size + 1)
    x = np.random.randint(0, w - patch_size + 1)
    return arr[:, y:y + patch_size, x:x + patch_size]


def center_crop(arr: np.ndarray, patch_size: int) -> np.ndarray:
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    _, h, w = arr.shape
    out = np.zeros((arr.shape[0], patch_size, patch_size), dtype=np.float32)
    y = max((h - patch_size) // 2, 0)
    x = max((w - patch_size) // 2, 0)
    crop = arr[:, y:y + min(h, patch_size), x:x + min(w, patch_size)]
    out[:, :crop.shape[1], :crop.shape[2]] = crop
    return out


class TripletPatchDataset(Dataset):
    """Triplet patch dataset for STF.

    SAM masks are optional single-band tif label maps. Different values indicate
    different object regions. Equality of labels is used in the object loss.
    """
    def __init__(
        self,
        records: List[Dict],
        patch_size: int = 128,
        patches_per_triplet: int = 64,
        train: bool = True,
        slic_segments: int = 80,
        slic_compactness: float = 10.0,
    ) -> None:
        self.records = records
        self.patch_size = patch_size
        self.patches_per_triplet = patches_per_triplet
        self.train = train
        self.slic_segments = slic_segments
        self.slic_compactness = slic_compactness
        self.cache: Dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.records) * self.patches_per_triplet

    def _load(self, path: str) -> np.ndarray:
        if path not in self.cache:
            self.cache[path], _ = read_raster(path)
        return self.cache[path]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec = self.records[idx // self.patches_per_triplet]

        f1 = self._load(rec["fine_prev"])
        f2 = self._load(rec["fine_tgt"])
        f3 = self._load(rec["fine_next"])
        c1 = self._load(rec["coarse_prev"])
        c2 = self._load(rec["coarse_tgt"])
        c3 = self._load(rec["coarse_next"])

        crop_fn = random_crop if self.train else center_crop
        f1 = crop_fn(f1, self.patch_size)
        f2 = crop_fn(f2, self.patch_size)
        f3 = crop_fn(f3, self.patch_size)

        if self.train:
            c_patch = max(self.patch_size // 20, 8)
        else:
            c_patch = max(self.patch_size // 20, 8)

        c1 = crop_fn(c1, c_patch)
        c2 = crop_fn(c2, c_patch)
        c3 = crop_fn(c3, c_patch)

        if "sam_tgt" in rec and rec["sam_tgt"]:
            sam = self._load(rec["sam_tgt"])
            sam = crop_fn(sam, self.patch_size)[0]
            sam = np.rint(sam).astype(np.int64)
        else:
            sam = None

        slic = build_slic_labels(f2, n_segments=self.slic_segments, compactness=self.slic_compactness)

        f1_t = torch.from_numpy(f1)
        f2_t = torch.from_numpy(f2)
        f3_t = torch.from_numpy(f3)
        c1_t = torch.from_numpy(c1)
        c2_t = torch.from_numpy(c2)
        c3_t = torch.from_numpy(c3)

        with torch.no_grad():
            t1, b1, c2u, t3, b3 = prepare_inputs(
                f1_t.unsqueeze(0), c1_t.unsqueeze(0), c2_t.unsqueeze(0), f3_t.unsqueeze(0), c3_t.unsqueeze(0)
            )
            target_delta = build_targets(f2_t.unsqueeze(0), c2_t.unsqueeze(0))

        item = {
            "t1": t1.squeeze(0).float(),
            "b1": b1.squeeze(0).float(),
            "c2u": c2u.squeeze(0).float(),
            "t3": t3.squeeze(0).float(),
            "b3": b3.squeeze(0).float(),
            "target_delta": target_delta.squeeze(0).float(),
            "slic_labels": torch.from_numpy(slic.astype(np.int64)),
        }
        if sam is not None:
            item["sam_labels"] = torch.from_numpy(sam)
        return item


def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def r2_score_np(pred: np.ndarray, target: np.ndarray) -> float:
    y = target.reshape(-1)
    x = pred.reshape(-1)
    ss_res = np.sum((y - x) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    if ss_tot <= 1e-12:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def train_model(
    index_json: str,
    model_dir: str,
    epochs: int = 100,
    patch_size: int = 128,
    batch_size: int = 8,
    val_ratio: float = 0.2,
    lr: float = 1e-4,
    lambda_reg: float = 0.1,
    patches_per_triplet: int = 64,
    seed: int = 42,
    num_workers: int = 0,
    embed_dim: int = 32,
    num_heads: int = 4,
    token_stride: int = 4,
) -> Dict:
    seed_everything(seed)
    os.makedirs(model_dir, exist_ok=True)

    with open(index_json, "r", encoding="utf-8") as f:
        records = json.load(f)

    train_records, val_records = train_test_split(records, test_size=val_ratio, random_state=seed, shuffle=True)

    train_ds = TripletPatchDataset(train_records, patch_size=patch_size, patches_per_triplet=patches_per_triplet, train=True)
    val_ds = TripletPatchDataset(val_records, patch_size=patch_size, patches_per_triplet=max(patches_per_triplet // 4, 8), train=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LWTSTF(in_bands=6, embed_dim=embed_dim, num_heads=num_heads, token_stride=token_stride).to(device)
    criterion = ReconstructionWithObjectLoss(lambda_reg=lambda_reg)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    best_val = float("inf")
    history = {"train_loss": [], "val_loss": [], "val_rmse": [], "val_r2": []}

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []

        for batch in train_loader:
            t1 = batch["t1"].to(device, non_blocking=True)
            b1 = batch["b1"].to(device, non_blocking=True)
            c2u = batch["c2u"].to(device, non_blocking=True)
            t3 = batch["t3"].to(device, non_blocking=True)
            b3 = batch["b3"].to(device, non_blocking=True)
            target_delta = batch["target_delta"].to(device, non_blocking=True)
            slic_labels = batch["slic_labels"].to(device, non_blocking=True)
            sam_labels = batch.get("sam_labels")
            if sam_labels is not None:
                sam_labels = sam_labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                pred_delta = model(t1, b1, c2u, t3, b3)
                losses = criterion(pred_delta, target_delta, slic_labels, sam_labels)
            scaler.scale(losses["loss"]).backward()
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(float(losses["loss"].detach().cpu().item()))

        scheduler.step()

        model.eval()
        val_losses, preds, tgts = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                t1 = batch["t1"].to(device, non_blocking=True)
                b1 = batch["b1"].to(device, non_blocking=True)
                c2u = batch["c2u"].to(device, non_blocking=True)
                t3 = batch["t3"].to(device, non_blocking=True)
                b3 = batch["b3"].to(device, non_blocking=True)
                target_delta = batch["target_delta"].to(device, non_blocking=True)
                slic_labels = batch["slic_labels"].to(device, non_blocking=True)
                sam_labels = batch.get("sam_labels")
                if sam_labels is not None:
                    sam_labels = sam_labels.to(device, non_blocking=True)

                pred_delta = model(t1, b1, c2u, t3, b3)
                losses = criterion(pred_delta, target_delta, slic_labels, sam_labels)
                val_losses.append(float(losses["loss"].detach().cpu().item()))
                preds.append(pred_delta.cpu().numpy())
                tgts.append(target_delta.cpu().numpy())

        pred_np = np.concatenate(preds, axis=0)
        tgt_np = np.concatenate(tgts, axis=0)
        val_rmse = rmse(pred_np, tgt_np)
        val_r2 = r2_score_np(pred_np, tgt_np)
        train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        val_loss = float(np.mean(val_losses)) if val_losses else 0.0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_rmse"].append(val_rmse)
        history["val_r2"].append(val_r2)

        print(
            f"Epoch {epoch:03d}/{epochs:03d} | "
            f"train {train_loss:.6f} | val {val_loss:.6f} | "
            f"rmse {val_rmse:.6f} | r2 {val_r2:.4f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": {
                        "in_bands": 6,
                        "embed_dim": embed_dim,
                        "num_heads": num_heads,
                        "token_stride": token_stride,
                    },
                    "history": history,
                },
                Path(model_dir) / "best_model.pth",
            )

    with open(Path(model_dir) / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    return history


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--index_json", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patch_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lambda_reg", type=float, default=0.1)
    parser.add_argument("--patches_per_triplet", type=int, default=64)
    args = parser.parse_args()

    train_model(
        index_json=args.index_json,
        model_dir=args.model_dir,
        epochs=args.epochs,
        patch_size=args.patch_size,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        lr=args.lr,
        lambda_reg=args.lambda_reg,
        patches_per_triplet=args.patches_per_triplet,
    )
