from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np


def compute_positions(length: int, patch_size: int, stride: int) -> List[int]:
    if patch_size >= length:
        return [0]
    positions = list(range(0, length - patch_size + 1, stride))
    if positions[-1] != length - patch_size:
        positions.append(length - patch_size)
    return positions


def extract_patches(arr: np.ndarray, patch_size: int, stride: int) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    c, h, w = arr.shape
    ys = compute_positions(h, patch_size, stride)
    xs = compute_positions(w, patch_size, stride)

    patches, coords = [], []
    for y in ys:
        for x in xs:
            patches.append(arr[:, y:y + patch_size, x:x + patch_size].astype(np.float32))
            coords.append((y, x))
    return np.stack(patches, axis=0), coords


def fold_patches(
    patches: np.ndarray,
    coords: Sequence[Tuple[int, int]],
    out_h: int,
    out_w: int,
) -> np.ndarray:
    patches = np.asarray(patches)
    if patches.ndim == 3:
        patches = patches[:, np.newaxis, ...]
    n, c, ph, pw = patches.shape
    acc = np.zeros((c, out_h, out_w), dtype=np.float32)
    cnt = np.zeros((1, out_h, out_w), dtype=np.float32)

    for idx, (y, x) in enumerate(coords):
        acc[:, y:y + ph, x:x + pw] += patches[idx]
        cnt[:, y:y + ph, x:x + pw] += 1.0

    cnt[cnt == 0] = 1.0
    return acc / cnt
