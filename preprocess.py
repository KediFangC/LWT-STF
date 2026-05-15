from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

from io_utils import list_rasters


def _build_records(
    coarse_files: List[str],
    fine_files: List[str],
    sam_files: Optional[List[str]] = None,
) -> List[Dict]:
    if len(coarse_files) != len(fine_files):
        raise ValueError("Coarse and fine image counts do not match.")
    if len(coarse_files) < 3:
        raise ValueError("At least 3 time points are required.")

    records: List[Dict] = []
    for k in range(1, len(coarse_files) - 1):
        rec = {
            "id": f"pair_{k + 1:03d}",
            "coarse_prev": coarse_files[k - 1],
            "coarse_tgt": coarse_files[k],
            "coarse_next": coarse_files[k + 1],
            "fine_prev": fine_files[k - 1],
            "fine_tgt": fine_files[k],
            "fine_next": fine_files[k + 1],
        }
        if sam_files is not None and len(sam_files) == len(fine_files):
            rec["sam_tgt"] = sam_files[k]
        records.append(rec)
    return records


def build_train_predict_index(
    coarse_dir: str,
    fine_dir: str,
    out_dir: str,
    sam_mask_dir: Optional[str] = None,
) -> None:
    """Build STF triplet indices.

    sam_mask_dir is optional. Each mask should be a single-band tif where
    different integer values (e.g. 1,2,3,4,...) indicate different object regions.
    The file order should match fine_dir.
    """
    os.makedirs(out_dir, exist_ok=True)

    coarse_files = list_rasters(coarse_dir)
    fine_files = list_rasters(fine_dir)
    sam_files = list_rasters(sam_mask_dir) if sam_mask_dir else None

    records = _build_records(coarse_files, fine_files, sam_files)

    with open(Path(out_dir) / "train_index.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    with open(Path(out_dir) / "predict_index.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    print(f"Records: {len(records)}")
    print(f"Saved: {Path(out_dir) / 'train_index.json'}")
    print(f"Saved: {Path(out_dir) / 'predict_index.json'}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--coarse_dir", required=True)
    parser.add_argument("--fine_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--sam_mask_dir", default=None)
    args = parser.parse_args()

    build_train_predict_index(
        coarse_dir=args.coarse_dir,
        fine_dir=args.fine_dir,
        out_dir=args.out_dir,
        sam_mask_dir=args.sam_mask_dir,
    )
