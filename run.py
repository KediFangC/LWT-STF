from __future__ import annotations

import os

from preprocess import build_train_predict_index
from predict import run_predict
from train import train_model


coarse_dir = "./Datasets/LIAHB/M"
fine_dir = "./Datasets/LIAHB/L"
sam_mask_dir = None  # optional single-band tif label maps

work_dir = "./Datasets/LIAHB/stf_run"
index_dir = os.path.join(work_dir, "index")
model_dir = os.path.join(work_dir, "model")
result_dir = os.path.join(work_dir, "result")

patch_size = 128
stride = 64
epochs = 100
batch_size = 8
patches_per_triplet = 64

if __name__ == "__main__":
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(index_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    build_train_predict_index(
        coarse_dir=coarse_dir,
        fine_dir=fine_dir,
        out_dir=index_dir,
        sam_mask_dir=sam_mask_dir,
    )

    train_model(
        index_json=os.path.join(index_dir, "train_index.json"),
        model_dir=model_dir,
        epochs=epochs,
        patch_size=patch_size,
        batch_size=batch_size,
        patches_per_triplet=patches_per_triplet,
    )

    run_predict(
        index_json=os.path.join(index_dir, "predict_index.json"),
        ckpt_path=os.path.join(model_dir, "best_model.pth"),
        out_dir=result_dir,
        patch_size=patch_size,
        stride=stride,
    )
