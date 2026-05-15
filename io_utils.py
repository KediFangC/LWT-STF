from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import rasterio
    _HAS_RASTERIO = True
except Exception:
    rasterio = None
    _HAS_RASTERIO = False

try:
    from osgeo import gdal
    _HAS_GDAL = True
except Exception:
    gdal = None
    _HAS_GDAL = False


def list_rasters(folder: str | Path, suffixes: Tuple[str, ...] = (".tif", ".tiff")) -> List[str]:
    folder = Path(folder)
    files = [str(p) for p in folder.iterdir() if p.suffix.lower() in suffixes]
    files.sort()
    return files


def read_raster(path: str | Path) -> Tuple[np.ndarray, Dict]:
    path = str(path)
    if _HAS_RASTERIO:
        with rasterio.open(path) as ds:
            arr = ds.read().astype(np.float32)
            meta = ds.meta.copy()
            meta["transform"] = ds.transform
            meta["crs"] = ds.crs
            return arr, meta

    if _HAS_GDAL:
        ds = gdal.Open(path)
        if ds is None:
            raise FileNotFoundError(f"Cannot open {path}")
        bands = ds.RasterCount
        height = ds.RasterYSize
        width = ds.RasterXSize
        arr = np.zeros((bands, height, width), dtype=np.float32)
        for i in range(bands):
            arr[i] = ds.GetRasterBand(i + 1).ReadAsArray().astype(np.float32)
        meta = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": bands,
            "dtype": "float32",
            "transform": ds.GetGeoTransform(),
            "crs": ds.GetProjection(),
        }
        return arr, meta

    raise ImportError("rasterio or GDAL is required to read rasters.")


def write_raster(
    path: str | Path,
    arr: np.ndarray,
    reference_meta: Optional[Dict] = None,
    dtype: str = "float32",
) -> None:
    path = str(path)
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]

    bands, height, width = arr.shape
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if _HAS_RASTERIO:
        meta = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": bands,
            "dtype": dtype,
        }
        if reference_meta is not None:
            for key in ("transform", "crs", "nodata"):
                if key in reference_meta:
                    meta[key] = reference_meta[key]
        with rasterio.open(path, "w", **meta) as ds:
            ds.write(arr.astype(dtype))
        return

    if _HAS_GDAL:
        dtype_map = {
            "uint8": gdal.GDT_Byte,
            "uint16": gdal.GDT_UInt16,
            "int16": gdal.GDT_Int16,
            "float32": gdal.GDT_Float32,
            "float64": gdal.GDT_Float64,
        }
        driver = gdal.GetDriverByName("GTiff")
        ds = driver.Create(path, width, height, bands, dtype_map.get(dtype, gdal.GDT_Float32))
        if reference_meta is not None:
            if "transform" in reference_meta and reference_meta["transform"] is not None:
                transform = reference_meta["transform"]
                if isinstance(transform, tuple):
                    ds.SetGeoTransform(transform)
            if "crs" in reference_meta and reference_meta["crs"] is not None:
                crs = reference_meta["crs"]
                if not isinstance(crs, str):
                    crs = str(crs)
                ds.SetProjection(crs)
        for i in range(bands):
            ds.GetRasterBand(i + 1).WriteArray(arr[i].astype(dtype))
        ds = None
        return

    raise ImportError("rasterio or GDAL is required to write rasters.")
