# LWT-STF
The method is designed to reconstruct high-spatial and high-temporal resolution remote sensing images under complex heterogeneous landscapes and strong phenological changes.
The model has three main components:

1. Image pyramid feature decoupling
Gaussian-Laplacian pyramid decomposition is used to separate high-frequency texture features and sensor bias before deep feature mapping.

2. Wavelet-Transformer mapping
Discrete wavelet transform, frequency attention, and Transformer blocks are used to enhance high-frequency detail preservation and long-range spatiotemporal dependency modeling.

3. Object-aware spatial regularization
Object masks are used to build adaptive spatial constraints, which help reduce cross-boundary spectral confusion and local over-smoothing in heterogeneous regions.

If you find this repository useful, please cite our paper after publication.

# Environment
Recommended environment:

- Python 3.10+
- PyTorch 2.1+
- torchvision
- numpy
- scipy
- scikit-image
- tifffile
- rasterio or GDAL
- fvcore
- thop

GPU is recommended for training and inference.

# Data organization
A typical directory structure is:

../text/Datasets/train/coarse/
../text/Datasets/train/fine/
../text/Datasets/train/sam_masks/
../predict/Datasets/train/coarse/
../predict/Datasets/train/fine/
../predict/Datasets/train/sam_masks/
../result/

# SAM masks
If you need to generate object masks, please use the official Segment Anything resources:
https://github.com/facebookresearch/segment-anything
