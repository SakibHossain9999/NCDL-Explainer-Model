# NCDL Explainer — Trained Model

This repository hosts the trained model weights used by the interactive NCDL Explainer notebook, a companion piece to the paper *NCDL: A Framework for Deep Learning on non-Cartesian Lattices* (Horacsek and Alim, NeurIPS 2023).

## What this is

A small autoencoder, built entirely from the real NCDL library, trained to compress and reconstruct MNIST handwritten digits. Every operation in the model, convolution, pooling, downsampling, and upsampling, runs on a quincunx (checkerboard) lattice rather than an ordinary square grid, using NCDL's real, published code.

## Which parts of NCDL this model uses

This model is built directly from the following real classes in the [NCDL library](https://github.com/jjh13/NCDL):

| Component | Source |
|---|---|
| `Lattice` | `ncdl/lattice/lattice.py` |
| `LatticeDownsample`, `LatticeUpsample` | `ncdl/nn/modules/resample.py` |
| `DoubleConv`, `OutConv` | `ncdl/modules/autoencoder.py` |
| `pad_like` | `ncdl/nn/functional/pad.py` |

No part of NCDL's own code has been copied into this repository. Install it directly from the source:
```bash
pip install git+https://github.com/jjh13/NCDL.git
```

## Architecture

A 10-layer encoder-decoder, alternating between the ordinary Cartesian grid and the quincunx lattice:

| # | Layer | Lattice | Shape (channels × height × width) |
|---|---|---|---|
| 1 | Input | Cartesian | 1 × 28 × 28 |
| 2 | Conv 1 | Cartesian | 16 × 28 × 28 |
| 3 | Downsample 1 | Cartesian → Quincunx | 16 × 14 × 14 (each of 2 cosets) |
| 4 | Conv 2 | Quincunx | 32 × 14 × 14 (each of 2 cosets) |
| 5 | Downsample 2 | Quincunx → Cartesian | 32 × 14 × 14 |
| 6 | Conv 3 (bottleneck) | Cartesian | 32 × 14 × 14 |
| 7 | Upsample 1 | Cartesian → Quincunx | 32 × 14 × 14 (each of 2 cosets) |
| 8 | Conv 4 | Quincunx | 16 × 14 × 14 (each of 2 cosets) |
| 9 | Upsample 2 | Quincunx → Cartesian | 16 × 28 × 28 |
| 10 | Conv 5 + output | Cartesian | 1 × 28 × 28 |

Total parameters: 46,689.

## Dataset

MNIST — 60,000 training images (54,000 train / 6,000 validation split) and 10,000 test images. Not included in this repository; downloaded automatically by `torchvision.datasets.MNIST(download=True)` the first time the training script runs.

## Training details

- **Task:** reconstruction (input digit → same digit)
- **Loss function:** L1 (mean absolute error)
- **Optimizer:** Adam, learning rate 0.001
- **Scheduler:** `ReduceLROnPlateau`, halving the learning rate after 3 epochs without validation improvement
- **Batch size:** 128
- **Epochs:** 20
- **Hardware:** NVIDIA RTX A6000

## Results

| Metric | Test set |
|---|---|
| L1 | *(fill in from your final run)* |
| L2 | *(fill in)* |
| PSNR | *(fill in)* |
| SSIM | *(fill in)* |

See `quantitative_results.png` for training curves and `qualitative_results.png` for example reconstructions.

## Files in this repository

- `model_best.pt` — trained weights (best validation checkpoint), used by the notebook
- `train_model.py` — full training script
- `results.py` — script generating the result images below
- `training_metrics.json` — complete per-epoch training history
- `qualitative_results.png`, `quantitative_results.png` — result images

## Using this model

```python
import torch
from train_model import SmallQuincunxAutoencoder

model = SmallQuincunxAutoencoder()
model.load_state_dict(torch.load('model_best.pt'))
model.eval()
```
