# gsplat-triton

[![Core Tests.](https://github.com/hieu1999210/gsplat-triton/actions/workflows/core_tests.yml/badge.svg?branch=main)](https://github.com/hieu1999210/gsplat-triton/actions/workflows/core_tests.yml)
[![Docs](https://github.com/hieu1999210/gsplat-triton/actions/workflows/doc.yml/badge.svg?branch=main)](https://github.com/hieu1999210/gsplat-triton/actions/workflows/doc.yml)


This project is a [Triton](https://triton-lang.org/main/index.html) implementation of [gsplat](http://www.gsplat.studio/), a CUDA accelerated implemetation of [3D Gaussian Splatting for Real-Time Rendering of Radiance Fields](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/). 

The goal of this project is to make 3D Gaussian Splatting research more accessible to ML researchers by reimplementing CUDA-based components in Triton. While Triton may not match CUDA in raw flexibility or peak efficiency, it enables writing high-performance GPU kernels using a Python-native, NumPy-like syntax with significantly less effort and hardware-specific knowledge. Thanks to its automatic low-level optimizations, Triton dramatically lowers the engineering barrier for prototyping and optimizing rendering pipelines—empowering researchers to focus more on innovation and less on specific GPU intricacies.

## Installation

**Dependence**: Please install [Pytorch](https://pytorch.org/get-started/locally/) first.

Install without Gsplat's CUDA implementations

```bash
git clone https://github.com/hieu1999210/gsplat-triton
cd gsplat-triton
BUILD_NO_CUDA=1 pip install .
```

Install without Gsplat CUDA implementations

```bash
git clone --recurse-submodules https://github.com/hieu1999210/gsplat-triton
cd gsplat-triton
pip install .
```

## Examples

Train a 3D Gaussian splatting model on a COLMAP capture.


- Triton backend
```bash
cd examples
GSPLAT_BACKEND="triton" python simple_trainer.py default \
    --data_dir data/360_v2/garden/ \
    --data_factor 4 \
    --result_dir ./results/garden
```
- Gsplat's CUDA backend
```bash
cd examples
GSPLAT_BACKEND="cuda" python simple_trainer.py default \
    --data_dir data/360_v2/garden/ \
    --data_factor 4 \
    --result_dir ./results/garden
```

