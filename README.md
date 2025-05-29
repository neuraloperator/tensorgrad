# TensorGRaD: Tensor Gradient Robust Decomposition for Memory-Efficient Neural Operator Training

This repository contains the official implementation of **TensorGRaD**, a memory-efficient gradient optimization framework for training large-scale neural operators. TensorGRaD uses a robust combination of low-rank tensor decomposition and unstructured sparsification to compress gradient updates. TensorGRaD achieves significant memory savings while maintaining or even improving model performance.

## Installation

Start from a clean conda environment:

```bash
# Create and activate a new conda environment with Python 3.10
conda create -n tensorgrad python=3.10
conda activate tensorgrad

# Install PyTorch with CUDA support
conda install pytorch torchvision -c pytorch

# Clone the repository
git clone https://github.com/neuraloperator/tensorgrad.git
cd tensorgrad

# Install dependencies
pip install -r requirements.txt
```

## Overview

TensorGRaD is a drop-in optimizer that replaces standard optimizers like AdamW. It applies compression at the gradient level through:
- **Low-rank decomposition** using a Tucker higher-order low-rank decomposition
- **Gradient sparsification** using structured or unstructured sparsity (top-$k$, random-$k$, or probabilistic)
- Support for **composite projectors** combining both methods for aggressive compression

TensorGRaD supports mixed-precision training and is implemented for scientific ML workloads that optimize tensors.

## Directory Structure

- `tensorgrad/`: Optimizer implementations
  - `adamw.py`: Single projector optimizers
  - `tensorgrad.py`: Composite projector variant (TensorGRaD)
  - `projectors/`: Includes all projector logic (tensor/matrix, sparse/low-rank)

- `scripts/experiments/`: Runs for ablation studies and benchmarks (low-rank, sparse, mixed)
- `scripts/profiling/`: Memory profiling tools for different architectures
- `train_ns_repro_tensorgrad.py`: Main training script on Navier–Stokes

## Running Experiments

Use YAML-based configs and command-line overrides for training:

```bash
python train_ns_repro_tensorgrad.py --config_file ns_tensorgrad_repro_config.yaml
```

Or use the prepared bash scripts in `scripts/experiments/`.

### Example Configurations

#### Optimizer Selection
```bash
--opt.tensorgrad True    # Enable TensorGRaD
--opt.tensorgrad False   # Use AdamW
```


####  TensorGRaD - unstructured sparse + low-rank 
```bash
--opt.proj_type unstructured_sparse \
--opt.sparse_ratio 0.05 \
--opt.sparse_type randk \
--opt.second_proj_type low_rank \
--opt.second_rank 0.20
```

#### Single Projector (Low-Rank or Sparse)
```bash
--opt.proj_type low_rank \
--opt.rank 0.25
```
or
```bash
--opt.proj_type structured_sparse \
--opt.sparse_ratio 0.25 \
--opt.sparse_type randk
```


#### Additional Flags
```bash
--opt.update_proj_gap 1000         # Projection update interval
--fno.fno_block_precision mixed    # Activations: mixed precision
--fno.fno_block_weights_precision half  # Weights: half precision
```

## Datasets

### Built-in Support

1. **Navier–Stokes ($Re=1000$)**
   - Resolutions: 128×128 and 1024×1024
   - Automatically downloaded via neuraloperator

2. **Navier–Stokes ($Re=10^5$)**
   - High-resolution (1024×1024)
   - Download manually from [Hugging Face](https://huggingface.co/datasets/sloeschcke/navier_stokes_res1024_Re10e5)
   - Requires `nsforcing_test_1024.hdf5` and `nsforcing_train_1024.hdf5`
   - See paper for pretraining and dataset details

### Custom Data

To prepare your own data:
- Follow the structure used in `neuraloperator`
- Review `FullSizeNavierStokes` class in `tensorgrad/navier_stokes.py`
- Utilities are available in `dataset_creation/`

## Memory Profiling

Use bash scripts under `scripts/profiling/` for benchmarking.

```bash
bash scripts/profiling/128modes_256channels_4layers/US_LR_025.sh
```

Profiling outputs are written to `memstats/` and `profiler_outputs/`.

## Citation

If you use this code, please cite:

```bibtex
@article{tensorgrad,
  title={TensorGRaD: Tensor Gradient Robust Decomposition for Memory-Efficient Neural Operator Training},
  author={...},
  journal={...},
  year={2025}
}
```
