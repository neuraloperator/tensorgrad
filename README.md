# TensorGRaD: Tensor Gradient Robust Decomposition for Memory-Efficient Neural Operator Training

This repository contains the implementation of TensorGRaD, a memory-efficient optimizer for training large neural operators. TensorGRaD applies tensor decomposition techniques to gradient updates, significantly reducing memory requirements while maintaining model performance.

## Installation

Start with a fresh conda environment, then install dependencies:

```bash
# Create and activate a new conda environment with Python 3.10
conda create -n tensorgrad python=3.10
conda activate tensorgrad

# Install PyTorch with CUDA support
conda install pytorch torchvision -c pytorch

# Clone the repository
git clone https://github.com/username/tensorgrad.git
cd tensorgrad

# Install dependencies
pip install -r requirements.txt

# Install local packages in development mode
pip install -e ./tensorly
pip install -e ./neuraloperator
```

## Project Overview

TensorGRaD is a memory-efficient optimizer that works by transforming the gradients into low-rank plus sparse decompositions

- **AdamW (tensorgrad/adamw.py)**: Implements a single projector configuration (low-rank or sparse alone)
- **TensorGRaD (tensorgrad/tensorgrad.py)**: Implements composite projectors with AdamW, combining low-rank and sparse projections
  - The rank/sparsity parameters define the percentage of the full tensor size
  - For example, 0.25 means the optimizer state uses 25% of the original size
  - Combinations like rank=0.20 and sparsity=0.05 give a total of 0.25 (25% of original size)

## Directory Structure

- **tensorgrad/**: Contains our optimizer implementation
  - `adamw.py`: Single projector configurations (low-rank or sparse)
  - `tensorgrad.py`: Composite projector implementations
  - Gradient projections for matrices and tensors
  - Utilities for profiling and large-scale experiments
  - **tensorgrad/projectors/**: Contains all projection implementations
    - Tensor projectors: low-rank, unstructured sparse, and structured sparse
    - Matrix projectors: `galore_projector.py` (low-rank) and `sparse_projector.py` (sparse)
    - Utility functions to create and manage projectors

- **scripts/experiments/**: Contains experiment configurations
  - Low-rank, structured sparse, and low-rank+sparse (TensorGRaD) experiments
  - Both half and full precision configurations
  - AdamW baseline configurations
  - All experiments run on the Navier Stokes 128 dataset using train_EXP_tensorgrad.py

- **scripts/profiling/**: Contains profiling experiments
  - Two model configurations: both with 128 channels
  - One model with 128 channel dimension, another with 256 channel dimension

- **training script**:
  - `train_ns_repro_tensorgrad.py`: Navier-Stokes training

## Running Experiments

Experiment configurations are managed via `configmypy`. You can configure scripts both from the command line and associated YAML files in the `config/` directory.

Example:
```bash
python train_ns_repro_tensorgrad.py --config_file ns_tensorgrad_repro_config.yaml
```

For pre-configured experiments, use the bash scripts in the `scripts/experiments/` directory.

### Key Configuration Options

- **Optimizer Selection**:
  - `--opt.tensorgrad True` - Enable TensorGRaD with projections
  - `--opt.tensorgrad False` - Use regular AdamW without projectors

- **Projection Types**:
  - Available projectors:
    - `low_rank` - Tucker decomposition for tensors
    - `structured_sparse` - Dimension-wise sparsity (structured along tensor dimensions)
    - `unstructured_sparse` - Element-wise sparsity (unstructured)
  
  - Single projector configuration examples:
    ```
    --opt.proj_type low_rank \
    --opt.rank 0.25
    ```
    or
    ```
    --opt.proj_type structured_sparse \
    --opt.sparse_ratio 0.25 \
    --opt.sparse_type topk
    ```

  - Composite projector configuration (TensorGRaD):
    ```
    --opt.proj_type unstructured_sparse \
    --opt.sparse_ratio 0.05 \
    --opt.sparse_type randk \
    --opt.second_proj_type low_rank \
    --opt.second_rank 0.20
    ```
    This creates a composite of 5% unstructured sparse + 20% low-rank for a total of 25% memory usage.

- **Projection Update Frequency**:
  - `--opt.update_proj_gap 1000` - Set interval between projection updates

- **Sparsity Type Options**:
  - `--opt.sparse_type topk` - Keep top-k elements by magnitude
  - `--opt.sparse_type randk` - Keep random k elements
  - `--opt.sparse_type probability` - Sample proportional to norm

- **Precision Control**:
  - Full precision: 
    ```
    --fno.fno_block_precision full \
    --fno.fno_block_weights_precision full
    ```
  
  - Half precision: 
    ```
    --fno.fno_block_precision mixed \
    --fno.fno_block_weights_precision half
    ```

## Data Handling

### Standard Datasets

The TensorGRaD experiments use several datasets, some of which are automatically downloaded:

1. **Navier-Stokes (Reynolds number 1000)**
   - Resolutions: 128×128 and 1024×1024
   - Automatically downloaded from the neuraloperator library
   - Used in most of the examples in `scripts/experiments/`

2. **Navier-Stokes (Reynolds number 10^5)**
   - Resolution: 1024×1024
   - Download: [Hugging Face Dataset](https://huggingface.co/datasets/sloeschcke/navier_stokes_res1024_Re10e5)
   - Change path to a folder containing the files: `nsforcing_test_1024.hdf5` and `nsforcing_train_1024.hdf5` 
   - Pre-trained model checkpoint also available at the same link
   - See our paper for details on dataset creation

### Custom Dataset Creation

To create your own datasets for use with TensorGRaD:

1. Check the `dataset_creation/` directory for examples and utilities
2. Follow the data format used in the neuraloperator library
3. See `FullSizeNavierStokes` class in `tensorgrad/navier_stokes.py` for an example of a custom dataset implementation

## Memory Profiling

Memory usage statistics are stored in the `memstats/` directory, and profiling outputs are in `profiler_outputs/`.

To run memory profiling experiments for tensorgrad 25% rank:
```bash
bash scripts/profiling/128modes_256channels_4layers/US_LR_025.sh
```

## Citation

If you use this code in your research, please cite our paper:

```
@article{tensorgrad,
  title={TensorGRaD: Tensor Gradient Robust Decomposition for Memory-Efficient Neural Operator Training},
  author={...},
  journal={...},
  year={...}
}
```