#!/usr/bin/env python3

import argparse
import numpy as np
import torch
import os
import psutil
import time
import h5py
from pathlib import Path
import gc
import sys
import random
import matplotlib.pyplot as plt

def print_with_timestamp(message):
    """Print message with timestamp"""
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}")
    sys.stdout.flush()

def get_memory_usage():
    """Get current memory usage"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

def detect_bad_examples(x_data, y_data, threshold=0.1):
    """Detect bad examples based on statistical properties."""
    bad_indices = []
    
    for i in range(len(x_data)):
        # Check for uniform y data (low standard deviation)
        y_std = y_data[i].std()
        if y_std < threshold:
            print_with_timestamp(f"Found uniform y data at index {i}: std={y_std:.4f}")
            bad_indices.append(i)
            continue
            
    return bad_indices

def analyze_data_complexity(x_data, y_data, T):
    """Analyze the complexity of the data based on the timestep T."""
    print_with_timestamp(f"\nAnalyzing data complexity for T={T}")
    
    # Calculate statistics for each sample
    x_means = []
    x_stds = []
    y_means = []
    y_stds = []
    correlations = []
    
    for i in range(len(x_data)):
        x = x_data[i].numpy()
        y = y_data[i].numpy()
        
        x_means.append(np.mean(x))
        x_stds.append(np.std(x))
        y_means.append(np.mean(y))
        y_stds.append(np.std(y))
        
        # Calculate correlation between x and y
        x_flat = x.flatten()
        y_flat = y.flatten()
        correlation = np.corrcoef(x_flat, y_flat)[0, 1]
        correlations.append(correlation)
    
    # Print statistics
    print_with_timestamp(f"Input statistics (x):")
    print_with_timestamp(f"  Mean: {np.mean(x_means):.4f} ± {np.std(x_means):.4f}")
    print_with_timestamp(f"  Std: {np.mean(x_stds):.4f} ± {np.std(x_stds):.4f}")
    
    print_with_timestamp(f"Output statistics (y):")
    print_with_timestamp(f"  Mean: {np.mean(y_means):.4f} ± {np.std(y_means):.4f}")
    print_with_timestamp(f"  Std: {np.mean(y_stds):.4f} ± {np.std(y_stds):.4f}")
    
    print_with_timestamp(f"Correlation between x and y:")
    print_with_timestamp(f"  Mean: {np.mean(correlations):.4f} ± {np.std(correlations):.4f}")
    
    # Calculate complexity metrics
    mean_correlation = np.mean(correlations)
    complexity_score = (1 - mean_correlation) * T  # Higher score means more complex
    
    print_with_timestamp(f"Complexity score: {complexity_score:.4f}")
    print_with_timestamp(f"  (Higher score indicates more complex relationship between x and y)")
    
    return complexity_score

def plot_sequential_pairs(x_data, y_data, output_dir, t_use, T, num_pairs=3):
    """Plot sequential input-output pairs to visualize temporal relationships."""
    print_with_timestamp("Creating visualization of sequential pairs...")
    
    # Create output directory for plots
    plot_dir = Path(output_dir) / "visualizations"
    plot_dir.mkdir(exist_ok=True)
    
    # Take samples spaced by t_use
    fig, axes = plt.subplots(2, num_pairs, figsize=(5*num_pairs, 10))
    fig.suptitle(f'Samples spaced by t_use={t_use}', fontsize=16)
    
    # Plot inputs
    for i in range(num_pairs):
        idx = i * t_use
        x = x_data[idx].numpy()
        
        # Plot input
        im1 = axes[0, i].imshow(x, cmap='viridis')
        axes[0, i].set_title(f'Input t={idx}')
        plt.colorbar(im1, ax=axes[0, i])
        
        # Plot output
        y = y_data[idx].numpy()
        im2 = axes[1, i].imshow(y, cmap='viridis')
        axes[1, i].set_title(f'Output t+{T}={idx+T}')
        plt.colorbar(im2, ax=axes[1, i])
    
    plt.tight_layout()
    plot_path = plot_dir / f'samples_tuse{t_use}_T{T}.png'
    plt.savefig(plot_path)
    plt.close()
    print_with_timestamp(f"Saved visualization to {plot_path}")
    
    # Create a separate plot for differences between inputs
    if num_pairs > 1:
        fig, axes = plt.subplots(1, num_pairs-1, figsize=(5*(num_pairs-1), 5))
        fig.suptitle(f'Differences between inputs spaced by t_use={t_use}', fontsize=16)
        
        for i in range(num_pairs-1):
            idx1 = i * t_use
            idx2 = (i+1) * t_use
            
            # Difference between inputs spaced by t_use
            x1 = x_data[idx1].numpy()
            x2 = x_data[idx2].numpy()
            x_diff = x2 - x1
            
            im = axes[i].imshow(x_diff, cmap='RdBu', vmin=-np.max(np.abs(x_diff)), vmax=np.max(np.abs(x_diff)))
            axes[i].set_title(f'Input diff t={idx2}-{idx1}')
            plt.colorbar(im, ax=axes[i])
            
            # Print statistics about the difference
            print_with_timestamp(f"Difference between t={idx1} and t={idx2}:")
            print_with_timestamp(f"  Max difference: {np.max(np.abs(x_diff)):.4f}")
            print_with_timestamp(f"  Mean difference: {np.mean(x_diff):.4f}")
            print_with_timestamp(f"  Std of difference: {np.std(x_diff):.4f}")
        
        plt.tight_layout()
        diff_plot_path = plot_dir / f'input_differences_tuse{t_use}.png'
        plt.savefig(diff_plot_path)
        plt.close()
        print_with_timestamp(f"Saved input differences visualization to {diff_plot_path}")

def process_data(input_path, train_path, test_path, t_use=4, T=4, train_ratio=0.8, seed=42, side_length=1024, max_side_length=1024, verbose=True):
    """Process NS data and create train/test datasets."""
    # Set random seed
    np.random.seed(seed)
    
    print_with_timestamp(f"Starting data load from {input_path}")
    print_with_timestamp(f"File size: {os.path.getsize(input_path) / 1e9:.2f} GB")

    # Try loading
    try:
        with open(input_path, 'rb') as f:
            data = torch.load(f, map_location='cpu')
    except Exception as e:
        print_with_timestamp(f"Failed to load file: {e}")
        return

    print_with_timestamp(f"Data loaded successfully. Shape: {data.shape}")
    
    # Check if resizing is needed
    current_size = data.shape[-1]
    if current_size > max_side_length:
        print_with_timestamp(f"Resizing data from {current_size}x{current_size} to {max_side_length}x{max_side_length}")
        orig_shape = data.shape
        data = data.reshape(-1, 1, current_size, current_size)
        data = data[:, :, ::2, ::2]
        data = data.reshape(orig_shape[:-2] + (max_side_length, max_side_length))
        print_with_timestamp(f"Resized data shape: {data.shape}")
        
    side_length = data.shape[-1]
    
    # Process each trajectory
    num_real_traj = data.shape[0]
    timesteps_per_traj = data.shape[1]
    
    print_with_timestamp(f"Processing {num_real_traj} trajectories with {timesteps_per_traj} timesteps each")
    
    all_samples = []
    all_x = []
    all_y = []
    
    for traj_idx in range(num_real_traj):
        if verbose and traj_idx % 5 == 0:
            print_with_timestamp(f"Processing trajectory {traj_idx + 1}/{num_real_traj}")
        
        current_traj = data[traj_idx]
        
        for t in range(0, timesteps_per_traj - T, t_use):
            x = current_traj[t]
            y = current_traj[t + T]
            
            all_x.append(x)
            all_y.append(y)
            all_samples.append({'x': x, 'y': y})
    
    print_with_timestamp(f"Total samples collected: {len(all_samples)}")
    
    # Analyze data complexity
    all_x_tensor = torch.stack(all_x)
    all_y_tensor = torch.stack(all_y)
    complexity_score = analyze_data_complexity(all_x_tensor, all_y_tensor, T)
    
    # Create visualizations of sequential pairs
    plot_sequential_pairs(all_x_tensor, all_y_tensor, os.path.dirname(train_path), t_use, T)
    
    # Detect bad examples
    print_with_timestamp("Detecting bad examples...")
    bad_indices = detect_bad_examples(all_x_tensor, all_y_tensor)
    
    if bad_indices:
        print_with_timestamp(f"Found {len(bad_indices)} bad examples: {bad_indices}")
        mask = np.ones(len(all_samples), dtype=bool)
        mask[bad_indices] = False
        all_samples = [all_samples[i] for i in range(len(all_samples)) if mask[i]]
        print_with_timestamp(f"Removed bad examples. Remaining samples: {len(all_samples)}")
    
    # Free memory
    del data, all_x, all_y, all_x_tensor, all_y_tensor
    torch.cuda.empty_cache()
    
    # Split into train and test
    num_samples = len(all_samples)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    
    train_size = int(train_ratio * num_samples)
    train_indices = indices[:train_size].tolist()
    test_indices = indices[train_size:].tolist()
    
    print_with_timestamp(f"Train samples: {len(train_indices)}, Test samples: {len(test_indices)}")
    
    # Create train dataset
    print_with_timestamp("Creating training dataset...")
    train_samples = [all_samples[i] for i in train_indices]
    train_x = torch.stack([s['x'] for s in train_samples])
    train_y = torch.stack([s['y'] for s in train_samples])
    
    train_data = {'x': train_x, 'y': train_y}
    torch.save(train_data, train_path)
    print_with_timestamp(f"Saved training dataset to {train_path}")
    
    # Create test dataset
    print_with_timestamp("Creating testing dataset...")
    test_samples = [all_samples[i] for i in test_indices]
    test_x = torch.stack([s['x'] for s in test_samples])
    test_y = torch.stack([s['y'] for s in test_samples])
    
    test_data = {'x': test_x, 'y': test_y}
    torch.save(test_data, test_path)
    print_with_timestamp(f"Saved testing dataset to {test_path}")
    
    # Clean up
    del train_samples, train_x, train_y, train_data
    del test_samples, test_x, test_y, test_data
    del all_samples
    torch.cuda.empty_cache()

def convert_pt_to_hdf5(pt_path, hdf5_path, chunk_size=100):
    """Convert PyTorch .pt file to HDF5 format."""
    print_with_timestamp(f"Converting {pt_path} to {hdf5_path}")
    
    try:
        data = torch.load(pt_path, map_location='cpu', weights_only=True)
        
        with h5py.File(hdf5_path, 'w') as f:
            for key, value in data.items():
                if isinstance(value, torch.Tensor):
                    shape = value.shape
                    chunks = tuple(min(chunk_size, s) for s in shape)
                    
                    dataset = f.create_dataset(
                        key, 
                        shape=shape,
                        chunks=chunks,
                        dtype=np.float32,
                        compression='gzip',
                        compression_opts=4
                    )
                    
                    for i in range(0, shape[0], chunk_size):
                        end_idx = min(i + chunk_size, shape[0])
                        print_with_timestamp(f"Processing {key} chunk {i}/{shape[0]}")
                        
                        chunk_data = value[i:end_idx].numpy()
                        dataset[i:end_idx] = chunk_data
                        
                        del chunk_data
                        gc.collect()
                        torch.cuda.empty_cache()
                        
                    print_with_timestamp(f"Completed processing {key} with shape {shape}")
                else:
                    f.create_dataset(key, data=np.array(value))
            
            f.attrs['creation_date'] = time.ctime(Path(pt_path).stat().st_ctime)
            f.attrs['source_file'] = str(pt_path)
            f.attrs['conversion_date'] = time.ctime()
        
        del data
        gc.collect()
        torch.cuda.empty_cache()
        
        print_with_timestamp(f"Successfully converted {pt_path} to {hdf5_path}")
        
    except Exception as e:
        print_with_timestamp(f"Error converting {pt_path}: {str(e)}")
        try:
            hdf5_path = Path(hdf5_path)
            if hdf5_path.exists():
                hdf5_path.unlink()
                print_with_timestamp(f"Cleaned up partial HDF5 file: {hdf5_path}")
        except Exception as cleanup_error:
            print_with_timestamp(f"Error during cleanup: {cleanup_error}")
        raise

def downsample_hdf5(input_path, output_path, target_sizes=[128], disable_normalization=False):
    """Downsample HDF5 data to multiple target sizes."""
    print_with_timestamp(f"Starting downsampling of {input_path}")
    
    try:
        with h5py.File(input_path, 'r') as f_in:
            for target_size in target_sizes:
                output_path_size = output_path.parent / f"{output_path.stem}_{target_size}{output_path.suffix}"
                print_with_timestamp(f"Creating downsampled version at resolution {target_size} -> {output_path_size}")
                
                with h5py.File(output_path_size, 'w') as f_out:
                    for key in ['x', 'y']:
                        if key not in f_in:
                            print_with_timestamp(f"Warning: Key {key} not found in input file")
                            continue
                            
                        input_data = f_in[key]
                        input_shape = input_data.shape
                        print_with_timestamp(f"Processing {key} with shape {input_shape}")
                        
                        subsample_factor = input_shape[-1] // target_size
                        output_shape = (input_shape[0], target_size, target_size)
                        
                        dataset = f_out.create_dataset(
                            key,
                            shape=output_shape,
                            dtype=np.float32,
                            compression='gzip',
                            compression_opts=4
                        )
                        
                        data = input_data[:]
                        subsampled = data[:, ::subsample_factor, ::subsample_factor]
                        
                        if not disable_normalization:
                            # Normalize to [-1, 1]
                            data_min = np.min(subsampled)
                            data_max = np.max(subsampled)
                            normalized = 2 * (subsampled - data_min) / (data_max - data_min) - 1
                            dataset[:] = normalized
                        else:
                            dataset[:] = subsampled
                        
                        print_with_timestamp(f"Completed processing {key} for resolution {target_size}")
                    
                    # Copy attributes
                    for key, value in f_in.attrs.items():
                        f_out.attrs[key] = value
                    
                    f_out.attrs['original_resolution'] = input_shape[-2:]
                    f_out.attrs['downsampled_resolution'] = (target_size, target_size)
                    f_out.attrs['subsampling_factor'] = subsample_factor
                    f_out.attrs['downsampling_date'] = time.ctime()
                    f_out.attrs['source_file'] = str(input_path)
                
                print_with_timestamp(f"Successfully downsampled to resolution {target_size}")
        
    except Exception as e:
        print_with_timestamp(f"Error downsampling {input_path}: {str(e)}")
        try:
            # Clean up any partial output files
            for target_size in target_sizes:
                output_path_size = output_path.parent / f"{output_path.stem}_{target_size}{output_path.suffix}"
                if output_path_size.exists():
                    output_path_size.unlink()
                    print_with_timestamp(f"Cleaned up partial HDF5 file: {output_path_size}")
        except Exception as cleanup_error:
            print_with_timestamp(f"Error during cleanup: {cleanup_error}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Process NS data and create train/test datasets')
    parser.add_argument('--input', type=str, required=True, help='Path to input .pt file')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save processed data')
    parser.add_argument('--t_use', type=int, default=4, help='Time step gap between samples')
    parser.add_argument('--T', type=int, default=4, help='Time gap between input and output')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Ratio of data to use for training')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--side_length', type=int, default=1024, help='Side length of the data')
    parser.add_argument('--max_side_length', type=int, default=1024, help='Maximum side length of the data')
    parser.add_argument('--target_sizes', type=int, nargs='+', default=[128], help='Target sizes for downsampling')
    parser.add_argument('--disable_normalization', action='store_true', help='Disable normalization during downsampling')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--keep_pt_files', action='store_true', help='Keep intermediate .pt files')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define paths
    train_pt = output_dir / "train_data.pt"
    test_pt = output_dir / "test_data.pt"
    train_hdf5 = output_dir / "train_data.hdf5"
    test_hdf5 = output_dir / "test_data.hdf5"
    train_downsampled = output_dir / "nsforcing_train.hdf5"
    test_downsampled = output_dir / "nsforcing_test.hdf5"
    
    try:
        # Check if HDF5 files already exist
        if train_hdf5.exists() and test_hdf5.exists():
            print_with_timestamp("Found existing HDF5 files, skipping processing and conversion steps")
        else:
            # Process data and save as .pt files
            process_data(
                args.input,
                train_pt,
                test_pt,
                args.t_use,
                args.T,
                args.train_ratio,
                args.seed,
                args.side_length,
                args.max_side_length,
                args.verbose
            )
            
            # Convert to HDF5
            convert_pt_to_hdf5(train_pt, train_hdf5)
            convert_pt_to_hdf5(test_pt, test_hdf5)
            
            # Clean up intermediate .pt files if not explicitly kept
            if not args.keep_pt_files:
                print_with_timestamp("Cleaning up intermediate .pt files...")
                if train_pt.exists():
                    train_pt.unlink()
                    print_with_timestamp(f"Deleted {train_pt}")
                if test_pt.exists():
                    test_pt.unlink()
                    print_with_timestamp(f"Deleted {test_pt}")
        
        # Downsample to multiple resolutions
        downsample_hdf5(train_hdf5, train_downsampled, args.target_sizes, args.disable_normalization)
        downsample_hdf5(test_hdf5, test_downsampled, args.target_sizes, args.disable_normalization)
        
        print_with_timestamp("All processing completed successfully")
        
    except Exception as e:
        print_with_timestamp(f"Error during processing: {str(e)}")
        # Don't delete .pt files if there was an error
        print_with_timestamp("Keeping intermediate .pt files due to error")
        raise

if __name__ == "__main__":
    main() 