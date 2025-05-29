#!/usr/bin/env python3

import argparse
import numpy as np
import torch
import os
import psutil
import time
import torch.nn.functional as F

from torch.utils.data import Dataset



class TrajectoryDataset(Dataset):
    def __init__(self, u, traj_indices, time_indices, t_use=2, T=2, verbose=False):
        """
        Memory-efficient dataset that only stores indices into the original tensor.

        Args:
            u: Tensor of shape (num_traj, num_time, H, W)
            traj_indices: List of trajectory indices to use (for train/test split)
            time_indices: Allowed time indices within each trajectory (e.g. list(range(block_size)))
            t_use: Time step gap between samples (stride along time)
            T: Time gap between input and output (prediction horizon)
            verbose: If True, prints debugging information.
        """
        self.u = u  # Keep a reference to the original huge tensor.
        self.sample_indices = []  # List to hold (trajectory, t, t+T) tuples.

        # Sort time indices for consistency and create a lookup set.
        time_indices = sorted(time_indices)
        time_indices_set = set(time_indices)

        for traj in traj_indices:
            samples_added = 0
            for i in range(0, len(time_indices) - 1, t_use):
                t = time_indices[i]
                output_t = t + T
                if output_t in time_indices_set:
                    self.sample_indices.append((traj, t, output_t))
                    samples_added += 1
                    if verbose:
                        print(f"Added sample from traj {traj}: {t} -> {output_t}")
            
            if verbose:
                print(f"Added {samples_added} samples from trajectory {traj}")

        if len(self.sample_indices) == 0:
            raise ValueError("No valid samples were collected! Check your parameters.")

        if verbose:
            print(f"Final dataset size: {len(self.sample_indices)} samples")

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        traj, t, t_out = self.sample_indices[idx]
        # Retrieve the required samples on demand from the original tensor.
        sample_x = self.u[traj, t]
        sample_y = self.u[traj, t_out]
        return {'x': sample_x, 'y': sample_y}

    @property
    def x(self):
        # This property stacks and returns all input samples.
        # Beware: it will require large memory!
        return torch.stack([self[i]['x'] for i in range(len(self))])

    @property
    def y(self):
        # This property stacks and returns all output samples.
        # Beware: it will require large memory!
        return torch.stack([self[i]['y'] for i in range(len(self))])

def save_dataset(dataset, path, chunk_size=1000):
    '''
    Save dataset as .pt file with 'x' and 'y' entries in a memory-efficient way.
    
    Args:
        dataset: TrajectoryDataset object
        path: Path to save data
        chunk_size: Number of samples to process at once
    '''
    n_samples = len(dataset)
    
    # First pass: determine final shapes
    sample = dataset[0]
    x_shape = sample['x'].shape
    y_shape = sample['y'].shape
    
    # Pre-allocate final tensors
    print(f"Pre-allocating tensors for {n_samples} samples with shapes x:{x_shape}, y:{y_shape}")
    x_data = torch.empty((n_samples,) + x_shape)
    y_data = torch.empty((n_samples,) + y_shape)
    
    print(f"Current memory usage before processing: {psutil.virtual_memory().used/1e9:.2f}GB")
    
    # Process in chunks and fill tensors directly
    for start_idx in range(0, n_samples, chunk_size):
        end_idx = min(start_idx + chunk_size, n_samples)
        print(f"Processing chunk {start_idx} to {end_idx} of {n_samples}")
        
        # Get chunk of samples
        chunk_x = []
        chunk_y = []
        for idx in range(start_idx, end_idx):
            sample = dataset[idx]
            chunk_x.append(sample['x'])
            chunk_y.append(sample['y'])
        
        # Stack and assign to pre-allocated tensors
        x_data[start_idx:end_idx] = torch.stack(chunk_x)
        y_data[start_idx:end_idx] = torch.stack(chunk_y)
        
        # Clean up
        del chunk_x
        del chunk_y
        torch.cuda.empty_cache()
        
        print(f"Current memory usage: {psutil.virtual_memory().used/1e9:.2f}GB")
    
    # Save final dataset
    print("Saving final dataset...")
    torch.save({
        'x': x_data,
        'y': y_data
    }, path)
    print(f'Saved dataset to {path}')
    print(f'Final dataset shape - x: {x_data.shape}, y: {y_data.shape}')
    
    # Clean up
    del x_data
    del y_data
    torch.cuda.empty_cache()

def save_dataset_individual(dataset, base_dir, chunk_size=1000):
    """
    Saves each sample from the dataset to its own file in base_dir in a memory-efficient way.
    """
    os.makedirs(base_dir, exist_ok=True)
    n_samples = len(dataset)
    
    for start_idx in range(0, n_samples, chunk_size):
        end_idx = min(start_idx + chunk_size, n_samples)
        print(f"Processing chunk {start_idx} to {end_idx} of {n_samples}")
        
        # Process chunk of samples
        for idx in range(start_idx, end_idx):
            sample = dataset[idx]  # This retrieves one sample on demand
            out_file = os.path.join(base_dir, f"sample_{idx:05d}.pt")
            torch.save(sample, out_file)
            
            if (idx + 1) % 100 == 0:
                print(f"Saved {idx + 1} / {n_samples} samples")
                torch.cuda.empty_cache()  # Periodic memory cleanup
                
        # Clean up after each chunk
        torch.cuda.empty_cache()
        
    print(f"Saved {n_samples} samples to {base_dir}")

def detect_bad_examples(x_data, y_data, threshold=0.1):
    """
    Detect bad examples based on statistical properties.
    
    Args:
        x_data: Input data tensor
        y_data: Output data tensor
        threshold: Threshold for detecting uniform/bad data
        
    Returns:
        List of indices of bad examples
    """
    bad_indices = []
    
    for i in range(len(x_data)):
        # Check for uniform y data (low standard deviation)
        y_std = y_data[i].std()
        if y_std < threshold:
            print(f"Found uniform y data at index {i}: std={y_std:.4f}")
            bad_indices.append(i)
            continue
            
    return bad_indices

def process_data(input_path, train_path, test_path, t_use=4, T=4, train_ratio=0.8, seed=42, side_length=1024, max_side_length=1024, verbose=True):
    """
    Process NS data and create train/test datasets.
    
    Args:
        input_path: Path to input .pt file
        train_path: Path to save training data
        test_path: Path to save testing data
        t_use: Time step gap between samples
        T: Time gap between input and output
        train_ratio: Ratio of data to use for training
        seed: Random seed for reproducibility
        max_side_length: Maximum side length of the data (will resize if larger)
        verbose: Whether to print progress information
    """
    # Set random seed
    np.random.seed(seed)
    
    # Log process info
    pid = os.getpid()
    print(f"\n [PID {pid}] Starting data load")
    print(f"Loading file from: {input_path}")
    print(f"File size: {os.path.getsize(input_path) / 1e9:.2f} GB")

    print(f"⏳ Waiting 5 seconds so you can attach `top -p {pid}`...")
    time.sleep(5)  # Let you observe in htop/top if running interactively

    # Try loading
    try:
        with open(input_path, 'rb') as f:
            data = torch.load(f, map_location='cpu')
    except Exception as e:
        print(f"Failed to load file: {e}")
        return

    # Check system memory usage
    mem = psutil.virtual_memory()
    print(f"Data loaded successfully.")
    print(f"System memory: used={mem.used/1e9:.2f}GB / total={mem.total/1e9:.2f}GB")
    
    # Show data stats
    print(f" Original data shape: {data.shape}")
    print(f" #Total elements: {data.numel()}")
    
    # Check if resizing is needed
    current_size = data.shape[-1]  # Get the spatial dimension size
    print(f"Current size: {current_size}, max_side_length: {max_side_length}")
    if current_size > max_side_length:
        print(f"Resizing data from {current_size}x{current_size} to {max_side_length}x{max_side_length}")
        # Reshape to combine batch dimensions for resizing
        orig_shape = data.shape
        data = data.reshape(-1, 1, current_size, current_size)  # Add channel dim for F.interpolate
        # sub-sample data instead of resizing
        data = data[:, :, ::2, ::2]
        data = data.reshape(orig_shape[:-2] + (max_side_length, max_side_length))  # Restore original batch dimensions
        print(f"✅ Resized data shape: {data.shape}")
        
    side_length = data.shape[-1]
    
    # Process each trajectory separately
    num_real_traj = data.shape[0]  # Number of real trajectories
    timesteps_per_traj = data.shape[1]  # Number of timesteps per trajectory
    
    print(f"Processing {num_real_traj} real trajectories with {timesteps_per_traj} timesteps each")
    
    # Initialize lists to store all samples
    all_samples = []
    all_x = []
    all_y = []
    
    # Process each real trajectory
    for traj_idx in range(num_real_traj):
        print(f"\nProcessing real trajectory {traj_idx + 1}/{num_real_traj}")
        
        # Get the current trajectory
        current_traj = data[traj_idx]  # Shape: [timesteps, H, W]
        
        # Create (t, t+T) pairs for this trajectory
        for t in range(0, timesteps_per_traj - T, t_use):
            # Get input and output samples
            x = current_traj[t]
            y = current_traj[t + T]
            
            # Add to lists for bad data detection
            all_x.append(x)
            all_y.append(y)
            
            # Add to all samples
            all_samples.append({
                'x': x,
                'y': y
            })
            
            if verbose and t % 100 == 0:
                print(f"Added sample from trajectory {traj_idx + 1}: t={t} -> t+T={t + T}")
    
    print(f"\nTotal samples collected: {len(all_samples)}")
    
    # Detect bad examples
    print("\nDetecting bad examples...")
    all_x_tensor = torch.stack(all_x)
    all_y_tensor = torch.stack(all_y)
    bad_indices = detect_bad_examples(all_x_tensor, all_y_tensor)
    
    if bad_indices:
        print(f"Found {len(bad_indices)} bad examples: {bad_indices}")
        # Remove bad examples
        mask = np.ones(len(all_samples), dtype=bool)
        mask[bad_indices] = False
        all_samples = [all_samples[i] for i in range(len(all_samples)) if mask[i]]
        print(f"Removed bad examples. Remaining samples: {len(all_samples)}")
    
    # Free memory
    del data
    del all_x
    del all_y
    del all_x_tensor
    del all_y_tensor
    torch.cuda.empty_cache()
    
    # Split into train and test
    num_samples = len(all_samples)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    
    train_size = int(train_ratio * num_samples)
    train_indices = indices[:train_size].tolist()
    test_indices = indices[train_size:].tolist()
    
    print(f"\nTrain samples: {len(train_indices)}")
    print(f"Test samples: {len(test_indices)}")
    
    # Create train dataset
    print("Creating training dataset...")
    train_samples = [all_samples[i] for i in train_indices]
    train_x = torch.stack([s['x'] for s in train_samples])
    train_y = torch.stack([s['y'] for s in train_samples])
    
    train_data = {
        'x': train_x,
        'y': train_y
    }
    torch.save(train_data, train_path)
    print(f"Saved training dataset to {train_path}")
    
    # Free memory
    del train_samples
    del train_x
    del train_y
    del train_data
    torch.cuda.empty_cache()
    
    # Create test dataset
    print("Creating testing dataset...")
    test_samples = [all_samples[i] for i in test_indices]
    test_x = torch.stack([s['x'] for s in test_samples])
    test_y = torch.stack([s['y'] for s in test_samples])
    
    test_data = {
        'x': test_x,
        'y': test_y
    }
    torch.save(test_data, test_path)
    print(f"Saved testing dataset to {test_path}")
    
    # Free memory
    del test_samples
    del test_x
    del test_y
    del test_data
    del all_samples
    torch.cuda.empty_cache()
    print(f"[PID {os.getpid()}] Data processing completed")

def main():
    parser = argparse.ArgumentParser(description='Process NS data and create train/test datasets')
    parser.add_argument('--input', type=str, required=True, help='Path to input .pt file')
    parser.add_argument('--train', type=str, required=True, help='Path to save training data')
    parser.add_argument('--test', type=str, required=True, help='Path to save testing data')
    parser.add_argument('--t_use', type=int, default=4, help='Time step gap between samples')
    parser.add_argument('--T', type=int, default=4, help='Time gap between input and output')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Ratio of data to use for training')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--side_length', type=int, default=1024, help='Side length of the data')
    parser.add_argument('--max_side_length', type=int, default=1024, help='Side length of the data')
    parser.add_argument('--verbose', type=bool, default=True, help='Verbose output')
    args = parser.parse_args()
    
    print(f"🔍 [PID {os.getpid()}] Starting data processing")
    process_data(
        args.input,
        args.train,
        args.test,
        args.t_use,
        args.T,
        args.train_ratio,
        args.seed,
        side_length=args.side_length,
        max_side_length=args.max_side_length,
        verbose=args.verbose
    )

if __name__ == '__main__':
    main() 