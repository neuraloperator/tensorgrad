import torch
import h5py
from pathlib import Path
import numpy as np
import gc
import time
import sys
import psutil
import os
import random

def print_with_timestamp(message):
    """Print message with timestamp"""
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}")
    sys.stdout.flush()

def get_memory_usage():
    """Get current memory usage"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

def normalize_data(data, name, disable_normalization=False):
    """Normalize data to match NS128 dataset characteristics:
    x: ~[-4,4], y: ~[-3,3]"""
    data_min = np.min(data)
    data_max = np.max(data)
    data_mean = np.mean(data)
    data_std = np.std(data)
    
    print_with_timestamp(f"{name} statistics before normalization:")
    print_with_timestamp(f"  min: {data_min}")
    print_with_timestamp(f"  max: {data_max}")
    print_with_timestamp(f"  mean: {data_mean}")
    print_with_timestamp(f"  std: {data_std}")
    
    if disable_normalization:
        print_with_timestamp(f"Normalization disabled for {name}, returning original data")
        return data, {
            'min': data_min,
            'max': data_max,
            'mean': data_mean,
            'std': data_std,
            'normalization_disabled': True
        }
    
    # Set target ranges based on variable type
    if name.endswith('x'):
        target_min, target_max = -1.0, 1.0
    else:  # y
        target_min, target_max = -1.0, 1.0
    
    # Min-max normalization to target range
    normalized = (data - data_min) * (target_max - target_min) / (data_max - data_min) + target_min
    
    print_with_timestamp(f"{name} statistics after normalization:")
    print_with_timestamp(f"  min: {np.min(normalized)}")
    print_with_timestamp(f"  max: {np.max(normalized)}")
    print_with_timestamp(f"  mean: {np.mean(normalized)}")
    print_with_timestamp(f"  std: {np.std(normalized)}")
    
    return normalized, {
        'min': data_min,
        'max': data_max,
        'mean': data_mean,
        'std': data_std,
        'target_min': target_min,
        'target_max': target_max,
        'normalization_disabled': False
    }

def validate_data(data, name):
    """Validate data for numerical stability"""
    if torch.is_tensor(data):
        data = data.detach().cpu().numpy()
    
    stats = {
        'min': np.min(data),
        'max': np.max(data),
        'mean': np.mean(data),
        'std': np.std(data),
        'nan_count': np.isnan(data).sum(),
        'inf_count': np.isinf(data).sum()
    }
    
    print_with_timestamp(f"{name} statistics:")
    for key, value in stats.items():
        print_with_timestamp(f"  {key}: {value}")
    
    if stats['nan_count'] > 0 or stats['inf_count'] > 0:
        raise ValueError(f"Invalid values found in {name}: {stats['nan_count']} NaNs, {stats['inf_count']} Infs")
    
    return stats

def open_hdf5_with_retry(file_path, mode='r', max_retries=5, base_delay=1):
    """Open HDF5 file with retry logic"""
    for attempt in range(max_retries):
        try:
            # Add a small random delay to prevent multiple processes from retrying at the same time
            if attempt > 0:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                print_with_timestamp(f"Retry attempt {attempt + 1}/{max_retries} after {delay:.2f} seconds")
                time.sleep(delay)
            
            # Try to open the file
            return h5py.File(file_path, mode, libver='latest', swmr=True)
        except (BlockingIOError, OSError) as e:
            if attempt == max_retries - 1:
                raise
            print_with_timestamp(f"Attempt {attempt + 1} failed: {str(e)}")
    return None

def downsample_hdf5(input_path, output_path, target_size=128, disable_normalization=False):
    print_with_timestamp(f"Starting downsampling of {input_path}")
    print_with_timestamp(f"Target size: {target_size}x{target_size}")
    print_with_timestamp(f"Output path: {output_path}")
    print_with_timestamp(f"Normalization disabled: {disable_normalization}")
    print_with_timestamp(f"Initial memory usage: {get_memory_usage():.2f} MB")
    
    try:
        # Check if input file exists
        if not Path(input_path).exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        # Open input HDF5 file
        print_with_timestamp(f"Opening input file: {input_path}")
        with open_hdf5_with_retry(input_path, 'r') as f_in:
            # Create output HDF5 file
            print_with_timestamp(f"Creating output file: {output_path}")
            with open_hdf5_with_retry(output_path, 'w') as f_out:
                # Initialize dictionary to store subsampled data for PyTorch
                subsampled_data = {}
                normalization_params = {}
                
                # Process each dataset
                for key in ['x', 'y']:
                    if key not in f_in:
                        print_with_timestamp(f"Warning: Key {key} not found in input file")
                        continue
                        
                    # Get input data shape
                    input_data = f_in[key]
                    input_shape = input_data.shape
                    print_with_timestamp(f"Processing {key} with shape {input_shape}")
                    
                    # Calculate subsampling factor
                    subsample_factor = input_shape[-1] // target_size
                    print_with_timestamp(f"Using subsampling factor: {subsample_factor}")
                    
                    # Calculate output shape
                    output_shape = (input_shape[0], target_size, target_size)
                    
                    # Create output dataset
                    print_with_timestamp(f"Creating output dataset with shape {output_shape}")
                    dataset = f_out.create_dataset(
                        key,
                        shape=output_shape,
                        dtype=np.float32,
                        compression='gzip',
                        compression_opts=4
                    )
                    
                    # Load and process entire dataset at once
                    print_with_timestamp("Loading data...")
                    data = input_data[:]
                    print_with_timestamp(f"Data loaded, memory usage: {get_memory_usage():.2f} MB")
                    
                    # Validate input data
                    input_stats = validate_data(data, f"Input {key}")
                    
                    # Subsample the data
                    print_with_timestamp("Subsampling data...")
                    subsampled = data[:, ::subsample_factor, ::subsample_factor]
                    print_with_timestamp(f"Subsampling complete, memory usage: {get_memory_usage():.2f} MB")
                    
                    # Normalize the subsampled data
                    print_with_timestamp(f"Normalizing {key} data...")
                    normalized, norm_params = normalize_data(subsampled, f"Subsampled {key}", disable_normalization)
                    normalization_params[key] = norm_params
                    
                    # Validate normalized data
                    validate_data(normalized, f"Normalized {key}")
                    
                    # Save the normalized data to HDF5
                    print_with_timestamp("Saving normalized data to HDF5...")
                    dataset[:] = normalized
                    print_with_timestamp(f"HDF5 save complete, memory usage: {get_memory_usage():.2f} MB")
                    
                    # Store normalized data for PyTorch
                    subsampled_data[key] = torch.from_numpy(normalized)
                    
                    # Clean up memory
                    print_with_timestamp("Cleaning up memory...")
                    del data, subsampled, normalized
                    gc.collect()
                    torch.cuda.empty_cache()
                    print_with_timestamp(f"Memory cleanup complete, memory usage: {get_memory_usage():.2f} MB")
                    
                    print_with_timestamp(f"Completed processing {key}")
                
                # Save the normalized data to PyTorch tensor
                pt_output_path = str(output_path).replace('.hdf5', '.pt')
                print_with_timestamp(f"Saving normalized data to PyTorch tensor: {pt_output_path}")
                torch.save(subsampled_data, pt_output_path)
                print_with_timestamp(f"PyTorch save complete, memory usage: {get_memory_usage():.2f} MB")
                
                # Copy attributes
                for key, value in f_in.attrs.items():
                    f_out.attrs[key] = value
                
                # Add downsampling and normalization metadata
                f_out.attrs['original_resolution'] = input_shape[-2:]
                f_out.attrs['downsampled_resolution'] = (target_size, target_size)
                f_out.attrs['subsampling_factor'] = subsample_factor
                f_out.attrs['downsampling_date'] = time.ctime()
                f_out.attrs['source_file'] = str(input_path)
                f_out.attrs['data_stats'] = str(input_stats)
                f_out.attrs['normalization_params'] = str(normalization_params)
                f_out.attrs['normalization_disabled'] = disable_normalization
        
        print_with_timestamp(f"Successfully downsampled and normalized {input_path} to {output_path}")
        
    except Exception as e:
        print_with_timestamp(f"Error downsampling {input_path}: {str(e)}")
        # Clean up partial output files if they exist
        try:
            output_path = Path(output_path)
            if output_path.exists():
                output_path.unlink()
                print_with_timestamp(f"Cleaned up partial HDF5 file: {output_path}")
            pt_output_path = output_path.with_suffix('.pt')
            if pt_output_path.exists():
                pt_output_path.unlink()
                print_with_timestamp(f"Cleaned up partial PyTorch file: {pt_output_path}")
        except Exception as cleanup_error:
            print_with_timestamp(f"Error during cleanup: {cleanup_error}")
        raise

def main():
    print_with_timestamp("Starting downsample_data.py")
    
    # Define paths
    #data_dir = Path("data/navier_stokes/1024_new")
    data_dir = Path("data/navier_stokes/ns_data/16traj_t4_T4")
    target_size = 128
    
    print_with_timestamp(f"Data directory: {data_dir}")
    print_with_timestamp(f"Target size: {target_size}")
    
    try:
        # Check if data directory exists
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        
        # List files in data directory
        print_with_timestamp("Files in data directory:")
        for f in data_dir.glob("*"):
            print_with_timestamp(f"  {f}")
        
        # Process training data
        train_input = data_dir / "nsforcing_train_1024.hdf5"
        train_output = data_dir / "nsforcing_train_128.hdf5"
        print_with_timestamp(f"Processing training data: {train_input} -> {train_output}")
        downsample_hdf5(train_input, train_output, target_size, disable_normalization=True)
        
        # Process test data
        test_input = data_dir / "nsforcing_test_1024.hdf5"
        test_output = data_dir / "nsforcing_test_128.hdf5"
        print_with_timestamp(f"Processing test data: {test_input} -> {test_output}")
        downsample_hdf5(test_input, test_output, target_size, disable_normalization=True)
        
        train_input = data_dir / "1024_train_data_t4_T4.hdf5"
        train_output = data_dir / "nsforcing_train_128_t4_T4.hdf5"
        print_with_timestamp(f"Processing training data: {train_input} -> {train_output}")
        downsample_hdf5(train_input, train_output, target_size, disable_normalization=True)
        
        # Process test data
        test_input = data_dir / "1024_test_data_t4_T4.hdf5"
        test_output = data_dir / "nsforcing_test_128_t4_T4.hdf5"
        print_with_timestamp(f"Processing test data: {test_input} -> {test_output}")
        downsample_hdf5(test_input, test_output, target_size, disable_normalization=True)
        
        print_with_timestamp("All processing completed successfully")
        
    except Exception as e:
        print_with_timestamp(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main() 