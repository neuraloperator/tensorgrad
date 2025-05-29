import torch
import h5py
from pathlib import Path
import numpy as np
import gc
import time

def convert_pt_to_hdf5(pt_path, hdf5_path, chunk_size=100):
    print(f"Converting {pt_path} to {hdf5_path}")
    
    try:
        # Load the .pt file with map_location to CPU and weights_only=True for memory efficiency
        print(f"Loading data from {pt_path}...")
        data = torch.load(pt_path, map_location='cpu', weights_only=True)
        print("Data loaded successfully")
        
        # Create HDF5 file
        with h5py.File(hdf5_path, 'w') as f:
            # Convert tensors to numpy arrays and store in HDF5
            for key, value in data.items():
                if isinstance(value, torch.Tensor):
                    # Get the shape and create dataset with chunks
                    shape = value.shape
                    chunks = tuple(min(chunk_size, s) for s in shape)
                    
                    # Create dataset with compression
                    dataset = f.create_dataset(
                        key, 
                        shape=shape,
                        chunks=chunks,
                        dtype=np.float32,
                        compression='gzip',
                        compression_opts=4
                    )
                    
                    # Write data in small chunks
                    for i in range(0, shape[0], chunk_size):
                        end_idx = min(i + chunk_size, shape[0])
                        print(f"Processing {key} chunk {i}/{shape[0]}")
                        
                        # Convert chunk to numpy and write
                        chunk_data = value[i:end_idx].numpy()
                        dataset[i:end_idx] = chunk_data
                        
                        # Clean up memory
                        del chunk_data
                        gc.collect()
                        torch.cuda.empty_cache()  # Just in case GPU memory is used
                        
                    print(f"Completed processing {key} with shape {shape}")
                else:
                    f.create_dataset(key, data=np.array(value))
            
            # Add metadata (fixed the Path.ctime issue)
            pt_path_obj = Path(pt_path)
            f.attrs['creation_date'] = time.ctime(pt_path_obj.stat().st_ctime)
            f.attrs['source_file'] = str(pt_path)
            f.attrs['conversion_date'] = time.ctime()
        
        # Clean up memory
        del data
        gc.collect()
        torch.cuda.empty_cache()
        
        print(f"Successfully converted {pt_path} to {hdf5_path}")
        
    except Exception as e:
        print(f"Error converting {pt_path}: {str(e)}")
        # Make sure to clean up any partially created files
        try:
            hdf5_path = Path(hdf5_path)
            if hdf5_path.exists():
                hdf5_path.unlink()
                print(f"Cleaned up partial HDF5 file: {hdf5_path}")
        except Exception as cleanup_error:
            print(f"Error during cleanup: {cleanup_error}")
        raise

def main():
    # Define paths
    #data_dir = Path("data/navier_stokes/ns_data/16traj")
    
    #TRAIN_PATH="data/navier_stokes/ns_data/1024_new/1024_train_data.pt"
    #data_dir = Path("data/navier_stokes/1024_new")
    data_dir = Path("data/navier_stokes/ns_data/16traj_t4_T4")
    
    try:
        # # Convert training data
        train_pt = data_dir / "1024_train_data.pt"
        train_hdf5 = data_dir / "nsforcing_train_1024.hdf5"
        convert_pt_to_hdf5(train_pt, train_hdf5)
        
        # Convert test data
        test_pt = data_dir / "1024_test_data.pt"
        test_hdf5 = data_dir / "nsforcing_test_1024.hdf5"
        convert_pt_to_hdf5(test_pt, test_hdf5)
        
        
    except Exception as e:
        print(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main() 