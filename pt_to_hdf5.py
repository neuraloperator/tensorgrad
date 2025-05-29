import torch
import h5py
import argparse
import numpy as np

def convert_tensor_to_hdf5(tensor, hdf5_file, dataset_name="u", chunk_size=1):
    """
    Converts a single torch.Tensor to an HDF5 dataset, writing it in slices along the first dimension.
    """
    # Ensure the tensor is on CPU
    tensor = tensor.cpu()
    # Get the numpy dtype and shape
    tensor_np = tensor.numpy()
    shape = tensor_np.shape
    dtype = tensor_np.dtype
    
    # Create the dataset in the HDF5 file with chunking enabled.
    # Here we use the provided chunk_size for the first dimension and full size for the remaining dimensions.
    chunks = (chunk_size,) + shape[1:]
    dset = hdf5_file.create_dataset(
        dataset_name, shape=shape, dtype=dtype, 
        compression="gzip", chunks=chunks
    )
    
    # Write the tensor data in chunks along the first dimension.
    for i in range(0, shape[0], chunk_size):
        end = min(i + chunk_size, shape[0])
        print(f"Writing slice {i}:{end} of dataset '{dataset_name}'")
        dset[i:end] = tensor[i:end].cpu().numpy()
    
def convert_dict_to_hdf5(data_dict, hdf5_path, chunk_size=1):
    """
    Converts a dictionary of torch.Tensors to an HDF5 file,
    storing each tensor under its respective key.
    """
    with h5py.File(hdf5_path, "w") as f:
        for key, value in data_dict.items():
            if isinstance(value, torch.Tensor):
                print(f"Converting tensor for key: {key}")
                convert_tensor_to_hdf5(value, f, dataset_name=key, chunk_size=chunk_size)
            else:
                print(f"Skipping key '{key}': not a torch.Tensor (found type {type(value)})")
                
def convert_pt_to_hdf5(pt_path, hdf5_path, chunk_size=1):
    """
    Loads a .pt file and converts its contents to an HDF5 file.
    Supports both single tensors and dictionaries of tensors.
    """
    print(f"Loading .pt file from: {pt_path}")
    data = torch.load(pt_path, map_location='cpu')
    print("File loaded successfully.")
    
    if isinstance(data, dict):
        print("Data is a dictionary; converting each tensor.")
        convert_dict_to_hdf5(data, hdf5_path, chunk_size)
    elif isinstance(data, torch.Tensor):
        print("Data is a single tensor; converting it.")
        with h5py.File(hdf5_path, "w") as f:
            convert_tensor_to_hdf5(data, f, dataset_name="u", chunk_size=chunk_size)
    else:
        raise ValueError(f"Unsupported data type: {type(data)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a large .pt file to an HDF5 file.")
    parser.add_argument("--pt_path", type=str, required=True, help="Path to the .pt file to be converted.")
    parser.add_argument("--hdf5_path", type=str, required=True, help="Output path for the HDF5 file.")
    parser.add_argument("--chunk_size", type=int, default=1,
                        help="Chunk size along the first dimension when writing to HDF5 (default: 1).")
    args = parser.parse_args()
    
    convert_pt_to_hdf5(args.pt_path, args.hdf5_path, args.chunk_size)