import torch
from torch.utils.data import DataLoader
from pathlib import Path
import time
import numpy as np
from tqdm import tqdm
import sys
from pathlib import Path
from typing import List, Union, Optional

# Add the parent directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

#from neuralop.data.datasets.hdf5_dataset import H5pyDataset
from neuralop.data.transforms.normalizers import UnitGaussianNormalizer
from neuralop.data.transforms.data_processors import DefaultDataProcessor

from tensorgrad.modified_hdf5 import H5pyDataset

class FullSizeNavierStokes(object):
    def __init__(self, 
                 root_dir: Union[Path, str],
                 n_train: Optional[int] = None,
                 n_tests: Optional[List[int]] = None,
                 train_resolution: int = 1024,
                 test_resolutions: Optional[List[int]] = None,
                 batch_size: int = 1,
                 test_batch_sizes: Optional[List[int]] = None,
                 encode_input: bool = True,
                 encode_output: bool = True,
                 encoding: str = "channel-wise",
                 channel_dim: int = 1,
                 fit_on_highest_res: bool = True,
                 max_chunk_size: int = 4000
                 ):
        
        if isinstance(root_dir, str):
            root_dir = Path(root_dir)
            
        # Set default values for test parameters if not provided
        if test_resolutions is None:
            test_resolutions = [train_resolution]
        if n_tests is None:
            n_tests = [None] * len(test_resolutions)
        if test_batch_sizes is None:
            test_batch_sizes = [1] * len(test_resolutions)
        
        # always add train_resolution to test_resolutions if not already present
        if train_resolution not in test_resolutions:
            test_resolutions.append(train_resolution)
            n_tests.append(n_tests[0] if n_tests else None)  # Use the first n_test value or None
            test_batch_sizes.append(test_batch_sizes[0] if test_batch_sizes else 1)  # Use the first batch size or 1
            
        # Ensure lists are of equal length
        assert len(test_resolutions) == len(n_tests) == len(test_batch_sizes), \
            "test_resolutions, n_tests, and test_batch_sizes must have the same length"
            
        print(f"Initializing FullSizeNavierStokes with:")
        print(f"  root_dir: {root_dir}")
        print(f"  train resolution: {train_resolution}")
        print(f"  test resolutions: {test_resolutions}")
        print(f"  test batch sizes: {test_batch_sizes}")
        print(f"  fit on highest resolution: {fit_on_highest_res}")
          
        # Check if downsampled version exists
        train_file = root_dir / f"nsforcing_train_{train_resolution}.hdf5"
        if train_file.exists():
            print(f"Using pre-downsampled training data at resolution {train_resolution}")
            self._train_db = H5pyDataset(train_file, no_downsample=True) # 1024
        else:
            print(f"Using full resolution data and downsampling to {train_resolution}")
            self._train_db = H5pyDataset(root_dir / f"nsforcing_train_1024.hdf5", resolution=train_resolution)
        
        # Get actual number of training samples
        actual_n_train = len(self._train_db)
        
        # If n_train not specified, use all available samples
        if n_train is None:
            n_train = actual_n_train
            
        # Ensure requested samples don't exceed available samples
        n_train = min(n_train, actual_n_train)
        
        print(f"  Available training samples: {actual_n_train}")
        print(f"  Using training samples: {n_train}")
        
        # Store the number of samples to use
        self._n_train = n_train
        self._n_tests = n_tests
        self._batch_size = batch_size
        self._test_batch_sizes = test_batch_sizes

        print("\nLoading training data...")
        t0 = time.time()
        
        # Load data in chunks for encoder fitting
        chunk_size = max_chunk_size  # Use 4000 samples for encoder fitting
        print(f"Loading first {chunk_size} samples for encoder fitting...")
        
        if fit_on_highest_res:
            # Load highest resolution data for fitting
            highest_res = max(test_resolutions)
            print(f"\nFitting normalizers on highest resolution: {highest_res}")
            fit_file = root_dir / f"nsforcing_train_{highest_res}.hdf5"
            if fit_file.exists():
                print(f"Using pre-downsampled data at resolution {highest_res}")
                fit_db = H5pyDataset(fit_file, no_downsample=True)
            else:
                print(f"Using full resolution data and downsampling to {highest_res}")
                fit_db = H5pyDataset(root_dir / f"nsforcing_train_1024.hdf5", resolution=highest_res)
            
            x_train = fit_db.data["x"][:chunk_size]
            y_train = fit_db.data["y"][:chunk_size]
        else:
            # Use current resolution data for fitting
            x_train = self._train_db.data["x"][:chunk_size]
            y_train = self._train_db.data["y"][:chunk_size]
        
        x_train = torch.tensor(x_train, dtype=torch.float32)
        if x_train.ndim == 3:
            x_train = x_train.unsqueeze(channel_dim)
        print(f"x_train shape: {x_train.shape}")
        
        t1 = time.time()
        print(f"Loaded x_train in {t1-t0:.2f} seconds")
        
        reduce_dims = list(range(x_train.ndim))
        reduce_dims.pop(channel_dim)
        
        if encode_input:
            print("\nFitting input encoder...")
            t2 = time.time()
            input_encoder = UnitGaussianNormalizer(dim=reduce_dims)
            input_encoder.fit(x_train)
            t3 = time.time()
            print(f"Input encoder fitted in {t3-t2:.2f} seconds")
            del x_train
        else:
            input_encoder = None

        print("\nLoading y_train...")
        y_train = torch.tensor(y_train, dtype=torch.float32)
        if y_train.ndim == 3:
            y_train = y_train.unsqueeze(channel_dim)
        print(f"y_train shape: {y_train.shape}")

        if encode_output:
            print("\nFitting output encoder...")
            t4 = time.time()
            output_encoder = UnitGaussianNormalizer(dim=reduce_dims)
            output_encoder.fit(y_train)
            t5 = time.time()
            print(f"Output encoder fitted in {t5-t4:.2f} seconds")
            del y_train
        else:
            output_encoder = None
        
        # Create data processor with CPU-based normalizers
        self.data_processor = DefaultDataProcessor(
            in_normalizer=input_encoder,
            out_normalizer=output_encoder
        )
        # Load test datasets
        self._test_dbs = {}
        for res, n_test in zip(test_resolutions, n_tests):
            print(f"\nLoading test database for resolution {res}...")
            # Check if downsampled version exists
            test_file = root_dir / f"nsforcing_test_{res}.hdf5"
            if test_file.exists():
                print(f"Using pre-downsampled test data at resolution {res}")
                test_db = H5pyDataset(test_file, no_downsample=True)
            else:
                print(f"Using full resolution test data and downsampling to {res}")
                test_db = H5pyDataset(root_dir / f"nsforcing_test_1024.hdf5", resolution=res)
            
            # Get actual number of test samples
            actual_n_test = len(test_db)
            
            # If n_test not specified, use all available samples
            if n_test is None:
                n_test = actual_n_test
                
            # Ensure requested samples don't exceed available samples
            n_test = min(n_test, actual_n_test)
            
            print(f"  Available test samples: {actual_n_test}")
            print(f"  Using test samples: {n_test}")
            
            self._test_dbs[res] = test_db
        
        print("\nDataset initialization complete!")
        print(f"Total initialization time: {t5-t0:.2f} seconds")
    
    
        
    
    @property
    def train_db(self):
        return self._train_db

    @property
    def test_dbs(self):
        return self._test_dbs
        
    @property
    def n_train(self):
        return self._n_train
        
    @property
    def n_tests(self):
        return self._n_tests
        
    @property
    def batch_size(self):
        return self._batch_size
        
    @property
    def test_batch_sizes(self):
        return self._test_batch_sizes
    
if __name__ == "__main__":
    from neuralop.layers.embeddings import GridEmbedding2D
    pos_embed = GridEmbedding2D()
    # get current directory
    root_dir = Path(__file__).parent
    ns_dataset = FullSizeNavierStokes(root_dir="data/navier_stokes/ns_data/16traj_t4_T4", res_train=1024, res_test=1024)
    print(f"{len(ns_dataset.test_dbs)=}")
    train_loader = DataLoader(ns_dataset.test_dbs[1024], batch_size=1)
    print(train_loader)
    dproc = ns_dataset.data_processor
    print(dproc) 
    
    print("\nTesting data processor:")
    print("=" * 50)
    for idx, batch in enumerate(train_loader):
        print(f"\nBatch {idx}:")
        print("-" * 30)
        
        # Print raw input statistics
        print("\nRaw input statistics:")
        print(f"x shape: {batch['x'].shape}")
        print(f"x mean: {torch.mean(batch['x']):.4f}")
        print(f"x std: {torch.std(batch['x']):.4f}")
        print(f"x min: {torch.min(batch['x']):.4f}")
        print(f"x max: {torch.max(batch['x']):.4f}")
        
        print(f"\ny shape: {batch['y'].shape}")
        print(f"y mean: {torch.mean(batch['y']):.4f}")
        print(f"y std: {torch.std(batch['y']):.4f}")
        print(f"y min: {torch.min(batch['y']):.4f}")
        print(f"y max: {torch.max(batch['y']):.4f}")
        
        # Process the batch
        processed = dproc.preprocess(batch)
        
        # Print processed statistics
        print("\nProcessed statistics:")
        print(f"x mean: {torch.mean(processed['x']):.4f}")
        print(f"x std: {torch.std(processed['x']):.4f}")
        print(f"x min: {torch.min(processed['x']):.4f}")
        print(f"x max: {torch.max(processed['x']):.4f}")
        
        print(f"\ny mean: {torch.mean(processed['y']):.4f}")
        print(f"y std: {torch.std(processed['y']):.4f}")
        print(f"y min: {torch.min(processed['y']):.4f}")
        print(f"y max: {torch.max(processed['y']):.4f}")
        
        # Test postprocessing
        print("\nTesting postprocessing:")
        # Create a dummy model output with same shape as y
        dummy_output = torch.randn_like(processed['y'])
        postprocessed_output, postprocessed_sample = dproc.postprocess(dummy_output, processed)
        
        print(f"Postprocessed output mean: {torch.mean(postprocessed_output):.4f}")
        print(f"Postprocessed output std: {torch.std(postprocessed_output):.4f}")
        print(f"Postprocessed output min: {torch.min(postprocessed_output):.4f}")
        print(f"Postprocessed output max: {torch.max(postprocessed_output):.4f}")
        
        # Test embedding
        print("\nTesting embedding:")
        embedded = pos_embed(processed['x'])
        print(f"Embedded shape: {embedded.shape}")
        
        if idx == 1:
            break