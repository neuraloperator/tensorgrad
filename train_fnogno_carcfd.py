import torch
import wandb
import sys
from neuralop.training import setup, AdamW
from neuralop import get_model
from neuralop.utils import get_wandb_api_key
from neuralop.losses.data_losses import LpLoss
from neuralop.training.trainer import Trainer
from neuralop.data.datasets import CarCFDDataset
from neuralop.data.transforms.data_processors import DataProcessor
from copy import deepcopy
from typing import Any, List, Optional

# Configuration classes
from zencfg import ConfigBase

# Import TensorGRaD optimizer setup
from tensorgrad.setup_optimizer import setup_optimizer_and_scheduler


class DistributedConfig(ConfigBase):
    use_distributed: bool = False
    seed: int = 666

class WandbConfig(ConfigBase):
    log: bool = True
    name: str = ""
    group: str = "car-cfd-fnogno"
    project: str = "ginofno"
    entity: str = "sloeschcke"
    sweep: bool = False
    log_output: bool = True
    eval_interval: int = 1
    log_train_interval: int = -1  # -1 means no logging and 1 means every step, N means every N steps
    log_ranks_interval: int = -1  # -1 means no logging and 1 means every step, N means every N steps

class PatchingConfig(ConfigBase):
    levels: int = 0
    padding: int = 0
    stitching: bool = False

class FNOGNOConfig(ConfigBase):
    data_channels: int = 1
    out_channels: int = 1
    fno_n_modes: List[int] = [16, 16, 16]
    fno_hidden_channels: int = 64
    fno_use_channel_mlp: bool = True
    fno_norm: str = 'instance_norm'
    fno_factorization: str = 'None'
    fno_rank: float = 0.4
    fno_domain_padding: float = 0.125
    fno_channel_mlp_expansion: float = 1.0
    gno_coord_dim: int = 3
    gno_coord_embed_dim: int = 16
    gno_radius: float = 0.055  # Changed from 0.033 to 0.055 to match paper's recommendation
    gno_transform_type: str = 'linear'

class CarCFDDatasetConfig(ConfigBase):
    root: str = "data/car_cfd/processed-car-pressure-data/data"
    sdf_query_resolution: int = 64  # Changed from 32 to 64 to match paper's best settings
    n_train: int = 500
    n_test: int = 111
    download: bool = False
    batch_size: int = 1

class CarCFDOptConfig(ConfigBase):
    # Basic training parameters
    n_epochs: int = 301
    learning_rate: float = 0.00025  # 2.5e-4 to match paper
    training_loss: str = "l2"
    testing_loss: str = "l2"
    weight_decay: float = 0.00025  # 2.5e-4 to match paper
    
    # Checkpointing and profiling
    checkpointing: bool = False
    profiling: bool = False
    max_batches: Optional[int] = None
    resume_from_dir: str = "./ckpts"
    save_dir: str = "./ckpts"
    save_every: int = 10
    
    # Mixed precision and complex support
    amp_autocast: bool = False
    enforce_full_complex_precision: bool = False  # Changed to False to disable complex support
    
    # Scheduler parameters
    scheduler: str = "CosineAnnealingLR"  # Or 'StepLR' OR 'ReduceLROnPlateau'
    step_size: int = 100
    gamma: float = 0.5
    scheduler_T_max: int = 301  # For cosine only, typically take n_epochs
    scheduler_patience: int = 50  # For ReduceLROnPlateau only
    
    # TensorGRaD parameters
    tensorgrad: bool = False
    tensorgrad_sum_lambda_sparse: float = 1.0  # 1 means that low ranks and sparse components are weighted equally
    svd_type: str = 'truncated_svd'  # Options: 'randomized_svd' or 'truncated_svd'
    n_iter_max_tucker: int = 1
    optimizer_type: str = 'adamw'  # or 'tensorgrad' for composite projectors
    galore_2d_proj_type: str = 'left'
    
    # Galore parameters
    tucker_warm_restart: bool = True
    per_layer_opt: bool = False
    activation_checkpoint: bool = False
    naive_galore: bool = False
    first_dim_rollup: int = 1
    adamw_support_complex: bool = False  # Changed to False to disable complex support
    
    # Scheduler update gap parameters
    update_proj_gap: int = 100  # start gap
    update_proj_gap_end: int = 100  # end gap
    update_proj_gap_mode: str = 'fixed'  # mode - fixed, linear, exponential
    
    # Minimum dimensions for tensorgrad
    min_dims_for_tensorgrad: int = 4
    
    # Base parameters (used for first projector in tensorgrad, or single projector otherwise)
    proj_type: str = "low_rank"  # Options: unstructured_sparse, structured_sparse, low_rank
    rank: List[float] = [1.0]  # Used if projector is low_rank
    sparse_ratio: float = 0.5  # Used if projector is sparse (changed from list to float)
    sparse_type: str = "randk"  # Options: randk,topk,probability
    scale: float = 1.0
    scale_by_mask_ratio: bool = False  # if True and sparse projector is used, scale will be multiplied by the mask ratio
    
    # Second projector parameters (only used if optimizer_type is tensorgrad)
    second_proj_type: str = "unstructured_sparse"  # Options: unstructured_sparse, structured_sparse, low_rank
    second_sparse_ratio: float = 0.05  # Used if second projector is sparse (changed from list to float)
    second_sparse_type: str = "randk"  # Options: randk,topk,probability
    second_scale: float = 1.0
    second_rank: List[float] = [1.0]  # Used if second projector is low_rank
    second_scale_by_mask_ratio: bool = False  # if True and sparse projector is used, scale will be multiplied by the mask ratio
    
    reset_sparse_optimizer_states: bool = True

class Default(ConfigBase):
    n_params_baseline: Optional[Any] = None
    verbose: bool = True
    arch: str = 'fnogno'
    
    # Model configuration (nested under arch name)
    fnogno: FNOGNOConfig = FNOGNOConfig()
    
    # Other configurations
    distributed: DistributedConfig = DistributedConfig()
    opt: CarCFDOptConfig = CarCFDOptConfig()
    data: CarCFDDatasetConfig = CarCFDDatasetConfig()
    patching: PatchingConfig = PatchingConfig()
    wandb: WandbConfig = WandbConfig()

# Read the configuration
config_name = 'cfd'
from zencfg import cfg_from_commandline

config = cfg_from_commandline(Default)
config = config.to_dict()

# Print current configuration for transparency
print("=" * 80)
print("FNOGNO CarCFD Training Configuration:")
print("=" * 80)
print(f"Architecture: {config['arch']}")
print(f"Dataset: {config['data']['n_train']} train, {config['data']['n_test']} test samples")
print(f"Query resolution: {config['data']['sdf_query_resolution']}")
print(f"Data root: {config['data']['root']}")
print()
print("Training Parameters:")
print(f"  Learning rate: {config['opt']['learning_rate']}")
print(f"  Epochs: {config['opt']['n_epochs']}")
print(f"  Weight decay: {config['opt']['weight_decay']}")
print(f"  Training loss: {config['opt']['training_loss']}")
print(f"  Testing loss: {config['opt']['testing_loss']}")
print(f"  Scheduler: {config['opt']['scheduler']}")
print(f"  Step size: {config['opt']['step_size']}")
print(f"  Gamma: {config['opt']['gamma']}")
print()
print("FNOGNO Model Parameters:")
print(f"  FNO hidden channels: {config['fnogno']['fno_hidden_channels']}")
print(f"  FNO modes: {config['fnogno']['fno_n_modes']}")
print(f"  FNO factorization: {config['fnogno']['fno_factorization']}")
print(f"  FNO rank: {config['fnogno']['fno_rank']}")
print(f"  GNO radius: {config['fnogno']['gno_radius']}")
print(f"  GNO coord embed dim: {config['fnogno']['gno_coord_embed_dim']}")
print()
print("Optimizer Parameters:")
print(f"  TensorGRaD enabled: {config['opt']['tensorgrad']}")
print(f"  Optimizer type: {config['opt']['optimizer_type']}")
print(f"  Projection type: {config['opt']['proj_type']}")
print(f"  Rank: {config['opt']['rank']}")
print(f"  Sparse ratio: {config['opt']['sparse_ratio']}")
print(f"  Naive Galore: {config['opt']['naive_galore']}")
print(f"  First dim rollup: {config['opt']['first_dim_rollup']}")
print()
print("WandB Configuration:")
print(f"  Project: {config['wandb']['project']}")
print(f"  Entity: {config['wandb']['entity']}")
print(f"  Group: {config['wandb']['group']}")
print(f"  Name: {config['wandb']['name']}")
print(f"  Logging enabled: {config['wandb']['log']}")
print("=" * 80)

#Set-up distributed communication, if using
device, is_logger = setup(config)

print('device', device)

# if the model's number of modes is greater than the query res, 
# shrink the model to avoid an ill-posed FNO

if config.data.sdf_query_resolution < config.fnogno.fno_n_modes[0]:
    config.fnogno.fno_n_modes = [config.data.sdf_query_resolution]*3

#Set up WandB logging
wandb_init_args = {}
config_name = 'car-pressure'
if config.wandb.log and is_logger:
    # Try to get WandB API key, with fallback to manual input
    try:
        wandb.login(key=get_wandb_api_key("./config/wandb_api_key.txt"))
    except (KeyError, FileNotFoundError):
        wandb.login(key=input("Enter your WandB API key: "))
    if config.wandb.name:
        wandb_name = config.wandb.name
    else:
        wandb_name = '_'.join(
            f'{var}' for var in [config_name, config.data.sdf_query_resolution])

    wandb_init_args = dict(config=config, 
                           name=wandb_name, 
                           group=config.wandb.group,
                           project=config.wandb.project,
                           entity=config.wandb.entity)

    if config.wandb.sweep:
        for key in wandb.config.keys():
            config.params[key] = wandb.config[key]
    wandb.init(**wandb_init_args)

#Load CFD body data
data_module = CarCFDDataset(root_dir=config.data.root, 
                             query_res=[config.data.sdf_query_resolution]*3, 
                             n_train=config.data.n_train, 
                             n_test=config.data.n_test, 
                             download=True
                             )


train_loader = data_module.train_loader(batch_size=1, shuffle=True)
test_loader = data_module.test_loader(batch_size=1, shuffle=False)

model = get_model(config)

print(model)

# Create logging name for optimizer
logging_name = ""
if config.wandb.name:
    logging_name += config.wandb.name
if config.opt.tensorgrad:
    if config.opt.naive_galore:
        logging_name += "_matrixgalore_r" + str(config.opt.rank).replace('[', '').replace(']', '')
        if config.opt.adamw_support_complex:
            logging_name += "_cplx"
        else:
            logging_name += "_real_only"
        logging_name += "_rollup_" + str(config.opt.first_dim_rollup)
    else:
        logging_name += "_tensorgrad_" 

#Create the optimizer
optimizer, scheduler = setup_optimizer_and_scheduler(config, model, logging_name)


l2loss = LpLoss(d=2,p=2)

if config.opt.training_loss == 'l2':
    train_loss_fn = l2loss
else: 
    raise ValueError(f'Got {config.opt.training_loss=}')

if config.opt.testing_loss == 'l2':
    test_loss_fn = l2loss
else:
    raise ValueError(f'Got {config.opt.testing_loss=}')

# Handle data preprocessing to FNOGNO 

class CFDDataProcessor(DataProcessor):
    """
    Implements logic to preprocess data/handle model outputs
    to train an FNOGNO on the CFD car-pressure dataset
    """

    def __init__(self, normalizer, device='cuda'):
        super().__init__()
        self.normalizer = normalizer
        self.device = device
        self.model = None

    def preprocess(self, sample):
        # Turn a data dictionary returned by MeshDataModule's DictDataset
        # into the form expected by the FNOGNO
        
        in_p = sample['query_points'].squeeze(0).to(self.device)
        out_p = sample['centroids'].squeeze(0).to(self.device)

        f = sample['distance'].squeeze(0).to(self.device)

        weights = sample['triangle_areas'].squeeze(0).to(self.device)

        #Output data
        truth = sample['press'].squeeze(0).unsqueeze(-1)

        # Take the first 3682 vertices of the output mesh to correspond to pressure
        output_vertices = truth.shape[1]
        if out_p.shape[0] > output_vertices:
            out_p = out_p[:output_vertices,:]

        truth = truth.to(device)

        inward_normals = -sample['triangle_normals'].squeeze(0).to(self.device)
        flow_normals = torch.zeros((sample['triangle_areas'].shape[1], 3)).to(self.device)
        flow_normals[:,0] = -1.0
        batch_dict = dict(in_p = in_p,
                        out_p=out_p,
                        f=f,
                        y=truth,
                        inward_normals=inward_normals,
                        flow_normals=flow_normals,
                        flow_speed=None,
                        vol_elm=weights,
                        reference_area=None)

        sample.update(batch_dict)
        return sample
    
    def postprocess(self, out, sample):
        if not self.training:
            out = self.normalizer.inverse_transform(out)
            y = self.normalizer.inverse_transform(sample['y'].squeeze(0))
            sample['y'] = y

        return out, sample
    
    def to(self, device):
        self.device = device
        self.normalizer = self.normalizer.to(device)
        return self
    
    def wrap(self, model):
        self.model = model

    def forward(self, sample):
        sample = self.preprocess(sample)
        out = self.model(sample)
        out, sample = self.postprocess(out, sample)
        return out, sample

output_encoder = deepcopy(data_module.normalizers['press']).to(device)
data_processor = CFDDataProcessor(normalizer=output_encoder, device=device)

trainer = Trainer(model=model, 
                  n_epochs=config.opt.n_epochs,
                  data_processor=data_processor,
                  device=device,
                  wandb_log=config.wandb.log,
                  verbose=is_logger
                  )

if config.wandb.log:
    wandb.log({'time_to_distance': data_module.time_to_distance}, commit=False)

trainer.train(
              train_loader=train_loader,
              test_loaders={'':test_loader},
              optimizer=optimizer,
              scheduler=scheduler,
              training_loss=train_loss_fn,
              #eval_losses={config.opt.testing_loss: test_loss_fn, 'drag': DragLoss},
              eval_losses={config.opt.testing_loss: test_loss_fn},
              regularizer=None,)