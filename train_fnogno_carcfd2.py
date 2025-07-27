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
    gno_radius: float = 0.033
    gno_transform_type: str = 'linear'

class CarCFDDatasetConfig(ConfigBase):
    root: str = "data/car_cfd/processed-car-pressure-data/data"
    sdf_query_resolution: int = 32
    n_train: int = 500
    n_test: int = 111
    download: bool = False

class CarCFDOptConfig(ConfigBase):
    n_epochs: int = 301
    learning_rate: float = 0.001
    training_loss: str = "l2"
    testing_loss: str = "l2"
    weight_decay: float = 1e-4
    scheduler: str = "StepLR"
    step_size: int = 50
    gamma: float = 0.5

class Default(ConfigBase):
    n_params_baseline: Optional[Any] = None
    verbose: bool = True
    arch: str = 'fnogno'
    
    # Model configuration (nested under arch name)
    fnogno: FNOGNOConfig = FNOGNOConfig()
    
    # Other configurations
    distributed: DistributedConfig = DistributedConfig()
    opt: ConfigBase = CarCFDOptConfig()
    data: CarCFDDatasetConfig = CarCFDDatasetConfig()
    patching: PatchingConfig = PatchingConfig()
    wandb: WandbConfig = WandbConfig()

# Read the configuration
config_name = 'cfd'
from zencfg import cfg_from_commandline

config = cfg_from_commandline(Default)
config = config.to_dict()

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

#Create the optimizer
optimizer = AdamW(model.parameters(), 
                                lr=config.opt.learning_rate, 
                                weight_decay=config.opt.weight_decay)

if config.opt.scheduler == 'ReduceLROnPlateau':
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=config.opt.gamma, patience=config.opt.scheduler_patience, mode='min')
elif config.opt.scheduler == 'CosineAnnealingLR':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.opt.scheduler_T_max)
elif config.opt.scheduler == 'StepLR':
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                step_size=config.opt.step_size,
                                                gamma=config.opt.gamma)
else:
    raise ValueError(f'Got {config.opt.scheduler=}')


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