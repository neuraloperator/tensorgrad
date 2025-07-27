import sys
import os
from pathlib import Path
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
from configmypy import ConfigPipeline, YamlConfig, ArgparseConfig
import torch
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data import DistributedSampler

import wandb
from neuralop import H1Loss, LpLoss, get_model
from neuralop.data.datasets.navier_stokes import NavierStokesDataset
from neuralop.training import setup
from neuralop.utils import get_wandb_api_key, count_model_params

from tensorgrad.navier_stokes import FullSizeNavierStokes

from tensorgrad.training_utils import CheckpointTrainer
from tensorgrad.profiler_trainer import Trainer as ProfilerTrainer


from tensorgrad.setup_optimizer import setup_optimizer_and_scheduler
import random
import numpy as np

import argparse

# Add argument parsing for config file
parser = argparse.ArgumentParser()
parser.add_argument('--config_file', type=str, default='ns_tensorgrad_repro_config.yaml', 
                   help='Path to the YAML config file')
args, remaining_args = parser.parse_known_args()

# Read the configuration
config_name = "default"
pipe = ConfigPipeline(
    [
        YamlConfig(args.config_file, config_name=config_name, config_folder="./config"),
        ArgparseConfig(infer_types=True, config_name=None, config_file=None),
        YamlConfig(config_folder="./config"),
    ]
)
config = pipe.read_conf()
config_name = pipe.steps[-1].config_name

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
    if getattr(getattr(config, config.arch), "activation_checkpoint", False):
        logging_name += "_activation_ckpt"
    if config.opt.per_layer_opt:
        logging_name += "_perlayer"
    # add a tag if using job checkpointing
    if config.opt.checkpointing:
        logging_name += "_job_ckpt"

# Set-up distributed communication, if using
if config.distributed.use_distributed:
    # Set-up distributed communication, if using
    dist.init_process_group(backend='nccl')
    gpu_id = int(os.environ["LOCAL_RANK"])
    print(f"Env {gpu_id=}")
    is_logger = (gpu_id == 0)
    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}"

    print(f"\n\n xxxxxxxx init local rank {device=} xxxxxxx")
    print(f"{dist.get_rank()=}")
    print(f"{dist.get_world_size()=}")
else:
    device, is_logger = setup(config)
    
# set seed for all processes
torch.manual_seed(config.distributed.seed)
random.seed(config.distributed.seed)
np.random.seed(config.distributed.seed)




# Set up WandB logging
wandb_args = None
if config.wandb.log and is_logger:
    # check if exists or prompt for key
    if not os.path.exists("./config/wandb_api_key.txt"):
        wandb.login(key=input("Enter your WandB API key: "))
    else:
        wandb.login(key=open("./config/wandb_api_key.txt").read())
    if config.wandb.name:
        wandb_name = logging_name
    else:
        # Create a list of config values that might exist
        config_values = []
        for var in [
            'n_layers',
            'hidden_channels',
            'n_modes',
            'projection_channel_ratio',
            'factorization',
            'rank',
        ]:
            if hasattr(config.fno, var):
                config_values.append(str(getattr(config.fno, var)))
        
        # Add patching values if they exist
        if hasattr(config, 'patching'):
            if hasattr(config.patching, 'levels'):
                config_values.append(str(config.patching.levels))
            if hasattr(config.patching, 'padding'):
                config_values.append(str(config.patching.padding))
        
        wandb_name = "_".join([config_name] + config_values)
    wandb_args =  dict(
        config=config,
        name=wandb_name,
        group=config.wandb.group,
        project=config.wandb.project,
        entity=config.wandb.entity,
    )
    if config.opt.checkpointing:
        wandb_args.update(
            id=wandb_name,
            resume="allow"
        )

    if config.wandb.sweep:
        for key in wandb.config.keys():
            config.params[key] = wandb.config[key]
    wandb.init(**wandb_args)

# Make sure we only print information when needed
config.verbose = config.verbose and is_logger

# Print config to screen
if config.verbose and is_logger:
    pipe.log()
    sys.stdout.flush()

# Choose dataset class based on resolution
if not config.data.download and config.data.train_resolution != 128:
    ns_dataset = FullSizeNavierStokes(
        root_dir=config.data.root,
        n_train=config.data.n_train,
        n_tests=config.data.n_tests,
        train_resolution=config.data.train_resolution,
        test_resolutions=config.data.test_resolutions,
        batch_size=config.data.batch_size,
        test_batch_sizes=config.data.test_batch_sizes,
        encode_input=True,
        encode_output=True,
        encoding='channel-wise',
        channel_dim=1,
        max_chunk_size=config.data.max_chunk_data_fitting
    )

else:
    print("Using NavierStokesDataset for lower resolution data")
    ns_dataset = NavierStokesDataset(
        root_dir=config.data.root,
        n_train=config.data.n_train, 
        n_tests=config.data.n_tests,
        train_resolution=config.data.train_resolution,
        test_resolutions=config.data.test_resolutions,
        batch_size=config.data.batch_size,
        test_batch_sizes=config.data.test_batch_sizes, 
        encode_input=True, 
        encode_output=True, 
        encoding='channel-wise', 
        channel_dim=1,
        download=config.data.download
    )

# Print dataset sizes
try:
    print("\nDataset Sizes:")
    print(f"Training dataset size: {len(ns_dataset.train_db)}")
    for res, test_db in ns_dataset.test_dbs.items():
        print(f"Test dataset size (resolution {res}): {len(test_db)}")
except Exception as e:
    print(f"Error printing dataset sizes: {e}")

if config.distributed.use_distributed:
    train_sampler = DistributedSampler(ns_dataset.train_db, rank=gpu_id)
else:
    train_sampler = None
train_loader = DataLoader(ns_dataset.train_db,
    batch_size=config.data.batch_size,
    num_workers=config.data.num_workers,
    pin_memory=True,
    persistent_workers=True,
    sampler=train_sampler
)

test_loaders = {}
# For both FullSizeNavierStokes and NavierStokesDataset, we handle multiple resolutions
for res, test_bsize in zip(config.data.test_resolutions, config.data.test_batch_sizes):
    test_db = ns_dataset.test_dbs[res]
    if config.distributed.use_distributed:
        test_sampler = DistributedSampler(test_db, rank=gpu_id)
    else:
        test_sampler = None

    test_loaders[res] = DataLoader(test_db,
        batch_size=test_bsize,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=False,  # Disable pinned memory for validation
        persistent_workers=True,
        sampler=test_sampler
    )

data_processor = ns_dataset.data_processor
data_processor = data_processor.to(device)
if config.verbose and is_logger:
    print(f"{data_processor=}")


model = get_model(config)

model = model.to(device)
# Use distributed data parallel
if config.distributed.use_distributed:
    model = DDP(
        model, device_ids=[device.index], output_device=device.index, static_graph=True
    )
    print(f"{model.device=}")

optimizer, scheduler = setup_optimizer_and_scheduler(config, model, logging_name)



# Creating the losses
l2loss = LpLoss(d=2, p=2)
h1loss = H1Loss(d=2)

# Handle multiple losses
if isinstance(config.opt.training_loss, str):
    if config.opt.training_loss == "l2":
        train_loss = l2loss
    elif config.opt.training_loss == "h1":
        train_loss = h1loss
    elif "," in config.opt.training_loss:
        # losses are comma-separated, e.g. "l2,h1"
        loss_names = [name.strip().lower() for name in config.opt.training_loss.split(",")]
        losses = []
        for name in loss_names:
            if name == "l2":
                losses.append(l2loss)
            elif name == "h1":
                losses.append(h1loss)
            else:
                raise ValueError(f"Unknown loss name: {name}")
      
        if len(losses) == 2:
            # Create a loss that forms random convex combinations
            class ShakeShakeLoss:
                def __init__(self, losses):
                    self.losses = losses
                    self.alpha = None
                
                def __call__(self, *args, **kwargs):
                    # Sample new alpha for each call
                    self.alpha = torch.rand(1, device=args[0].device)
                    # Compute convex combination of losses
                    loss1 = self.losses[0](*args, **kwargs)
                    loss2 = self.losses[1](*args, **kwargs)
                    return self.alpha * loss1 + (1 - self.alpha) * loss2
                
            train_loss = ShakeShakeLoss(losses)
        else:
            raise ValueError("Shake-shake loss combination only supports exactly 2 losses")
    else:
        raise ValueError(
            f'Got training_loss={config.opt.training_loss} '
            f'but expected one of ["l2", "h1"] or comma-separated combination'
        )
else:
    raise ValueError("training_loss must be a string")

eval_losses = {"h1": h1loss, "l2": l2loss}

if config.verbose and is_logger:
    print("\n### MODEL ###\n", model)
    print("\n### OPTIMIZER ###\n", optimizer)
    print("\n### SCHEDULER ###\n", scheduler)
    print("\n### LOSSES ###")
    print(f"\n * Train: {train_loss}")
    print(f"\n * Test: {eval_losses}")
    print(f"\n### Beginning Training...\n")
    sys.stdout.flush()
# check if fno or tfno
arch_config = getattr(config, config.arch)
# only if fno else say false
if config.arch == 'fno':
    mixed_precision = True if config.fno.fno_block_precision in ["half", "mixed"] else False
else:
    mixed_precision= False

if config.opt.profiling:
    trainer = ProfilerTrainer(
        model=model,
        n_epochs=config.opt.n_epochs,
        device=device,
        data_processor=data_processor,
        wandb_log=config.wandb.log,
        eval_interval=config.wandb.log_test_interval,
        log_output=config.wandb.log_output,
        use_distributed=config.distributed.use_distributed,
        verbose=config.verbose and is_logger,
        mixed_precision=mixed_precision,
        max_batches=config.opt.max_batches
        )
else:
    trainer = CheckpointTrainer(
        model=model,
        n_epochs=config.opt.n_epochs,
        device=device,
        data_processor=data_processor,
        wandb_log=config.wandb.log,
        eval_interval=config.wandb.log_test_interval,
        log_output=config.wandb.log_output,
        use_distributed=config.distributed.use_distributed,
        verbose=config.verbose and is_logger,
        log_train_interval=config.wandb.get('log_train_interval', -1),
        mixed_precision=mixed_precision,
    )

# Log parameter count
if is_logger:
    n_params = count_model_params(model)

    if config.verbose:
        print(f"\nn_params: {n_params}")
        sys.stdout.flush()

    if config.wandb.log:
        current_step = wandb.run.step
        to_log = {"n_params": n_params}
        log_ranks_interval=config.wandb.get('log_ranks_interval', -1),
        if config.n_params_baseline is not None:
            to_log["n_params_baseline"] = (config.n_params_baseline,)
            to_log["compression_ratio"] = (config.n_params_baseline / n_params,)
            to_log["space_savings"] = 1 - (n_params / config.n_params_baseline)
        wandb.log(to_log,step=current_step, commit=False)
        wandb.watch(model)

# turn checkpoint names into paths
if config.opt.save_dir is not None:
    if isinstance(config.opt.save_dir, str):
        config.opt.save_dir = Path(config.opt.save_dir)
    save_dir = config.opt.save_dir /  logging_name
else:
    save_dir = None

if config.opt.resume_from_dir is not None:
    if isinstance(config.opt.resume_from_dir, str):
        config.opt.resume_from_dir = Path(config.opt.resume_from_dir)
    resume_from_dir = config.opt.resume_from_dir / logging_name
else:
    resume_from_dir = None

if config.opt.profiling:
    print("Profiling mode")
    print(f"{save_dir=}")
    print(f"{resume_from_dir=}")
    # Create the save directory if it doesn't exist
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
    else:
        # Provide a default save directory for profiling
        save_dir = Path("profiler_outputs")
        save_dir.mkdir(parents=True, exist_ok=True)
    print("Profiling mode")
    print(f"{save_dir=}")
    print(f"{resume_from_dir=}")
    trainer.train(
        train_loader=train_loader,
        test_loaders=test_loaders,
        optimizer=optimizer,
        per_layer_opt=config.opt.per_layer_opt,
        scheduler=scheduler,
        regularizer=False,
        training_loss=train_loss,
        eval_losses=eval_losses,
        save_every=config.opt.save_every,
        save_dir=save_dir,  # This must not be None for profiling
        resume_from_dir=resume_from_dir,
        run_name=logging_name
    )
else:
    trainer.train(
        train_loader=train_loader,
        test_loaders=test_loaders,
        optimizer=optimizer,
        per_layer_opt=config.opt.per_layer_opt,
        scheduler=scheduler,
        regularizer=False,
        training_loss=train_loss,
        eval_losses=eval_losses,
        save_every=config.opt.save_every,
        save_dir=save_dir,
        resume_from_dir=resume_from_dir,
    )
if config.wandb.log and is_logger:
    wandb.finish()
'''
snapshot = torch.cuda.memory._snapshot()
pprint(snapshot['segments'])
dump(snapshot, open(f'./profiler_outputs/{logging_name}_snapshot.pickle', 'wb'))'''


if config.distributed.use_distributed:
    dist.destroy_process_group()
