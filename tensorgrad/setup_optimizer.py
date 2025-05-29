import torch
from neuralop.training import AdamW as DefaultComplexAdam
from torch_optimizer import Lamb
from torch.optim import SGD

# Base imports
from tensorgrad.adamw import AdamW
from tensorgrad.tensorgrad import TensorGRaD
from tensorgrad.training_utils import get_scheduler
from tensorgrad.adam_full import AdamW as AdamWFull

def setup_optimizer_and_scheduler(config, model, logging_name):
    """Set up optimizer based on whether galore is enabled"""
    # Handle conditional imports based on config
    if config.opt.optimizer_type in ["tensorgrad", "tensorgrad_sum"]:
        from tensorgrad.tensorgrad import TensorGRaD
    else:
        from tensorgrad.adamw import AdamW

    if config.opt.tensorgrad:
        tensorgrad_params = []
        if hasattr(model, 'fno_blocks'):
            if config.distributed.use_distributed:
                tensorgrad_params.extend(list(model.module.fno_blocks.convs.parameters()))
            else:
                tensorgrad_params.extend(list(model.fno_blocks.convs.parameters()))
        else:
            # Handle the case when model structure is different (like in Burgers)
            for name, param in model.named_parameters():
                if 'convs' in name and not 'lifting' in name:
                    tensorgrad_params.append(param)
                    
        # First parameter is usually the lifting layer, we don't want to apply tensorgrad to it
        if len(tensorgrad_params) > 0:
            tensorgrad_params.pop(0)
            
        # if p has less params that 1000 then remove it
        tensorgrad_params = [p for p in tensorgrad_params if p.numel() > 1000] # remove the small params where no real memory is saved
        id_tensorgrad_params = [id(p) for p in tensorgrad_params]
        regular_params = [p for p in model.parameters() if id(p) not in id_tensorgrad_params]

        # Create the tensorgrad parameter group configuration
        tensorgrad_param_group = {
            'params': tensorgrad_params,
            'type': "tucker",
            'rank': config.opt.rank,
            'dim': 5,
            'optimizer_type': config.opt.optimizer_type,
            'scale': config.opt.scale,
            'proj_type': config.opt.proj_type,
            'galore_2d_proj_type': config.opt.galore_2d_proj_type,
            'sparse_ratio': config.opt.sparse_ratio,
            'sparse_type': config.opt.sparse_type,
            'scale_by_mask_ratio': config.opt.scale_by_mask_ratio,
            'reset_optimizer_states': config.opt.reset_optimizer_states,
            'enforce_full_complex_precision': getattr(config.opt, 'enforce_full_complex_precision', False),
            'svd_type': getattr(config.opt, 'svd_type', 'truncated_svd'),
            
            # Second projector parameters
            'second_proj_type': config.opt.second_proj_type,
            'second_sparse_ratio': config.opt.second_sparse_ratio,
            'second_sparse_type': config.opt.second_sparse_type,
            'second_scale': config.opt.second_scale,
            'second_rank': config.opt.second_rank,
            'second_scale_by_mask_ratio': config.opt.second_scale_by_mask_ratio,
            
            # Scheduler update gap parameters
            'update_proj_gap': config.opt.update_proj_gap,
            'update_proj_gap_end': config.opt.update_proj_gap_end,
            'update_proj_gap_mode': config.opt.update_proj_gap_mode,
            'batch_size': config.data.batch_size,
            'epochs': config.opt.n_epochs,
            'scheduler_T_max': config.opt.scheduler_T_max,
            'training_samples': config.data.n_train,
            
            # Tucker parameters
            'n_iter_max_tucker': config.opt.n_iter_max_tucker,
            'tucker_warm_restart': config.opt.tucker_warm_restart,
            
            'log_ranks_interval': config.wandb.log_ranks_interval,
            
            'lambda_sparse': config.opt.tensorgrad_sum_lambda_sparse,
        }
        
        param_groups = [
            {'params': regular_params},
            tensorgrad_param_group
        ]

        # Common optimizer arguments
        optimizer_args = {
            'lr': config.opt.learning_rate,
            'matrix_only': config.opt.naive_galore,
            'support_complex': config.opt.adamw_support_complex,
            'run_name': logging_name,
            'enforce_full_complex_precision': config.opt.enforce_full_complex_precision,
        }

        if config.opt.optimizer_type in ["tensorgrad", "tensorgrad_sum"]:
            optimizer_args['use_sum'] = (config.opt.optimizer_type == "tensorgrad_sum")
            optimizer = TensorGRaD(param_groups, **optimizer_args)
        else:
            # add first_dim_rollup to optimizer_args
            optimizer_args['first_dim_rollup'] = config.opt.first_dim_rollup
            optimizer = AdamW(param_groups, **optimizer_args)
    else:
        param_groups = model.parameters()
        if config.opt.optimizer_type == "adamw_full": # this enforces full precision for states
            print("Using full-precision AdamW optimizer")
            optimizer = AdamWFull(
                param_groups,
                lr=config.opt.learning_rate,
                weight_decay=config.opt.weight_decay,
            )
        elif config.opt.optimizer_type == "sgd":
            print("Using SGD optimizer")
            optimizer = SGD(
                param_groups,
                lr=config.opt.learning_rate,
                weight_decay=config.opt.weight_decay,
            )
        else:
            print("Using complex Adam optimizer")
            # Create the optimizer
            optimizer = DefaultComplexAdam(
                model.parameters(),
                lr=config.opt.learning_rate,
                weight_decay=config.opt.weight_decay,
            )
    
    # Set up scheduler for non-per-layer optimization
    scheduler = get_scheduler(
        scheduler_name=config.opt.scheduler,
        optimizer=optimizer,
        gamma=config.opt.gamma,
        patience=config.opt.scheduler_patience,
        T_max=config.opt.scheduler_T_max,
        step_size=config.opt.step_size
    )
    
    return optimizer, scheduler