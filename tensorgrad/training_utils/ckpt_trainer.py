import torch
from torch.cuda import amp
from torch import nn
from timeit import default_timer
from pathlib import Path
from typing import Union
import sys

import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.nn.parallel import DistributedDataParallel as DDP

# Only import wandb and use if installed
wandb_available = False
try:
    import wandb
    wandb_available = True
except ModuleNotFoundError:
    wandb_available = False

import neuralop.mpu.comm as comm
from neuralop.losses import LpLoss

#from .training_state import load_training_state, save_training_state
from .training_state_async import load_training_state, save_training_state


class CheckpointTrainer:
    """
    A general Trainer class to train neural-operators on given datasets
    """
    def __init__(
        self,
        *,
        model: nn.Module,
        n_epochs: int,
        wandb_log: bool=False,
        device: str='cpu',
        mixed_precision: bool=False,
        data_processor: nn.Module=None,
        eval_interval: int=1,
        log_output: bool=False,
        use_distributed: bool=False,
        verbose: bool=False,
        log_train_interval: int=-1,
        log_ranks_interval: int=100,
    ):
        """
        Parameters
        ----------
        model : nn.Module
        n_epochs : int
        wandb_log : bool, default is False
            whether to log results to wandb
        device : torch.device, or str 'cpu' or 'cuda'
        mixed_precision : bool, default is False
            whether to use torch.autocast to compute mixed precision
        data_processor : DataProcessor class to transform data, default is None
            if not None, data from the loaders is transform first with data_processor.preprocess,
            then after getting an output from the model, that is transformed with data_processor.postprocess.
        eval_interval : int, default is 1
            how frequently to evaluate model and log training stats
        log_output : bool, default is False
            if True, and if wandb_log is also True, log output images to wandb
        use_distributed : bool, default is False
            whether to use DDP
        verbose : bool, default is False
        log_train_interval : int, default is 1
            how frequently to log training loss (1 = every step, N = every N steps)
        log_ranks_interval : int, default is 100
            how frequently to log rank statistics (1 = every step, N = every N steps)
        """

        self.model = model
        self.n_epochs = n_epochs
        # only log to wandb if a run is active
        self.wandb_log = False
        if wandb_available:
            self.wandb_log = (wandb_log and wandb.run is not None)
        self.eval_interval = eval_interval
        self.log_output = log_output
        self.verbose = verbose
        self.use_distributed = use_distributed
        self.device = device
        # handle autocast device
        if isinstance(self.device, torch.device):
            self.autocast_device_type = self.device.type
        else:
            if "cuda" in self.device:
                self.autocast_device_type = "cuda"
            else:
                self.autocast_device_type = "cpu"
        self.mixed_precision = mixed_precision
        self.data_processor = data_processor
        self.log_train_interval = log_train_interval
        self.log_ranks_interval = log_ranks_interval
        
      
    def train(
        self,
        train_loader,
        test_loaders,
        optimizer,
        scheduler,
        per_layer_opt: bool=False,
        regularizer=None,
        training_loss=None,
        eval_losses=None,
        save_every: int=None,
        save_best: int=None,
        save_dir: Union[str, Path]="./ckpt",
        resume_from_dir: Union[str, Path]=None,
    ):
        """Trains the given model on the given datasets.

        Parameters
        -----------
        train_loader: torch.utils.data.DataLoader
            training dataloader
        test_loaders: dict[torch.utils.data.DataLoader]
            testing dataloaders
        optimizer: torch.optim.Optimizer
            optimizer to use during training
        optimizer: torch.optim.lr_scheduler
            learning rate scheduler to use during training
        training_loss: training.losses function
            cost function to minimize
        eval_losses: dict[Loss]
            dict of losses to use in self.eval()
        save_every: int, optional, default is None
            if provided, interval at which to save checkpoints
        save_best: str, optional, default is None
            if provided, key of metric f"{loader_name}_{loss_name}"
            to monitor and save model with best eval result
            Overrides save_every and saves on eval_interval
        save_dir: str | Path, default "./ckpt"
            directory at which to save training states if
            save_every and/or save_best is provided
        resume_from_dir: str | Path, default None
            if provided, resumes training state (model, 
            optimizer, regularizer, scheduler) from state saved in
            `resume_from_dir`
        
        Returns
        -------
        all_metrics: dict
            dictionary keyed f"{loader_name}_{loss_name}"
            of metric results for last validation epoch across
            all test_loaders
            
        """
        self.train_loader_len = len(train_loader)  # Store length of train_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        if regularizer:
            self.regularizer = regularizer
        else:
            self.regularizer = None
        self.per_layer_opt = per_layer_opt
        if training_loss is None:
            training_loss = LpLoss(d=2)

        if eval_losses is None:  # By default just evaluate on the training loss
            eval_losses = dict(l2=training_loss)
        
        # Initialize wandb metrics dictionary
        self.wandb_epoch_metrics = {}

        # attributes for checkpointing
        self.save_every = save_every
        self.save_best = save_best

        self.start_epoch = 0

        if resume_from_dir is not None:
            print(resume_from_dir)
            if Path(resume_from_dir).exists():
                print(f"resuming from {resume_from_dir}...")
                self.resume_state_from_dir(resume_from_dir)
        
        # load model to device and initialize DDP within trainer
        self.model = self.model.to(self.device)
        if self.use_distributed:
            self.model = DDP(
                self.model, device_ids=[dist.get_rank()], output_device=dist.get_rank()
            )
            
        # ensure save_best is a metric we collect
        if self.save_best is not None:
            metrics = []
            for name in test_loaders.keys():
                for metric in eval_losses.keys():
                    metrics.append(f"{name}_{metric}")
            assert self.save_best in metrics,\
                f"Error: expected a metric of the form <loader_name>_<metric>, got {save_best}"
            best_metric_value = float('inf')
            # either monitor metric or save on interval, exclusive for simplicity
            self.save_every = None

        if self.verbose:
            print(f'Training on {len(train_loader.dataset)} samples')
            print(f'Testing on {[len(loader.dataset) for loader in test_loaders.values()]} samples'
                  f'         on resolutions {[name for name in test_loaders]}.')
            sys.stdout.flush()
        
        # Initialize epoch_metrics at the beginning to avoid UnboundLocalError
        epoch_metrics = {}
        
        print(f"{self.start_epoch=} {self.n_epochs=}")
        for epoch in range(self.start_epoch, self.n_epochs):
            train_err, avg_loss, avg_lasso_loss, epoch_train_time =\
                  self.train_one_epoch(epoch, train_loader, training_loss)
            torch.cuda.empty_cache()
            epoch_metrics = dict(
                train_err=train_err,
                avg_loss=avg_loss,
                avg_lasso_loss=avg_lasso_loss,
                time=epoch_train_time
            )

            if self.verbose and epoch % self.eval_interval == 0:
                lr = None
                if not self.per_layer_opt:
                    for pg in self.optimizer.param_groups:
                        lr = pg["lr"]
                self.log_training(
                    lr=lr,
                    **epoch_metrics
                )

            if epoch % self.eval_interval == 0:
                torch.cuda.empty_cache()
                # evaluate and gather metrics across each loader in test_loaders
                eval_metrics = self.evaluate_all(eval_losses=eval_losses,
                                                test_loaders=test_loaders)
                
                if self.verbose:
                    self.log_eval(eval_metrics)
                
                epoch_metrics.update(**eval_metrics)
                # save checkpoint if conditions are met
                if save_best is not None:
                    if eval_metrics[save_best] < best_metric_value:
                        best_metric_value = eval_metrics[save_best]
                        self.checkpoint(save_dir)
                torch.cuda.empty_cache()
            
            # commit wandb step
            if self.wandb_log:
                self.commit_wandb_step()
                self.wandb_epoch_metrics = {} 

            # save checkpoint if save_every and save_best is not set
            if self.save_every is not None:
                if epoch % self.save_every == 0:
                    self.checkpoint(save_dir)

        return epoch_metrics

    def train_one_epoch(self, epoch, train_loader, training_loss):
        
        """train_one_epoch trains self.model on train_loader
        for one epoch and returns training metrics

        Parameters
        ----------
        epoch : int
            epoch number
        train_loader : torch.utils.data.DataLoader
            data loader of train examples
        test_loaders : dict
            dict of test torch.utils.data.DataLoader objects

        Returns
        -------
        all_errors
            dict of all eval metrics for the last epoch
        """
        self.on_epoch_start(epoch)
        avg_loss = 0
        avg_lasso_loss = 0
        self.model.train()
        if self.data_processor:
            self.data_processor.train()
        t1 = default_timer()
        train_err = 0.0

        # track number of training examples in batch
        self.n_samples = 0

        for idx, sample in enumerate(train_loader):
            torch.cuda.empty_cache()
            loss = self.train_one_batch(idx, sample, training_loss)
            loss.backward()

            # step optimizer if no hooks are registered
            if not self.per_layer_opt:
                self.optimizer.step()

            train_err += loss.item()
            with torch.no_grad():
                avg_loss += loss.item()
                if self.regularizer:
                    avg_lasso_loss += self.regularizer.loss

                # Log the step loss if wandb logging is enabled and it's time to log
                if self.wandb_log and idx % self.log_train_interval == 0 and self.log_train_interval > 0:
                    wandb.log({
                        "train/step_loss": loss.item(),
                        "train/batch": idx,
                        "train/epoch": epoch,
                    }, step=epoch * self.train_loader_len + idx)

        if not self.per_layer_opt:
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(train_err)
            else:
                self.scheduler.step()


        epoch_train_time = default_timer() - t1

        train_err /= len(train_loader)
        avg_loss /= self.n_samples
        if self.regularizer:
            avg_lasso_loss /= self.n_samples
        else:
            avg_lasso_loss = None

        return train_err, avg_loss, avg_lasso_loss, epoch_train_time

    def evaluate_all(self, eval_losses, test_loaders):
        # evaluate and gather metrics across each loader in test_loaders
        all_metrics = {}
        for loader_name, loader in test_loaders.items():
            loader_metrics = self.evaluate(eval_losses, loader,
                                    log_prefix=loader_name)   
            all_metrics.update(**loader_metrics)
        return all_metrics
    
    def evaluate(self, loss_dict, data_loader, log_prefix="", epoch=None):
        """Evaluates the model on a dictionary of losses

        Parameters
        ----------
        loss_dict : dict of functions
          each function takes as input a tuple (prediction, ground_truth)
          and returns the corresponding loss
        data_loader : data_loader to evaluate on
        log_prefix : str, default is ''
            if not '', used as prefix in output dictionary
        epoch : int | None
            current epoch. Used when logging both train and eval
            default None
        Returns
        -------
        errors : dict
            dict[f'{log_prefix}_{loss_name}] = loss for loss in loss_dict
        """

        self.model.eval()
        if self.data_processor:
            self.data_processor.eval()

        errors = {f"{log_prefix}_{loss_name}": 0 for loss_name in loss_dict.keys()}

        
        self.n_samples = 0
        with torch.no_grad():
            for idx, sample in enumerate(data_loader):
                return_output = False
                if idx == len(data_loader) - 1:
                    return_output = True
                eval_step_losses, outs = self.eval_one_batch(sample, loss_dict, return_output=return_output)

                for loss_name, val_loss in eval_step_losses.items():
                    errors[f"{log_prefix}_{loss_name}"] += val_loss
            
        for key in errors.keys():
            errors[key] /= self.n_samples

        # on last batch, log model outputs
        if self.log_output:
            # set max resolution of image to 256
            if outs.shape[1] > 256:
                # downsample to 256 by sub-sampling
                outs = outs[:, ::outs.shape[1]//256, ::outs.shape[2]//256, :]
            errors[f"{log_prefix}_outputs"] = wandb.Image(outs)
        
        return errors
    
    def on_epoch_start(self, epoch):
        """on_epoch_start runs at the beginning
        of each training epoch. This method is a stub
        that can be overwritten in more complex cases.

        Parameters
        ----------
        epoch : int
            index of epoch

        Returns
        -------
        None
        """
        self.epoch = epoch
        return None

    def train_one_batch(self, idx, sample, training_loss):
        if not self.per_layer_opt:
            self.optimizer.zero_grad(set_to_none=True)
        if self.regularizer:
            self.regularizer.reset()

        if self.data_processor is not None:
            sample = self.data_processor.preprocess(sample)
        else:
            sample = {k: v.to(self.device) for k, v in sample.items() if torch.is_tensor(v)}
        
        batch_size = 1 if sample["y"].ndim == 2 else sample["y"].shape[0]
        self.n_samples += batch_size

        if self.mixed_precision:
            with torch.autocast(device_type=self.autocast_device_type):
                out = self.model(**sample)
        else:
            out = self.model(**sample)
        
        if self.epoch == 0 and idx == 0 and self.verbose:
            print(f"Raw outputs of shape {out.shape}")

        if self.data_processor is not None:
            out, sample = self.data_processor.postprocess(out, sample)

        loss = 0.0
        if self.mixed_precision:
            with torch.autocast(device_type=self.autocast_device_type):
                loss += training_loss(out, **sample)
        else:
            loss += training_loss(out, **sample)

        if self.regularizer:
            loss += self.regularizer.loss
        
        # check if nan in loss
        if torch.isnan(loss):
            print(f"Loss is nan at epoch {self.epoch} batch {idx}")
            print(f"Loss: {loss}")
            
        
        return loss
    
    def eval_one_batch(self,
                       sample: dict,
                       eval_losses: dict,
                       return_output: bool=False):
        """eval_one_batch runs inference on one batch
        and returns eval_losses for that batch.

        Parameters
        ----------
        sample : dict
            data batch dictionary
        eval_losses : dict
            dictionary of named eval metrics
        return_outputs : bool
            whether to return model outputs for plotting
            by default False
        Returns
        -------
        eval_step_losses : dict
            keyed "loss_name": step_loss_value for each loss name
        outputs: torch.Tensor | None
            optionally returns batch outputs
        """
        if self.data_processor is not None:
            sample = self.data_processor.preprocess(sample)
        else:
            # load data to device if no preprocessor exists
            sample = {
                k: v.to(self.device)
                for k, v in sample.items()
                if torch.is_tensor(v)
            }
            
        if sample["y"].ndim == 2:
            batch_size = 1
        else:
            batch_size = sample["y"].shape[0]
        self.n_samples += batch_size

        if self.mixed_precision:
            with torch.autocast(device_type=self.autocast_device_type):
                out = self.model(**sample)
        else:
            out = self.model(**sample)
        
        if self.data_processor is not None:
            out, sample = self.data_processor.postprocess(out, sample)
        
        eval_step_losses = {}

        for loss_name, loss in eval_losses.items():
            if self.mixed_precision:
                with torch.autocast(device_type=self.autocast_device_type):
                    val_loss = loss(out, **sample)
            else:
                val_loss = loss(out, **sample)
            eval_step_losses[loss_name] = val_loss
        
        if return_output:
            return eval_step_losses, out
        else:
            return eval_step_losses, None
        
    def plot_output_vs_ground_truth(self, output, ground_truth, epoch, idx):
        """Plot model output vs ground truth and save the plot.
        
        Parameters
        ----------
        output : torch.Tensor
            Model output tensor of shape (batch_size, ...)
        ground_truth : torch.Tensor
            Ground truth tensor of shape (batch_size, ...)
        epoch : int
            Current epoch number
        idx : int
            Current batch index
        """
        import matplotlib.pyplot as plt
        import os
        import math
        
        # Create plots directory if it doesn't exist
        os.makedirs('plots', exist_ok=True)
        
        # Convert tensors to numpy arrays
        output = output.detach().cpu().numpy()
        ground_truth = ground_truth.detach().cpu().numpy()
        
        # Get batch size
        batch_size = output.shape[0]
        
        # Calculate grid dimensions - aim for roughly square grid
        grid_size = math.ceil(math.sqrt(batch_size))
        n_rows = grid_size
        n_cols = grid_size
        
        # Create figure with subplots for each sample in batch
        fig = plt.figure(figsize=(5*n_cols, 5*n_rows))
        
        for i in range(batch_size):
            # Ground truth
            plt.subplot(n_rows, n_cols*3, i*3 + 1)
            plt.imshow(ground_truth[i].squeeze(), cmap='viridis')
            plt.colorbar()
            if i == 0:
                plt.title('Ground Truth')
            plt.axis('off')
            
            # Model output
            plt.subplot(n_rows, n_cols*3, i*3 + 2)
            plt.imshow(output[i].squeeze(), cmap='viridis')
            plt.colorbar()
            if i == 0:
                plt.title('Model Output')
            plt.axis('off')
            
            # Difference
            plt.subplot(n_rows, n_cols*3, i*3 + 3)
            diff = output[i].squeeze() - ground_truth[i].squeeze()
            plt.imshow(diff, cmap='RdBu')
            plt.colorbar()
            if i == 0:
                plt.title('Difference')
            plt.axis('off')
        
        plt.suptitle(f'Batch Comparison (Epoch {epoch}, Batch {idx})')
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(f'plots/output_vs_ground_truth_epoch{epoch}_idx{idx}.png', 
                    bbox_inches='tight', dpi=150)
        plt.close()
        
      
    def log_training(self, 
            time: float,
            avg_loss: float,
            train_err: float,
            avg_lasso_loss: float=None,
            lr: float=None
            ):
        """Basic method to log results
        from a single training epoch. 
        

        Parameters
        ----------
        time: float
            training time of epoch
        avg_loss: float
            average train_err per individual sample
        train_err: float
            train error for entire epoch
        avg_lasso_loss: float
            average lasso loss from regularizer, optional
        lr: float
            learning rate at current epoch
        """
        # accumulate info to log to wandb
        if self.wandb_log:
            values_to_log = dict(
                train_err=train_err,
                time=time,
                avg_loss=avg_loss,
                avg_lasso_loss=avg_lasso_loss,
                lr=lr)

        msg = f"[{self.epoch}] time={time:.2f}, "
        msg += f"avg_loss={avg_loss:.4f}, "
        msg += f"train_err={train_err:.4f}"
        if avg_lasso_loss is not None:
            msg += f", avg_lasso={avg_lasso_loss:.4f}"
        maxmem = torch.cuda.max_memory_reserved() / (1024 ** 3)
        msg += f", mem={maxmem:.4f}"

        print(msg)
        sys.stdout.flush()
        
        if self.wandb_log:
            wandb.log(data=values_to_log,
                      step=self.epoch+1,
                      commit=False)
    
    def log_eval(self,
                 eval_metrics: dict):
        """log_eval logs outputs from evaluation
        on all test loaders to stdout and wandb

        Parameters
        ----------
        eval_metrics : dict
            metrics collected during evaluation
            keyed f"{test_loader_name}_{metric}" for each test_loader
       
        """
        values_to_log = {}
        msg = ""
        for metric, value in eval_metrics.items():
            if isinstance(value, float) or isinstance(value, torch.Tensor):
                msg += f"{metric}={value:.4f}, "
            if self.wandb_log:
                values_to_log[metric] = value       
        
        msg = f"Eval: " + msg[:-2] # cut off last comma+space
        print(msg)
        sys.stdout.flush()

        if self.wandb_log:
            wandb.log(data=values_to_log,
                        step=self.epoch+1,
                        commit=False)
        
    
            
            
    def resume_state_from_dir(self, save_dir):
        # delegate to load_training_state:
        model, optimizer, scheduler, regularizer, epoch = load_training_state(
            save_dir=save_dir,
            save_name="model",   # or "best_model"
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            regularizer=self.regularizer,
            distributed=self.use_distributed,
            map_location=self.device
        )
        
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.regularizer = regularizer
        self.start_epoch = epoch
        
    def checkpoint(self, save_dir):
        """checkpoint saves current training state
        to a directory for resuming later.
        See neuralop.training.training_state

        Parameters
        ----------
        save_dir : str | Path
            directory in which to save training state
        """
        if self.save_best is not None:
            save_name = 'best_model'
        else:
            save_name = "model"
        
        # if self.use_distributed:
        save_training_state(save_dir=save_dir, 
                        save_name=save_name,
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        regularizer=self.regularizer,
                        epoch=self.epoch,
                        )
        

    def commit_wandb_step(self):
        if not self.wandb_log or not self.wandb_epoch_metrics:
            return
        
        metrics_to_log = {}
        
        # Process summed metrics for warm start, ranks, etc.
        for key in self.wandb_epoch_metrics:
            if key.endswith('_sum') and key != 'recon_errors_by_id_and_config':
                if 'grad_recon_err_avg/' in key:
                    # Handle reconstruction error averages
                    base_key = key.replace('_sum', '')
                    rank_str = base_key.split('/')[-1]  # Extract rank string
                    metric_name = f'grad_recon_err_avg/rank_{rank_str}'
                    count_key = f"{base_key}_count"
                elif '_mode' in key:  # Per-mode ranks
                    base_key = key.replace('_sum', '')
                    metric_name = base_key  # e.g., 'ranks/group1_param0_mode0'
                    count_key = f"{base_key}_count"
                elif 'avg_rank_sum' in key:  # Average rank
                    base_key = key.replace('_avg_rank_sum', '')
                    metric_name = f"{base_key}_avg_rank"  # e.g., 'ranks/group1_param0_avg_rank'
                    count_key = f"{base_key}_avg_count"
                elif 'correlation_sum' in key:  # Warm start correlation
                    base_key = key.replace('_correlation_sum', '')
                    metric_name = f"{base_key}_correlation"  # Changed from cosine to correlation
                    count_key = f"{base_key}_count"
                elif 'sign_flips_sum' in key:  # Warm start sign flips
                    base_key = key.replace('_sign_flips_sum', '')
                    metric_name = f"{base_key}_sign_flips"  # e.g., 'warm_start/group1_param0_sign_flips'
                    count_key = f"{base_key}_count"
                else:
                    continue
                
                if count_key in self.wandb_epoch_metrics and self.wandb_epoch_metrics[count_key] > 0:
                    try:
                        avg_value = self.wandb_epoch_metrics[key] / self.wandb_epoch_metrics[count_key]
                        metrics_to_log[metric_name] = avg_value
                    except Exception as e:
                        print(f"Error calculating avg for {key}: {str(e)}")
        
        # Process reconstruction error averages for each specific ID and rank config
        if 'recon_errors_by_id_and_config' in self.wandb_epoch_metrics:
            for id_config_key, errors in self.wandb_epoch_metrics['recon_errors_by_id_and_config'].items():
                if errors:  # Only if we have errors for this ID and rank config
                    avg_error = sum(errors) / len(errors)
                    metrics_to_log[f'grad_recon_err_avg/{id_config_key}'] = avg_error
                    
                    # Also log min and max if there are multiple values
                    if len(errors) > 1:
                        metrics_to_log[f'grad_recon_err_min/{id_config_key}'] = min(errors)
                        metrics_to_log[f'grad_recon_err_max/{id_config_key}'] = max(errors)
        
        
        # Log all other metrics directly
        for key in self.wandb_epoch_metrics:
            if key != 'recon_errors_by_id_and_config' and not key.endswith('_sum') and not key.endswith('_count') and 'grad_recon_err/' not in key:
                # Skip keys we've already processed
                if key not in metrics_to_log:
                    metrics_to_log[key] = self.wandb_epoch_metrics[key]
        
        if metrics_to_log:
            try:
                wandb.log(metrics_to_log, step=self.epoch + 1, commit=True)
            except Exception as e:
                print(f"W&B logging failed: {str(e)}")
        else:
            print("No metrics to log after processing")
        
        # Clear the epoch_metrics for the next epoch
        self.wandb_epoch_metrics = {}