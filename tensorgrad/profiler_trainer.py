import torch
from torch.cuda import amp
from torch import nn
from timeit import default_timer
from pathlib import Path
from typing import Union
import sys
import wandb

from neuralop.losses import LpLoss
#from neuralop.training.training_state import load_training_state, save_training_state
from .training_utils.training_state_async import load_training_state, save_training_state
from torch.profiler import profile, record_function, ProfilerActivity
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .mem_trace import trace_handler, trace_handler_obj

class Trainer:
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
        amp_autocast: bool=False,
        data_processor: nn.Module=None,
        eval_interval: int=1,
        log_output: bool=False,
        use_distributed: bool=False,
        verbose: bool=False,
        mixed_precision: bool=False,
        max_batches: int=None,
    ):
        """
        Parameters
        ----------
        model : nn.Module
        n_epochs : int
        wandb_log : bool, default is False
            whether to log results to wandb
        device : str 'cpu' or 'cuda'
        amp_autocast : bool, default is False
            whether to use torch.amp automatic mixed precision
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
        mixed_precision : bool, default is False
            whether to use torch.autocast to compute mixed precision
        """

        self.model = model
        self.n_epochs = n_epochs
        # only log to wandb if a run is active
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
        self.amp_autocast = amp_autocast
        self.data_processor = data_processor
        self.mixed_precision = mixed_precision
        self.max_batches = max_batches
        if self.max_batches is not None:
            # check if string
            if isinstance(self.max_batches, str):
                self.max_batches = int(self.max_batches)
            print(f"Will stop after {self.max_batches} batches for profiling")

    def train(
        self,
        train_loader,
        test_loaders,
        optimizer,
        scheduler,
        per_layer_opt: bool=False,
        run_name: str=None,
        out_dir: str=None,
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
        print("IN PROFILER TRAINER")
        print(f"{save_dir=}")
        print(f"{resume_from_dir=}")
        
            
        self.optimizer = optimizer
        self.per_layer_opt = per_layer_opt
        self.scheduler = scheduler
        if regularizer:
            self.regularizer = regularizer
        else:
            self.regularizer = None

        if training_loss is None:
            training_loss = LpLoss(d=2)

        if eval_losses is None:  # By default just evaluate on the training loss
            eval_losses = dict(l2=training_loss)
        
        # accumulated wandb metrics
        self.wandb_epoch_metrics = None

        # attributes for checkpointing
        self.save_every = save_every
        self.save_best = save_best
        if resume_from_dir is not None:
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
        
        if self.verbose:
            tracer = trace_handler_obj(run_name, out_dir=save_dir)
        else:
            tracer=None
        print(f"{tracer=}")
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            # In this example with wait=1, warmup=1, active=2, repeat=1,
            # profiler will skip the first step/iteration,
            # start warming up on the second, record
            # the third and the forth iterations,
            # after which the trace will become available
            # and on_trace_ready (when set) is called;
            # the cycle repeats starting with the next step
            schedule=torch.profiler.schedule(wait=2, warmup=1, active=5, repeat=1),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            on_trace_ready=tracer,
            experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True)
            ) as prof:
            
            for epoch in range(self.n_epochs):
                torch.cuda.empty_cache()
                #prof.step()
                train_err, avg_loss, avg_lasso_loss, epoch_train_time =\
                    self.train_one_epoch(epoch, train_loader, training_loss, prof)
                epoch_metrics = dict(
                    train_err=train_err,
                    avg_loss=avg_loss,
                    avg_lasso_loss=avg_lasso_loss,
                    epoch_train_time=epoch_train_time
                )
                if self.eval_interval > 0 and epoch % self.eval_interval == 0:
                    print(f"Evaluating at epoch {epoch}")
                    torch.cuda.empty_cache()
                    # evaluate and gather metrics across each loader in test_loaders
                    eval_metrics = self.evaluate_all(epoch=epoch,
                                                    eval_losses=eval_losses,
                                                    test_loaders=test_loaders)

                    epoch_metrics.update(**eval_metrics)
                    # save checkpoint if conditions are met
                    if save_best is not None:
                        if eval_metrics[save_best] < best_metric_value:
                            best_metric_value = eval_metrics[save_best]
                            self.checkpoint(save_dir)
                    torch.cuda.empty_cache()

                # save checkpoint if save_every and save_best is not set
                if self.save_every is not None:
                    if epoch % self.save_every == 0:
                        self.checkpoint(save_dir)

            return epoch_metrics

    def train_one_epoch(self, epoch, train_loader, training_loss, prof=None):
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
        prof : torch.profiler.profile, optional
            profiler object

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
        
        batch_count = 0
        early_stop = False

        for idx, sample in enumerate(train_loader):
            if prof is not None:
                prof.step()
                
            loss = self.train_one_batch(idx, sample, training_loss)
            
            with record_function("### loss backward ###"):
                loss.backward()

            # step optimizer if no hooks are registered
            if not self.per_layer_opt:
                with record_function("### opt step ###"):
                    # Simply step the optimizer without moving the model
                    self.optimizer.step()
            train_err += loss.item()

            with torch.no_grad():
                avg_loss += loss.item()
                if self.regularizer:
                    avg_lasso_loss += self.regularizer.loss
                    
            batch_count += 1
            
            if self.max_batches is not None and batch_count >= self.max_batches:
                print(f"Stopping after {batch_count} batches due to max_batches limit")
                early_stop = True
                break

        if not self.per_layer_opt:
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(train_err)
            else:
                self.scheduler.step()

        epoch_train_time = default_timer() - t1

        # Normalize by actual number of batches processed
        num_batches = batch_count if early_stop else len(train_loader)
        train_err /= num_batches
        avg_loss /= self.n_samples if self.n_samples > 0 else 1
        if self.regularizer:
            avg_lasso_loss /= self.n_samples if self.n_samples > 0 else 1
        else:
            avg_lasso_loss = None
        
        lr = None
        if not self.per_layer_opt:
            for pg in self.optimizer.param_groups:
                lr = pg["lr"]
        if self.verbose and self.eval_interval > 0 and epoch % self.eval_interval == 0:
            self.log_training(
                epoch=epoch,
                time=epoch_train_time,
                avg_loss=avg_loss,
                train_err=train_err,
                avg_lasso_loss=avg_lasso_loss,
                lr=lr
            )

        return train_err, avg_loss, avg_lasso_loss, epoch_train_time

    def evaluate_all(self, epoch, eval_losses, test_loaders):
        # evaluate and gather metrics across each loader in test_loaders
        all_metrics = {}
        for loader_name, loader in test_loaders.items():
            loader_metrics = self.evaluate(eval_losses, loader,
                                    log_prefix=loader_name)   
            all_metrics.update(**loader_metrics)
        if self.verbose:
            self.log_eval(epoch=epoch,
                        eval_metrics=all_metrics)
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
        """Run one batch of input through model
           and return training loss on outputs

        Parameters
        ----------
        idx : int
            index of batch within train_loader
        sample : dict
            data dictionary holding one batch

        Returns
        -------
        loss: float | Tensor
            float value of training loss
        """

        if not self.per_layer_opt:
            self.optimizer.zero_grad(set_to_none=True)
        if self.regularizer:
            self.regularizer.reset()

        if self.data_processor is not None:
            sample = self.data_processor.preprocess(sample)
        else:
            # load data to device if no preprocessor exists
            sample = {
                k: v.to(self.device)
                for k, v in sample.items()
                if torch.is_tensor(v)
            }
        self.n_samples += sample["y"].shape[0]

        with record_function("### model forward ###"):
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
        with record_function("## loss calculation ##"):
            if self.mixed_precision:
                with torch.autocast(device_type=self.autocast_device_type):
                    loss += training_loss(out, **sample)
            else:
                loss += training_loss(out, **sample)

            if self.regularizer:
                loss += self.regularizer.loss
        
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

        self.n_samples += sample["y"].size(0)

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
    
    def log_training(self, 
            epoch:int,
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
        epoch: int
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

        msg = f"[{epoch}] time={time:.2f}, "
        msg += f"avg_loss={avg_loss:.4f}, "
        msg += f"train_err={train_err:.4f}"
        if avg_lasso_loss is not None:
            msg += f", avg_lasso={avg_lasso_loss:.4f}"

        print(msg)
        sys.stdout.flush()
        
        if self.wandb_log:
            wandb.log(data=values_to_log,
                      step=epoch+1,
                      commit=False)
    
    def log_eval(self,
                 epoch: int,
                 eval_metrics: dict):
        """log_eval logs outputs from evaluation
        on all test loaders to stdout and wandb

        Parameters
        ----------
        epoch : int
            current training epoch
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
                      step=epoch+1,
                      commit=True)

    def resume_state_from_dir(self, save_dir):
        """
        Resume training from save_dir created by `neuralop.training.save_training_state`
        
        Params
        ------
        save_dir: Union[str, Path]
            directory in which training state is saved
            (see neuralop.training.training_state)
        """
        if isinstance(save_dir, str):
            save_dir = Path(save_dir)

        # check for save model exists
        if (save_dir / "best_model_state_dict.pt").exists():
            save_name = "best_model"
        elif (save_dir / "model_state_dict.pt").exists():
            save_name = "model"
        else:
            raise FileNotFoundError("Error: resume_from_dir expects a model\
                                        state dict named model.pt or best_model.pt.")
        # returns model, loads other modules in-place if provided
        self.model = load_training_state(save_dir=save_dir, save_name=save_name,
                                                model=self.model,
                                                optimizer=self.optimizer,
                                                regularizer=self.regularizer,
                                                scheduler=self.scheduler)

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
        save_training_state(save_dir=save_dir, 
                            save_name=save_name,
                            model=self.model,
                            optimizer=self.optimizer,
                            scheduler=self.scheduler,
                            regularizer=self.regularizer
                            )
        if self.verbose:
            print(f"Saved training state to {save_dir}")

       