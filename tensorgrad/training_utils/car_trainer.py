import random

import torch

from .ckpt_trainer import CheckpointTrainer

class CarTrainer(CheckpointTrainer):
    """
    A subclass of CheckpointTrainer that implements sample_max for geometry models
    """
    def __init__(
        self,
        *args,
        **kwargs
    ):
        """
        Inherit init from CheckpointTrainer
        """
        super().__init__(*args, **kwargs)

    def train(
        self,
        *args,
        sample_max: int=None,
        **kwargs
    ):
        """Trains the given model on the given datasets.

        Inherits all from CheckpointTrainer except for sample_max
        """
        self.sample_max = sample_max
        return super().train(*args, **kwargs)


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
        
        # assume 'y' has shape (n_out, out_channels)
        batch_size = 1
        n_out = sample["y"].shape[0]
        
        self.n_samples += batch_size

        # query the graph and max sample_max locations and accumulate grads if sample_max is set
        if self.sample_max is None:
            num_queries = 1
            sample_max = n_out
        else:
            sample_max = self.sample_max
            num_queries = n_out // self.sample_max + 1
        full_y = sample.pop('y')
        full_out_p = sample.pop('out_p')
        full_out_inds = list(range(n_out))
        #random.shuffle(full_out_inds)
        for i in range(num_queries):
            start_offset = int(self.sample_max * i)
            out_inds = full_out_inds[start_offset:start_offset+sample_max]
            sample['y'] = full_y[out_inds]
            sample['out_p'] = full_out_p[out_inds]
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
            
        # assume 'y' has shape (n_out, out_channels)
        batch_size = 1
        n_out = sample["y"].shape[0]
        
        self.n_samples += batch_size

        # query the graph and max sample_max locations and accumulate grads if sample_max is set
        if self.sample_max is None:
            num_queries = 1
            sample_max = n_out
        else:
            sample_max = self.sample_max
            num_queries = n_out // self.sample_max + 1
        full_y = sample.pop('y')
        full_out_p = sample.pop('out_p')
        full_out_inds = list(range(n_out))
        eval_step_losses = {loss_name: 0. for loss_name in eval_losses.keys()}
        for i in range(num_queries):
            start_offset = int(self.sample_max * i)
            out_inds = full_out_inds[start_offset:start_offset+sample_max]
            sample['y'] = full_y[out_inds]
            sample['out_p'] = full_out_p[out_inds]
            out = self.model(**sample)
            if self.data_processor is not None:
                out, sample = self.data_processor.postprocess(out, sample)
        
            for loss_name, loss in eval_losses.items():
                val_loss = loss(out, **sample)
                eval_step_losses[loss_name] += val_loss
        
        for loss_name in eval_losses.keys():
            eval_step_losses[loss_name] /= num_queries
        if return_output:
            return eval_step_losses, out
        else:
            return eval_step_losses, None
