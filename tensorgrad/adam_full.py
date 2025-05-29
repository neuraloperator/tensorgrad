# copy dependencies from transformers/optimization.py
import math
import warnings
from typing import Callable, Iterable, Tuple, Union, List

import torch
from torch import nn
from torch.optim import Optimizer

#from transformers.utils.versions import require_version
from torch.autograd.profiler import profile, record_function


from .training_utils import get_scheduler


class AdamW(Optimizer):
    """
    Implements Adam algorithm with weight decay fix as introduced in [Decoupled Weight Decay
    Regularization](https://arxiv.org/abs/1711.05101).

    Parameters:
        params (`Iterable[nn.parameter.Parameter]`):
            Iterable of parameters to optimize or dictionaries defining parameter groups.
        lr (`float`, *optional*, defaults to 0.001):
            The learning rate to use.
        betas (`Tuple[float,float]`, *optional*, defaults to `(0.9, 0.999)`):
            Adam's betas parameters (b1, b2).
        eps (`float`, *optional*, defaults to 1e-06):
            Adam's epsilon for numerical stability.
        weight_decay (`float`, *optional*, defaults to 0.0):
            Decoupled weight decay to apply.
        correct_bias (`bool`, *optional*, defaults to `True`):
            Whether or not to correct bias in Adam (for instance, in Bert TF repository they use `False`).
        matrix_only : bool, default True
            whether to use TensorGrad or flatten tensors to matrices and use classic Galore
        no_deprecation_warning (`bool`, *optional*, defaults to `False`):
            A flag used to disable the deprecation warning (set to `True` to disable the warning).
        use_tf32: bool, default False
            whether to use TF32 for matrix operations
    """

    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
        matrix_only: bool = True,
        no_deprecation_warning: bool = False,
        first_dim_rollup = 0,
        support_complex: bool=False,
        n_iter_max: int=10,
        run_name=None,
        use_tf32: bool = False,
    ):
        if not no_deprecation_warning:
            warnings.warn(
                "This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch"
                " implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this"
                " warning",
                FutureWarning,
            )
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0)")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0)")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay, "correct_bias": correct_bias}
        super().__init__(params, defaults)
        self.matrix_only = matrix_only
        self.use_tf32 = use_tf32
        
        assert first_dim_rollup <= 3, "Error: cannot roll up more than 3 dimensions for first matrix dim"
        self.first_dim_rollup = first_dim_rollup
        self.support_complex = support_complex
        self.n_iter_max = n_iter_max
        self.run_name = run_name
        self.logged_sizes = False
        self.print_agreement_steps = 5000
        self.verbose = False
        self.id_count = 0
        self.enforce_full_complex_precision = True
        
        # Enable TF32 for matrix operations if requested
        if use_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("### Using C-AdamW with TF32 precision ###")
        else:
            print("### Using C-AdamW with full precision states ###")

    @torch.no_grad()
    def step(self, closure: Callable = None):
        """
        Performs a single optimization step.

        Arguments:
            closure (`Callable`, *optional*): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                    
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                state = self.state[p]
                
                if "step" not in state:
                    state["step"] = 0
                    state["input_shape"] = grad.shape

                if "id" not in state:
                    self.id_count += 1
                    state["id"] = self.id_count  # Assign unique ID to this parameter's state

                if 'dim' not in group:
                    group['dim'] = 2
                     
                                # Convert grad to appropriate precision for state updates
                if self.use_tf32:
                    # For TF32, convert to float32 and let CUDA handle the TF32 conversion
                    grad_full = grad.to(torch.float32)
                elif self.enforce_full_complex_precision and grad.dtype == torch.complex32:
                    grad_full = grad.to(torch.cfloat)
                else:
                    grad_full = grad
                    
                # State initialization - use appropriate precision based on settings
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(grad, dtype=grad_full.dtype)
                    state["exp_avg_sq"] = torch.zeros_like(grad, dtype=grad_full.dtype)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad_full, alpha=(1.0 - beta1))
                
                # Handle complex case
                if torch.is_complex(grad_full):
                    exp_avg_sq.mul_(beta2).addcmul_(grad_full, grad_full.conj(), value=1.0 - beta2)
                else:
                    exp_avg_sq.mul_(beta2).addcmul_(grad_full, grad_full, value=1.0 - beta2)

                denom = exp_avg_sq.sqrt().add_(group["eps"])
                norm_grad = exp_avg / denom

                step_size = group["lr"]
                if group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                # Convert norm_grad back to parameter's dtype for update
                norm_grad = norm_grad.to(p.dtype)
                p.add_(norm_grad, alpha=-step_size)

                # Add weight decay at the end (fixed version)
                if group["weight_decay"] > 0.0:
                    p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))

        return loss


    @classmethod
    def per_layer_weight_opt(cls, 
                             model: torch.nn.Module,
                             id_tensorgrad_params: list,
                             rank: Union[int, float, List[int]], 
                             update_proj_gap: int,
                             tensorgrad_scale: float,
                             proj_type: str,
                             lr: float,
                             weight_decay: float,
                             matrix_only: bool,
                             first_dim_rollup: int,
                             scheduler_name: str,
                             gamma: float,
                             patience: int,
                             T_max: int,
                             step_size: int):
        '''
        Optimize weight gradients per parameter and discard gradients immediately after stepping
        Returns a dict of optimizers per parameter
        '''
        optimizer_dict = {}
        scheduler_dict = {}
        tensorgrad_params = []
        tensorgrad_params.extend(list(model.fno_blocks.convs.parameters()))
        print(tensorgrad_params[0].shape, tensorgrad_params[1].shape, tensorgrad_params[2].shape, tensorgrad_params[3].shape)
        # drop the first projection layer
        tensorgrad_params.pop(0)
        id_tensorgrad_params = [id(p) for p in tensorgrad_params]
        # make parameters without "rank" to another group
        regular_params = [p for p in model.parameters() if id(p) not in id_tensorgrad_params]

        for p in regular_params:
            if p.requires_grad:
                optimizer_dict[p] = AdamW([p], lr=lr, 
                                          weight_decay=weight_decay)
                scheduler_dict[p] = get_scheduler(
                    scheduler_name=scheduler_name,
                    optimizer=optimizer_dict[p],
                    gamma=gamma,
                    patience=patience,
                    T_max=T_max,
                    step_size=step_size)
                
        for p in tensorgrad_params:
            if p.requires_grad:       
                optimizer_dict[p] = AdamW([{'params': [p], 
                                            'rank': rank, 
                                            'dim': p.ndim,
                                            'update_proj_gap': update_proj_gap * 2, 
                                            'scale': tensorgrad_scale, 
                                            'proj_type': proj_type}], 
                                            lr=lr, 
                                            weight_decay=weight_decay,
                                            matrix_only=matrix_only,
                                            first_dim_rollup=first_dim_rollup)
                scheduler_dict[p] = get_scheduler(
                    scheduler_name=scheduler_name,
                    optimizer=optimizer_dict[p],
                    gamma=gamma,
                    patience=patience,
                    T_max=T_max,
                    step_size=step_size)
                    
        # define a hook function to update the parameter p during the backward pass
        def optimizer_hook(p):
            if p.grad is None: 
                return
            optimizer_dict[p].step()
            optimizer_dict[p].zero_grad()
            scheduler_dict[p].step()

        # Register the hook onto every parameter
        for p in regular_params:
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(optimizer_hook)
        for p in tensorgrad_params:
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(optimizer_hook)

        return optimizer_dict, scheduler_dict