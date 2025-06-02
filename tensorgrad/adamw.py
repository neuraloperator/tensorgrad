# copy dependencies from transformers/optimization.py
import math
import warnings
from typing import Callable, Iterable, Tuple, Union, List

import torch
from torch import nn
from torch.optim import Optimizer

#from transformers.utils.versions import require_version
from torch.autograd.profiler import profile, record_function

from .projectors.projector_utils import get_projector
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
        enforce_full_complex_precision: bool=False,
        run_name=None,
    ):
        if not no_deprecation_warning:
            warnings.warn(
                "This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch"
                " implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this"
                " warning",
                FutureWarning,
            )
       # require_version("torch>=1.5.0")  # add_ with alpha
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
        self.enforce_full_complex_precision = enforce_full_complex_precision
        
        assert first_dim_rollup <= 3, "Error: cannot roll up more than 3 dimensions for first matrix dim"
        self.first_dim_rollup = first_dim_rollup
        self.support_complex = support_complex
        self.n_iter_max = n_iter_max
        self.run_name = run_name
        self.logged_sizes = False
        self.id_counter = 0
        self.mixed_precision = True
        
        if enforce_full_complex_precision:
            print("### Using AdamW with full complex precision for states ###")

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
                
                grad_is_complex = torch.is_complex(grad)
                input_shape = grad.shape
                
                state = self.state[p]
                
                # Convert grad to full precision if needed
                if self.enforce_full_complex_precision and grad.dtype == torch.complex32:
                    grad = grad.to(torch.cfloat)
                
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                if "step" not in state:
                    state["step"] = 0
                    state["id"] = self.id_counter
                self.id_counter += 1
                    
                if 'dim' not in group:
                    group['dim'] = 2
                # Low-rank Projection
                if "rank" in group:
                    if "projector" not in state:
                        state["projector"] = get_projector(group, matrix_only=self.matrix_only, 
                                                           support_complex=self.support_complex)
                        # Reset optimizer states if requested when creating new projector
                        if group.get("reset_sparse_optimizer_states", False):
                            state.pop("exp_avg", None)
                            state.pop("exp_avg_sq", None)
                            state["step"] = 0

                    if grad.ndim == 5: # if complex tensor is stored as 2 real tensors
                        grad = torch.view_as_complex(grad)

                    if self.matrix_only or grad.ndim <= 2:
                        proj_input = grad.view(grad.shape[0],-1)
                        if grad.ndim > 2:                            
                            input_shape = grad.shape
                            first_dim_reshape_size = math.prod(grad.shape[:self.first_dim_rollup])
                            proj_input = grad.view(first_dim_reshape_size, -1)
                        else:
                            proj_input = grad.view(grad.shape[0],-1)
                    else:
                        proj_input = grad

                    with record_function("#### GRAD FORWARD PROJ ####"):
                        grad = state["projector"].project(proj_input, state["step"])
                        if not self.logged_sizes:
                            ratio = torch.numel(grad) / torch.numel(proj_input)
                            print(f"{torch.numel(proj_input)=} | {proj_input.shape=}")
                            print(f"{torch.numel(grad)=} | {grad.shape=}")
                            print(f"{ratio=}")
                            rank = group['rank']
                            print(f"{rank=}")
                            self.logged_sizes = True
                        if self.run_name:
                            msg = f"Orig shape= {input_shape}\n"
                            msg += f"Proj shape= {grad.shape}"
                            with open(f"./memstats/{self.run_name}_grad_size", "w") as f:
                                f.write(msg)
                            f.close()
                
                # Convert grad to full precision if needed
                if self.enforce_full_complex_precision and grad.dtype == torch.complex32:
                    grad = grad.to(torch.cfloat)

                # State initialization
                if "exp_avg" not in state:
                    # Use full precision for states if enforce_full_complex_precision is True
                    state["exp_avg"] = torch.zeros_like(grad, dtype=grad.dtype)
                    state["exp_avg_sq"] = torch.zeros_like(grad, dtype=grad.dtype)
                
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

                beta1, beta2 = group["betas"]

                state["step"] += 1
                
                exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                
                if grad.dtype == torch.complex32 and not self.enforce_full_complex_precision:
                    # Use basic operations instead of addcmul_ for comple   x32
                    exp_avg_sq.mul_(beta2)
                    if grad_is_complex:
                        exp_avg_sq.add_((grad * grad.conj()) * (1.0 - beta2))
                    else:
                        exp_avg_sq.add_((grad * grad) * (1.0 - beta2))
                else:
                    # Original logic for other dtypes
                    if grad_is_complex:
                        exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1.0 - beta2)
                    else:
                        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                denom = exp_avg_sq.sqrt().add_(group["eps"])
                
                del grad

                step_size = group["lr"]
                if group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                # compute norm gradient
                norm_grad = exp_avg / denom
                

                # Low-rank Projection Back
                if "rank" in group:
                    with record_function("#### GRAD BACKWARD PROJ ####"):
                        # Validate projector is on the correct device
                        if hasattr(state["projector"], "proj_tensor") and state["projector"].proj_tensor is not None:
                            proj_device = state["projector"].proj_tensor[0].device
                            assert proj_device == norm_grad.device, f"Device mismatch: projector on {proj_device}, norm_grad on {norm_grad.device}"
                        
                        norm_grad = state["projector"].project_back(norm_grad)
                    
                    if norm_grad.shape != input_shape:
                        # put complex grads (d1 x d2 x .. dn) back into real (d1 x d2 x .. dn x 2)
                        if torch.is_complex(norm_grad) and not torch.is_complex(p):
                            print(f"viewing {norm_grad.shape} as real")
                            norm_grad = torch.view_as_real(norm_grad)
                        norm_grad = norm_grad.view(input_shape)
                    
                if self.enforce_full_complex_precision and p.dtype == torch.complex32:
                    norm_grad = norm_grad.to(torch.complex32)
                
                        

                torch.mul(norm_grad, -step_size, out=norm_grad)
                p.add_(norm_grad)
                del norm_grad

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                # Add weight decay at the end (fixed version)
                if group["weight_decay"] > 0.0:
                    p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))

        return loss

    def _move_state_to_device(self, state, device):
        """Helper method to move optimizer states to a specific device"""
        if "exp_avg" in state:
            state["exp_avg"] = state["exp_avg"].to(device)
            state["exp_avg_sq"] = state["exp_avg_sq"].to(device)
        return state