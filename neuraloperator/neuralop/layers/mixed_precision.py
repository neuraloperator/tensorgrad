"""
Mixed precision utilities for neural operators.

This module provides utilities for working with mixed precision and stochastic
rounding in neural operators, particularly for SpectralConv layers.
"""

import torch
import torch.nn as nn
from typing import Tuple, Union, Optional



def fp16_sr_bit(x: torch.Tensor) -> torch.Tensor:
    """
    Convert a tensor to FP16 using stochastic rounding.
    Handles both real and complex tensors appropriately.
    
    Parameters
    ----------
    x : torch.Tensor
        Input tensor to round (can be real or complex)
        
    Returns
    -------
    torch.Tensor
        Stochastically rounded tensor in half precision 
        (torch.half for real tensors, torch.complex32/chalf for complex tensors)
    """
    # Handle complex tensors properly
    if torch.is_complex(x):
        # Process real and imaginary parts separately
        real_part = _fp16_sr_component(x.real)
        imag_part = _fp16_sr_component(x.imag)
        # Return as complex32 (chalf)
        return torch.complex(real_part, imag_part).to(torch.complex32)
    else:
        # For real tensors, directly apply stochastic rounding
        return _fp16_sr_component(x).to(torch.half)


def _fp16_sr_component(x: torch.Tensor) -> torch.Tensor:
    """
    Helper function to apply stochastic rounding to a real tensor component.
    
    Parameters
    ----------
    x : torch.Tensor
        Real input tensor to round
        
    Returns
    -------
    torch.Tensor
        Stochastically rounded real tensor
    """
    # Get the deterministically rounded value (to nearest fp16 value)
    x_rounded = x.half().float()
    
    # Calculate the spacing between fp16 values at this magnitude
    abs_x = torch.abs(x_rounded)
    spacing = torch.where(abs_x == 0,
                        torch.tensor(2**-24, device=x.device),  # min positive fp16
                        abs_x * 2**-10)  # normal spacing
    
    # Calculate the remainder
    remainder = x - x_rounded
    
    # Calculate probability of rounding up
    prob = (remainder / spacing).clamp(0, 1)
    
    # Generate random values - using rand_like to ensure matching shapes
    rand = torch.rand_like(prob)
    
    # Apply stochastic rounding through direct indexing
    result = x_rounded.clone()
    round_up_mask = rand < prob
    result[round_up_mask] += spacing[round_up_mask]
    
    return result


def round_complex(x: torch.Tensor) -> torch.Tensor:
    """
    Apply stochastic rounding to complex tensor.
    
    Parameters
    ----------
    x : torch.Tensor
        Complex input tensor
        
    Returns
    -------
    torch.Tensor
        Complex tensor with both real and imaginary parts stochastically rounded
    """
    if not torch.is_complex(x):
        return fp16_sr_bit(x)
    
    # Handle real and imaginary parts separately
    x_real = fp16_sr_bit(x.real)
    x_imag = fp16_sr_bit(x.imag)
    
    # Combine them back into a complex tensor
    return torch.complex(x_real, x_imag)
