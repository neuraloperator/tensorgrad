import torch

def count_total_params(param_groups: dict):
    total_params = 0
    for group in param_groups:
        total_params += sum([p.numel() for p in group['params']])
    return total_params
        