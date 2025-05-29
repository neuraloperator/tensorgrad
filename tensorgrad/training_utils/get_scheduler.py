import torch
import math

def get_scheduler(scheduler_name: str,
                  optimizer: torch.optim.Optimizer,
                  gamma: float,
                  patience: int,
                  T_max: int,
                  step_size: int,):
    '''
    Returns LR scheduler of choice from available options
    '''
    if scheduler_name == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=gamma,
            patience=patience,
            mode="min",
        )
    elif scheduler_name == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=T_max
        )
    elif scheduler_name == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma
        )
    elif scheduler_name == "constant":
        def constant_scheduler(step):
            return 1.0
        return constant_scheduler
    elif scheduler_name == "step":
        def step_scheduler(step):
            return gamma ** (step // step_size)
        return step_scheduler
    elif scheduler_name == "exponential":
        def exp_scheduler(step):
            return gamma ** step
        return exp_scheduler
    elif scheduler_name == "cosine":
        def cosine_scheduler(step):
            return 0.5 * (1 + math.cos(math.pi * step / T_max))
        return cosine_scheduler
    else:
        raise ValueError(f"Got scheduler={scheduler_name}")

    return scheduler