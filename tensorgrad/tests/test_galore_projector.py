import torch
from torch import nn
from torch.testing import assert_close

from ..projectors.galore_projector import GaLoreProjector

input_shape = [128,128]

def test_full_rank_project_and_project_back():
    in_grads = nn.Parameter(torch.randn(input_shape, dtype=torch.cfloat))

    projector = GaLoreProjector(rank=1.0)

    proj_grads = projector.project(in_grads, iter=0)
    assert proj_grads.shape == torch.Size(input_shape)

    out_grads = projector.project_back(proj_grads)

    assert_close(in_grads, out_grads)


def test_low_rank_project():
    device = 'cuda:0' if torch.backends.cuda.is_built() else 'cpu'

    float_rank = 0.25
    in_grads = torch.randn(input_shape)

    from neuralop.data.datasets.darcy import load_darcy_flow_small

    float_rank_projector = GaLoreProjector(rank=float_rank)

    float_proj_grads = float_rank_projector.project(in_grads, iter=0)

    print(f"{float_proj_grads.shape=}")