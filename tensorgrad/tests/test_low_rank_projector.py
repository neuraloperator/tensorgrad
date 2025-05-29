import torch
from torch import nn
from torch.testing import assert_close

from ..tensor_lowrank_projector import TensorGradLowRankProjector

input_shape = [32,32,16,16]

def test_full_rank_project_and_project_back():
    proj_shape = input_shape
    in_grads = nn.Parameter(torch.randn(input_shape, dtype=torch.cfloat))

    projector = TensorGradLowRankProjector(rank=proj_shape)

    proj_grads = projector.project(in_grads, iter=0)
    assert proj_grads.shape == torch.Size(proj_shape)

    out_grads = projector.project_back(proj_grads)

    assert_close(in_grads, out_grads)

def test_warm_restart():
    proj_shape = input_shape
    in_grads = nn.Parameter(torch.randn(input_shape, dtype=torch.cfloat))

    projector = TensorGradLowRankProjector(rank=proj_shape, update_proj_gap=1, warm_restart=True)

    proj_grads = projector.project(in_grads, iter=0)
    assert proj_grads.shape == torch.Size(proj_shape)

    out_grads = projector.project_back(proj_grads)
    assert_close(in_grads, out_grads)

    proj_again = projector.project(in_grads, iter=1)
    assert proj_again.shape == torch.Size(proj_shape)

def test_low_rank_project():
    device = 'cuda:0' if torch.backends.cuda.is_built() else 'cpu'

    proj_shape = [int(x // 4) for x in input_shape]
    float_rank = 0.25
    in_grads = torch.randn(input_shape)

    from neuralop.data.datasets.darcy import load_darcy_flow_small

    int_rank_projector = TensorGradLowRankProjector(rank=proj_shape)
    float_rank_projector = TensorGradLowRankProjector(rank=float_rank)

    int_proj_grads = int_rank_projector.project(in_grads, iter=0)
    float_proj_grads = float_rank_projector.project(in_grads, iter=0)
    assert int_proj_grads.shape == torch.Size(proj_shape)

    print(f"{float_proj_grads.shape=}")
    assert float_proj_grads.shape == torch.Size(proj_shape)

'''
def test_high_rank_project_and_project_back():
    device = 'cuda:0' if torch.backends.cuda.is_built() else 'cpu'

    proj_shape = [int(x-1) for x in input_shape]

    from neuralop.data.datasets.darcy import load_darcy_flow_small
    from neuralop.losses import LpLoss
    from neuralop import Trainer
    lploss = LpLoss(d=2, p=2, L=1.)
    train_loader, _, data_processor = load_darcy_flow_small(n_train=50,
                                                            n_tests=[50],
                                                            batch_size=16, 
                                                            test_batch_sizes=[16], 
                                                            encode_input=True, 
                                                            test_resolutions=[16])
    
    trainer = Trainer()
    projector = TensorGradLowRankProjector(rank=proj_shape)

    proj_grads = projector.project(in_grads, iter=0)
    assert proj_grads.shape == torch.Size(proj_shape)

    out_grads = projector.project_back(proj_grads)

    error = out_grads - in_grads

    rel_rec_error = tl.norm(error) / tl.norm(in_grads)
    print(f"{rel_rec_error=}")
'''