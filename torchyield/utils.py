import torch

def monkeypatch_tensor_repr():
    torch.Tensor.__repr__ = lambda x: f'torch.Tensor(shape={tuple(x.shape)})'

