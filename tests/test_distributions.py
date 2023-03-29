import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from distributions import CircleSampler, SpiralSampler, TwoMoonsSampler

torch.random.manual_seed(42)


def test_circle():
    sampler = CircleSampler()
    x = next(iter(sampler))
    assert x.shape == torch.Size([2])
    assert x.dtype == torch.float32

    dl = DataLoader(sampler, batch_size=4)
    batch = next(iter(dl))
    assert batch.shape == torch.Size([4, 2])
    assert batch.dtype == torch.float32


def test_two_moons():
    sampler = TwoMoonsSampler()
    x = next(iter(sampler))
    assert x.shape == torch.Size([2])
    assert x.dtype == torch.float32

    dl = DataLoader(sampler, batch_size=4)
    batch = next(iter(dl))
    assert batch.shape == torch.Size([4, 2])
    assert batch.dtype == torch.float32


def test_spiral():
    sampler = SpiralSampler()
    x = next(iter(sampler))
    assert x.shape == torch.Size([2])
    assert x.dtype == torch.float32

    dl = DataLoader(sampler, batch_size=4)
    batch = next(iter(dl))
    assert batch.shape == torch.Size([4, 2])
    assert batch.dtype == torch.float32
