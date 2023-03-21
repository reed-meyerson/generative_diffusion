import torch
from matplotlib import pyplot as plt

import distributions

torch.random.manual_seed(42)


def test_circle():
    x = distributions.circle_sampler(1)
    assert x.shape == (1, 2)

    x = distributions.circle_sampler(50)
    assert x.shape == (50, 2)
    assert x.dtype == torch.float32


def test_two_moons():
    x = distributions.two_moons_sampler(1)
    assert x.shape == (1, 2)

    x = distributions.two_moons_sampler(50)
    assert x.shape == (50, 2)
    assert x.dtype == torch.float32


def test_spiral():
    x = distributions.spiral_sampler(1)
    assert x.shape == (1, 2)

    x = distributions.spiral_sampler(50)
    assert x.shape == (50, 2)
    assert x.dtype == torch.float32
