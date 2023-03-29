import torch

from model import ResNet


def test_resnet():
    m = ResNet([3, 5, 4, 2])

    # test single input
    x = torch.rand(3)
    y = m(x)
    assert y.shape == torch.Size([2])

    # test batch input
    x = torch.rand((5, 3))
    y = m(x)
    assert y.shape == torch.Size([5, 2])
