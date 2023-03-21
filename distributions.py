# generators for a variety of difficult probability distributions
import torch


def circle_sampler(batch_size: int) -> torch.Tensor:
    """
    Uniform random sample from unit circle
    """
    theta = 2.0 * torch.pi * torch.rand(batch_size).view(-1, 1)
    return torch.hstack((torch.cos(theta), torch.sin(theta)))


def two_moons_sampler(batch_size: int) -> torch.Tensor:
    """
    classic 'two moons' distribution
    """
    offset = torch.Tensor([0.5, 0.25])
    output = circle_sampler(batch_size)
    top_mask = output[:, 1] >= 0.0
    bottom_mask = top_mask.logical_not()
    output[top_mask] -= offset
    output[bottom_mask] += offset
    return output

def spiral_sampler(batch_size: int, num_wraps: int = 3) -> torch.Tensor:
    """
    Note: this samples from a spiral uniformly by angle, NOT by length
    """
    t = torch.rand(batch_size).view(-1, 1)
    theta = num_wraps * 2.0 * torch.pi * t
    return t * torch.hstack((torch.cos(theta), torch.sin(theta)))
