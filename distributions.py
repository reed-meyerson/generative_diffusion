# generators for a variety of difficult probability distributions
import torch
from torch.utils.data import IterableDataset


class CircleSampler(IterableDataset):
    """
    sample uniformly from unit circle
    """

    def __init__(self):
        super().__init__()

    def __iter__(self):
        return self

    def __next__(self) -> torch.FloatTensor:
        theta = 2.0 * torch.pi * torch.rand(1)
        return torch.tensor([torch.cos(theta), torch.sin(theta)])


class TwoMoonsSampler(IterableDataset):
    """
    sample uniformly from the classic two-moons data set
    """

    def __init__(self):
        super().__init__()
        self.circle_sampler = CircleSampler()

    def __iter__(self):
        return self

    def __next__(self) -> torch.FloatTensor:
        offset = torch.Tensor([0.5, 0.25])
        output = next(self.circle_sampler)
        if output[1] >= 0.0:
            output -= offset
        else:
            output += offset
        return output


class SpiralSampler(IterableDataset):
    """
    Note: this samples from a spiral uniformly by angle, NOT by length
    """

    def __init__(self, num_wraps: int = 3):
        super().__init__()
        self.num_wraps = num_wraps

    def __iter__(self):
        return self

    def __next__(self) -> torch.FloatTensor:
        t = torch.rand(1)
        theta = self.num_wraps * 2.0 * torch.pi * t
        return t * torch.tensor([torch.cos(theta), torch.sin(theta)])
