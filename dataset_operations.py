import torch
from torch.utils.data import Dataset, IterableDataset


class MapIDS(IterableDataset):
    def __init__(self, func, ds: IterableDataset):
        super().__init__()
        self.ds = ds
        self.func = func

    def __iter__(self):
        self.ds_iter = iter(self.ds)
        return self

    def __next__(self):
        return self.func(next(self.ds_iter))
