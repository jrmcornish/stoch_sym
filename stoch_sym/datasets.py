from typing import Callable

import torch
from torch.utils.data import IterableDataset


class InfiniteDataset(IterableDataset):
    def __init__(
        self,
        size: int,
        sampler: Callable[[], tuple[torch.Tensor, torch.Tensor]],
    ):
        self.size = size
        self.sampler = sampler

    def __iter__(self):
        x_buffer, y_buffer = self.sampler()

        assert len(x_buffer) == len(y_buffer) == len(self)

        buffer_index = 0
        for _ in range(len(self)):
            yield x_buffer[buffer_index], y_buffer[buffer_index]
            buffer_index += 1

    def __len__(self):
        return self.size


# TODO: Rename these to match CLI


def get_matrix_inv_data(size: int, matrix_dim: int):
    x = torch.randn(size, matrix_dim, matrix_dim)
    y = torch.inverse(x)
    assert y.isfinite().all()
    return x, y


def get_matrix_exp_data(size: int, matrix_dim: int):
    x = torch.randn(size, matrix_dim, matrix_dim)
    y = torch.matrix_exp(x)
    assert y.isfinite().all()
    return x, y


def get_linreg_data(size: int, dim: int):
    n = 25
    y = torch.randn(size, dim)
    z = torch.randn(size, dim, n)
    u = torch.randn(size, 1, n)

    x = torch.cat((z, y.unsqueeze(1) @ z + u), dim=1)
    return x, y

    # y_hat = torch.inverse(x @ x.transpose(1, 2)) @ x @ y
    # return torch.cat((x, y), dim=1), y_hat


def get_cov_data(size: int, dim: int):
    n = 25

    G = torch.randn(size, dim, dim)
    x = G @ torch.randn(size, dim, n)
    Sigma = G @ G.transpose(1, 2)

    return x, Sigma
