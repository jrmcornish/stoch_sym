from typing import Callable

import numpy as np

import torch


# TODO: Could remove this and just dispatch on name
def get_pure_function(name: str, config: dict) -> Callable:
    match name:
        case "gaussian":
            return lambda x: torch.randn(x.shape[0], config["dim"], device=x.device)

        case "orthogonal_haar":
            return lambda x: orthogonal_haar(x, dim=config["dim"])

        case "qr":
            return lambda x: qr(x)[0]

        case "transpose":
            return transpose

        case "matmul":
            return matmul

        case "gram":
            return gram

        case "cholesky":
            return cholesky

        case "flatten":
            return flatten

        case "make_square":
            return make_square

        case "append":
            return append

        case "cat":
            return cat

        case "split":
            return split

        case "pop":
            return pop

        case "insert":
            return insert

        case "unsqueeze":
            return unsqueeze

        case "squeeze":
            return squeeze

        case "mse":
            return torch.nn.functional.mse_loss

        case _:
            raise NotImplementedError(f"Component not implemented: {name}")


def qr(inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    assert inputs.ndim == 3 and inputs.shape[1] == inputs.shape[2]

    # NOTE: the current implementation of torch.linalg.qr can be numerically
    # unstable during backwards pass when output has (close to) linearly
    # dependent columns, at least until pivoting is implemented upstream
    # (see comment torch.linalg.qr docs, as well as
    # https://github.com/pytorch/pytorch/issues/42792). Hence we convert to
    # double before applying the QR (and then convert back)
    #
    # NOTE: In addition, for some reason, QR decomposition on GPU is very
    # slow in PyTorch. This is a known issue: see
    # https://github.com/pytorch/pytorch/issues/22573). We work around this
    # as follows, although this is *still* much slower than it could be if
    # it were properly taking advantage of the GPU...
    #
    Q, R = torch.linalg.qr(inputs.cpu().double())
    Q = Q.to(torch.get_default_dtype()).to(inputs.device)
    R = R.to(torch.get_default_dtype()).to(inputs.device)

    # This makes sure the diagonal is positive, so that the Q matrix is
    # unique (and coresponds to the output produced by Gram-Schmidt, which
    # is equivariant)
    diag_sgns = torch.diag_embed(torch.diagonal(R, dim1=-2, dim2=-1).sign())

    # *Shouldn't* do anything but just to be safe:
    diag_sgns = diag_sgns.detach()

    return Q @ diag_sgns, diag_sgns @ R


def gram(x: torch.Tensor) -> torch.Tensor:
    x = x.to(torch.float64)  # TODO: Test without
    return x @ x.transpose(1, 2)


def cholesky(x: torch.Tensor) -> torch.Tensor:
    assert len(x.shape) == 3
    assert x.shape[1] == x.shape[2]
    result = torch.linalg.cholesky(x)
    return result.to(torch.get_default_dtype())  # TODO: Test without


# TODO: Just make this take 1 argument
def flatten(*args: torch.Tensor) -> torch.Tensor:
    return torch.cat([x.flatten(start_dim=1) for x in args], dim=1)


def make_square(x: torch.Tensor) -> torch.Tensor:
    assert len(x.shape) == 2

    d = np.sqrt(x.shape[1])

    assert d == int(d), "Input must be squareable"

    return x.reshape(x.shape[0], int(d), int(d))


# TODO: Clarify usage vs insert
def append(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert len(x.shape) == 3
    assert len(y.shape) == 2
    return torch.cat((x, y.unsqueeze(2)), dim=2)


# TODO: Clarify usage vs insert and append
def cat(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert len(x.shape) == len(y.shape) == 2
    return torch.cat((x, y), dim=1)


def matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert len(x.shape) == 3, x.shape
    assert len(y.shape) == 3, y.shape
    assert x.shape[0] == y.shape[0]
    assert x.shape[2] == y.shape[1]
    return torch.bmm(x, y)


def transpose(x: torch.Tensor) -> torch.Tensor:
    assert len(x.shape) == 3
    return x.transpose(1, 2)


# TODO: Use a more descriptive name
def split(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    assert len(x.shape) >= 2
    assert x.shape[1] % 2 == 0
    return x[:, : x.shape[1] // 2], x[:, x.shape[1] // 2 :]


def pop(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    assert len(x.shape) == 3
    assert x.shape[1] > 1
    return x[:, :-1], x[:, -1]


def insert(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert len(x.shape) == 3
    assert len(y.shape) == 2
    assert x.shape[0] == y.shape[0]
    assert x.shape[2] == y.shape[1]
    return torch.cat((x, y.unsqueeze(1)), dim=1)


# TODO: Could make more generic
def unsqueeze(x: torch.Tensor) -> torch.Tensor:
    assert len(x.shape) == 2
    return x.unsqueeze(2)


# TODO: Could make more generic
def squeeze(x: torch.Tensor) -> torch.Tensor:
    assert len(x.shape) == 3
    return x.squeeze(2)


def orthogonal_haar(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Implements the method of https://arxiv.org/pdf/math-ph/0609050v2.pdf
    (see (5.12) of that paper in particular)
    """

    noise = torch.randn(x.shape[0], dim, dim, device=x.device)
    return qr(noise)[0]
