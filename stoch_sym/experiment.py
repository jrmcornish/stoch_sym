from dataclasses import dataclass
from typing import Callable
import random

import numpy as np

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from .datasets import (
    InfiniteDataset,
    get_cov_data,
    get_matrix_exp_data,
    get_matrix_inv_data,
    get_linreg_data,
)

from .model.builder import get_equi_score, get_model


@dataclass
class Experiment:
    model: nn.Module
    device: torch.device
    train_loader: DataLoader
    test_loader: DataLoader
    optimiser: torch.optim.Optimizer
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    test_metrics: dict[
        str, Callable[[nn.Module, torch.Tensor, torch.Tensor], torch.Tensor]
    ]


def setup_experiment(config: dict):
    seed_everything(seed=config["seed"])

    model = get_model(config=config)

    device = get_device(device_name=config["device"])
    model.to(device)

    optimiser = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    train_loader = get_data_loader(size=config["num_train_examples"], config=config)
    test_loader = get_data_loader(size=config["num_test_examples"], config=config)

    loss_fn = get_loss_fn(config)
    test_metrics = get_test_metrics(config)

    return Experiment(
        model=model,
        device=device,
        train_loader=train_loader,
        test_loader=test_loader,
        optimiser=optimiser,
        loss_fn=loss_fn,
        test_metrics=test_metrics,
    )


def seed_everything(seed: int):
    random.seed(seed)
    torch.manual_seed(seed + 1)
    np.random.seed(seed + 2)


def get_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
    else:
        return torch.device(device_name)


def get_data_loader(size: int, config: dict) -> DataLoader:
    return torch.utils.data.DataLoader(
        InfiniteDataset(size=size, sampler=lambda: get_data(size=size, config=config)),
        batch_size=config["batch_size"],
        drop_last=False,
        pin_memory=True,
    )


# TODO: Tidy these up
def get_data(size: int, config: dict) -> tuple[torch.Tensor, torch.Tensor]:
    match config["dataset"]:
        case "inv":
            return get_matrix_inv_data(size=size, matrix_dim=config["dim"])

        case "expm":
            return get_matrix_exp_data(size=size, matrix_dim=config["dim"])

        case "linreg":
            return get_linreg_data(size=size, dim=config["dim"])

        case "cov":
            return get_cov_data(size=size, dim=config["dim"])

        case _:
            raise ValueError(f"Invalid dataset name: {config['dataset']}")


def get_loss_fn(config: dict) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    match config["loss"]:
        case "mse":
            return nn.functional.mse_loss

        case "relative-sse":
            return lambda y, y_hat: relative_sse(y=y, y_hat=y_hat)

        case _:
            raise ValueError(f"Invalid loss: {config['loss']}")


def relative_sse(y: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
    assert y.shape == y_hat.shape

    numer = (y_hat - y).pow(2).flatten(start_dim=1).sum(dim=1)
    denom = y.pow(2).flatten(start_dim=1).sum(dim=1)

    return (numer / denom).mean()


def get_test_metrics(
    config: dict,
) -> dict[str, Callable[[nn.Module, torch.Tensor, torch.Tensor], torch.Tensor]]:
    test_loss = lambda model, x, y: get_loss_fn(config)(y, model(x))
    det_equi_score = lambda model, x, _: get_equi_score(model=model, config=config)(x)

    return {
        f"test-{config['loss']}": test_loss,
        "det-equi-score": det_equi_score,
    }
