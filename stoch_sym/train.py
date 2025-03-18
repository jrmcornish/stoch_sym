from typing import Callable

import tqdm

import torch
import torch.nn as nn

import wandb

from .checkpointing import save_checkpoint


def train(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    optimiser: torch.optim.Optimizer,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    test_metrics: dict[
        str, Callable[[nn.Module, torch.Tensor, torch.Tensor], torch.Tensor]
    ],
    start_epoch: int,
    num_epochs: int,
    epochs_per_test: int,
    device: torch.device,
):
    if start_epoch == 0:
        log(test(model, test_loader, test_metrics, device), epoch=start_epoch)
        start_epoch = 1

    progress_bar = get_train_progress_bar(start_epoch, epochs_per_test)

    for epoch in range(start_epoch, num_epochs + 1):
        model.train()

        tot_loss = 0
        tot_data_points = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimiser.zero_grad()
            y_hat = model(x)
            loss = loss_fn(y, y_hat)

            if not torch.isfinite(loss):
                raise FloatingPointError(f"Obtained loss {loss}. Stopping training.")

            loss.backward()
            optimiser.step()

            tot_loss += x.shape[0] * loss.item()
            tot_data_points += x.shape[0]

        progress_bar.update(1)

        if progress_bar.n == progress_bar.total:
            progress_bar.close()

            train_metric_values: dict[str, float] = {
                "ave-loss": tot_loss / tot_data_points
            }
            test_metric_values = test(model, test_loader, test_metrics, device)

            log(test_metric_values | train_metric_values, epoch=epoch)

            save_checkpoint(epoch=epoch, model=model, optimiser=optimiser)

            progress_bar = get_train_progress_bar(epoch + 1, epochs_per_test)


def get_train_progress_bar(next_epoch: int, epochs_per_test: int) -> tqdm.tqdm:
    if next_epoch == 1:
        # We test after the first epoch since the model usually changes a lot then
        epochs_before_test = 1
    else:
        epochs_before_test = epochs_per_test - ((next_epoch - 1) % epochs_per_test)

    return tqdm.tqdm(total=epochs_before_test, leave=False, desc="Train", unit="epoch")


def test(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    test_metrics: dict[
        str, Callable[[nn.Module, torch.Tensor, torch.Tensor], torch.Tensor]
    ],
    device: torch.device,
) -> dict[str, float]:
    model.eval()

    num_test_points = 0
    test_metric_values: dict[str, float] = {}
    for x, y in tqdm.tqdm(test_loader, leave=False, desc="Test", unit="batch"):
        x = x.to(device)
        y = y.to(device)

        batch_size = x.shape[0]
        num_test_points += batch_size

        for metric_name, metric_fn in test_metrics.items():
            test_metric_values.setdefault(metric_name, 0.0)

            with torch.no_grad():
                metric_val = metric_fn(model, x, y).item()

            # Normalise by batch size (NOTE: assumes each metric is an average)
            test_metric_values[metric_name] += batch_size * metric_val

    # Reaverage everything
    for metric_name in test_metric_values:
        test_metric_values[metric_name] /= num_test_points

    return test_metric_values


def log(test_metric_values: dict[str, float], epoch: int):
    wandb.log(test_metric_values, step=epoch)
    print({"epoch": epoch} | test_metric_values, flush=True)
