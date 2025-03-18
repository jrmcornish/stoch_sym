import os
import random
import tempfile

import numpy as np

import torch
import torch.nn as nn

import wandb
import wandb.apis


CHECKPOINT_FILENAME = "checkpoint.pt"


def save_checkpoint(epoch: int, model: nn.Module, optimiser: torch.optim.Optimizer):
    if wandb.run is not None and wandb.run.settings.mode != "disabled":
        (a, b, c, d, e) = np.random.get_state()
        np_random_state = (a, torch.tensor(b), c, d, e)

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "opt_state_dict": optimiser.state_dict(),
                "python_rng_state": random.getstate(),
                "torch_rng_state": torch.get_rng_state(),  # NOTE: Unclear if this saves CUDA RNG state
                "numpy_rng_state": np_random_state,
            },
            os.path.join(wandb.run.dir, CHECKPOINT_FILENAME),
        )

        wandb.save(
            os.path.join(wandb.run.dir, CHECKPOINT_FILENAME),
            base_path=wandb.run.dir,
            policy="now",
        )


def load_checkpoint(
    run: wandb.apis.public.Run, model: nn.Module, optimiser: torch.optim.Optimizer
) -> int:
    with tempfile.TemporaryDirectory() as checkpoint_dir:
        checkpoint_file = run.file(CHECKPOINT_FILENAME).download(  # type: ignore
            root=checkpoint_dir
        )

        checkpoint = torch.load(checkpoint_file.name, map_location=torch.device("cpu"))

    model.load_state_dict(checkpoint["model_state_dict"])
    optimiser.load_state_dict(checkpoint["opt_state_dict"])

    random.setstate(checkpoint["python_rng_state"])
    torch.set_rng_state(checkpoint["torch_rng_state"])

    (a, b, c, d, e) = checkpoint["numpy_rng_state"]
    np_random_state = (a, b.numpy(), c, d, e)
    np.random.set_state(np_random_state)

    return checkpoint["epoch"]
