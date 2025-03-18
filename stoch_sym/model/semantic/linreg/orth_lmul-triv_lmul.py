from typing import Callable

from ..emlp_pytorch.emlp_pytorch.groups import O
from ..emlp_pytorch.emlp_pytorch.reps import T

from ..networks import get_mlp, get_emlp


def get_component(name: str, config: dict) -> Callable | None:
    dim = config["dim"]
    n = 25
    data_size = n * (dim + 1)  # n data points (X, y), with X in R^dim and y in R

    match name:
        case "f_mlp":
            return get_mlp(
                num_input_channels=data_size,
                num_output_channels=dim,
                hidden_channels=config["base_hidden_channels"],
            )

        case "gamma_mlp":
            return get_mlp(
                num_input_channels=data_size + dim,
                num_output_channels=dim**2,
                hidden_channels=config["gamma_hidden_channels"],
            )

        case "f_emlp":
            return get_emlp(
                group=O(config["dim"]),
                input_rep=n * (T(1) + T(0)),
                output_rep=T(1),
                hidden_channels=config["base_hidden_channels"],
            )

        case "gamma_emlp":
            return get_emlp(
                group=O(config["dim"]),
                input_rep=n * (T(1) + T(0)) + T(1),
                output_rep=dim * T(1),
                hidden_channels=config["gamma_hidden_channels"],
            )
