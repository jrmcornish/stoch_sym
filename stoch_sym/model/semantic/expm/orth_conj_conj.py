from typing import Callable

from ..emlp_pytorch.emlp_pytorch.groups import O
from ..emlp_pytorch.emlp_pytorch.reps import T

from ..networks import get_mlp, get_emlp


def get_component(name: str, config: dict) -> Callable | None:
    match name:
        case "f_mlp":
            return get_mlp(
                num_input_channels=config["dim"] ** 2,
                num_output_channels=config["dim"] ** 2,
                hidden_channels=config["base_hidden_channels"],
            )

        case "f_emlp":
            return get_emlp(
                group=O(config["dim"]),
                input_rep=T(2),
                output_rep=T(2),
                hidden_channels=config["base_hidden_channels"],
            )

        case "gamma_mlp":
            return get_mlp(
                num_input_channels=config["dim"] ** 2 + config["dim"],
                num_output_channels=config["dim"] ** 2,
                hidden_channels=config["gamma_hidden_channels"],
            )

        case "gamma_emlp":
            return get_emlp(
                group=O(config["dim"]),
                # Another option is input_rep=T(2) + config["dim"] * T(0)
                # But this seems to give a constant function that doesn't learn anything
                input_rep=T(2) + T(1),
                output_rep=config["dim"] * T(1),
                hidden_channels=config["gamma_hidden_channels"],
            )
