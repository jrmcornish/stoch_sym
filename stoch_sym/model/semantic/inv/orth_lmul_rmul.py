from typing import Callable

import torch.nn as nn

from ..emlp_pytorch.emlp_pytorch.groups import O
from ..emlp_pytorch.emlp_pytorch.reps import T

from ..networks import ScalarsModule, get_mlp, get_emlp


def get_component(name: str, config: dict) -> Callable | None:
    match name:
        case "f_mlp":
            return get_mlp(
                num_input_channels=config["dim"] ** 2,
                num_output_channels=config["dim"] ** 2,
                hidden_channels=config["base_hidden_channels"],
            )

        case "gamma_mlp":
            return get_mlp(
                num_input_channels=config["dim"] ** 2 + config["dim"],
                num_output_channels=config["dim"] ** 2,
                hidden_channels=config["gamma_hidden_channels"],
            )

        case "f_emlp":
            return get_emlp(
                group=O(config["dim"]),
                input_rep=config["dim"] * T(1),
                output_rep=config["dim"] * T(1),
                hidden_channels=config["base_hidden_channels"],
            )

        case "gamma_emlp":
            return get_emlp(
                group=O(config["dim"]),
                input_rep=(config["dim"] + 1) * T(1),
                output_rep=config["dim"] * T(1),
                hidden_channels=config["gamma_hidden_channels"],
            )

        case "gamma_scalars_mlp":
            d = config["dim"]

            return ScalarsModule(
                module=nn.Sequential(
                    nn.Flatten(start_dim=1),
                    get_mlp(
                        num_input_channels=(d + 1) ** 2,
                        num_output_channels=(d + 1) * d,
                        hidden_channels=config["gamma_hidden_channels"],
                        activation=nn.Tanh(),
                        final_activation=False,  # TODO: Generalise
                    ),
                    nn.Unflatten(1, (d + 1, d)),
                )
            )
