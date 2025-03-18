from typing import Callable

from ..networks import get_mlp


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
                num_output_channels=2 * config["dim"] ** 2,
                hidden_channels=config["gamma_hidden_channels"],
            )
