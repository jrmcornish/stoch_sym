from typing import Callable

import torch
from torch.nn import Module

from discopy.markov import Diagram

from .coarse_string_diagram import (
    get_coarse_string_diagram,
    get_equi_score_string_diagram,
)
from .fine_string_diagram import get_fine_string_diagram_functor
from .pytorch_model import get_module, get_semantic_functor

# TODO: Rename this file to something more descriptive


def get_equi_score(model: Module, config: dict) -> Module:
    equi_score_string_diagram = get_equi_score_string_diagram()
    F1 = get_fine_string_diagram_functor(config)
    F2 = get_semantic_functor(modules={"ave(k)": model}, config=config)
    return F2(F1(equi_score_string_diagram))


def get_fine_string_diagram(config: dict) -> Diagram:
    coarse_string_diagram = get_coarse_string_diagram(gamma_type=config["gamma"])
    return get_component_string_diagram(coarse_string_diagram, config)


def get_component_string_diagram(
    coarse_string_diagram: Diagram, config: dict
) -> Diagram:
    F = get_fine_string_diagram_functor(config)
    return F(coarse_string_diagram)


def get_model(config: dict) -> Module:
    return AveragedPredictor(
        get_module(get_fine_string_diagram(config=config), config=config),
        num_train_samples=config["num_train_samples"],
        num_test_samples=config["num_test_samples"],
    )


class AveragedPredictor(Module):
    def __init__(self, model: Module, num_train_samples: int, num_test_samples: int):
        super().__init__()
        self.model = model
        self.num_train_samples = num_train_samples
        self.num_test_samples = num_test_samples

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        num_samples = self.num_train_samples if self.training else self.num_test_samples
        y_samples = _repeated_sample(self.model, x, num_samples)
        return torch.mean(y_samples, dim=1)


def _repeated_sample(model: Module, x: torch.Tensor, num_samples: int) -> torch.Tensor:
    x_repeated = x.repeat_interleave(num_samples, dim=0)
    y_samples = _batch_apply(lambda x: model(x), x_repeated)
    return y_samples.view(x.shape[0], num_samples, *y_samples.shape[1:])


def _batch_apply(
    f: Callable[[*tuple[torch.Tensor]], torch.Tensor],
    *inputs: torch.Tensor,
    max_batch_size: int = 512,
) -> torch.Tensor:
    assert (
        len(set(x.shape[0] for x in inputs)) == 1
    ), "Inputs must share the same batch dimension"

    assert len(set(x.device for x in inputs)) == 1, "Inputs must share the same device"

    num_inputs = inputs[0].shape[0]
    device = inputs[0].device

    for i in range(0, num_inputs, max_batch_size):
        batch_inputs = [x[i : i + max_batch_size] for x in inputs]
        batch_outputs = f(*batch_inputs)

        assert (
            batch_outputs.shape[0] == batch_inputs[0].shape[0]
        ), "Batch function must not change batch dimension"

        if i == 0:
            # Avoid reallocating if possible
            if batch_outputs.shape[0] == num_inputs:
                outputs = batch_outputs
                break

            else:
                outputs = torch.empty(
                    num_inputs, *batch_outputs.shape[1:], device=device
                )

        outputs[i : i + max_batch_size] = batch_outputs

    return outputs
