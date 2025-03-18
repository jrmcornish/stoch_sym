import numpy as np

import torch
import torch.nn as nn

from .emlp_pytorch.emlp_pytorch.nn import EMLPBlock, Linear, uniform_rep

from .emlp_pytorch.emlp_pytorch.groups import Group as EMLPGroup
from .emlp_pytorch.emlp_pytorch.reps import Rep as EMLPRep


# TODO: Make into a module
def get_mlp(
    num_input_channels: int,
    num_output_channels: int,
    hidden_channels: list[int],
    activation: nn.Module = nn.Tanh(),
    final_activation: bool = False,
):
    prev_num_channels = num_input_channels
    layers = []

    for num_hidden_channels in hidden_channels:
        layers.append(nn.Linear(prev_num_channels, num_hidden_channels))
        layers.append(activation)
        prev_num_channels = num_hidden_channels

    layers.append(nn.Linear(prev_num_channels, num_output_channels))
    if final_activation:
        layers.append(activation)

    return nn.Sequential(*layers)


# TODO: Make into a module
def get_emlp(
    group: EMLPGroup,
    input_rep: EMLPRep,
    output_rep: EMLPRep,
    hidden_channels: list[int],
) -> nn.Sequential:
    hidden_reps = [uniform_rep(n, group) for n in hidden_channels]

    last_rep = input_rep(group)
    blocks = []
    for hidden_rep in hidden_reps:
        blocks.append(EMLPBlock(last_rep, hidden_rep))
        last_rep = hidden_rep

    blocks.append(Linear(last_rep, output_rep(group)))

    return nn.Sequential(*blocks)


class ScalarsModule(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert len(x.shape) == 3
        gram = x.transpose(1, 2) @ x
        scalars = self.module(gram)
        return x @ scalars


class DeepSets(nn.Module):
    def __init__(self, embedding_network: nn.Module, output_network: nn.Module):
        super().__init__()
        self.embedding_network = embedding_network
        self.output_network = output_network

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        embedding = self.embedding_network(input)
        return self.output_network(torch.sum(embedding, dim=1))
