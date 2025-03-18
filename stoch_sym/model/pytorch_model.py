from typing import Callable, Mapping
from importlib import import_module
import warnings

import torch
import torch.nn as nn

from discopy.cat import Category
from discopy.markov import Diagram, Functor, Box
import discopy.python

from .semantic.functions import get_pure_function


# TODO: Rename to get_model
def get_module(
    fine_string_diagram: Diagram,
    config: dict,
):
    modules = get_modules(fine_string_diagram, config)
    F = get_semantic_functor(modules, config)
    return Wrapper(F(fine_string_diagram), modules)


def get_semantic_functor(modules: Mapping[str, Callable], config: dict):
    ob = lambda ob: ob

    ar = lambda box: (
        modules[box.name]
        if box.name in modules
        else get_pure_function(box.name, config)
    )

    return Functor(
        ob=ob,
        ar=ar,
        cod=Category(discopy.python.Ty, discopy.python.Function),
    )


def get_modules(fine_string_diagram: Diagram, config: dict) -> nn.ModuleDict:
    module_name = f".semantic.{config['dataset']}.{config['group']}_{config['input_action']}_{config['output_action']}"
    get_component = import_module(module_name, package=__package__).get_component

    modules = nn.ModuleDict()

    for layer in fine_string_diagram:
        for item in layer:
            if type(item) is Box and item.name not in modules:
                # TODO: Document/warn about weight sharing
                component = get_component(item.name, config)

                if component is None:
                    continue

                modules[item.name] = component

    return modules


class Wrapper(nn.Module):
    def __init__(self, f: Callable, modules: nn.ModuleDict):
        super().__init__()
        self.f = f
        self.modules = modules

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.f(*args, **kwargs)
