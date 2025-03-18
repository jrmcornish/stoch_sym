import importlib
from typing import Callable

from discopy.monoidal import Ob
from discopy.markov import Box, Diagram, Functor


def get_fine_string_diagram_functor(config: dict) -> Functor:
    ob_map = get_ob_mapper(config)
    box_map = get_box_mapper(config)
    return Functor(ob=ob_map, ar=box_map)


def get_ob_mapper(config: dict) -> Callable[[Ob], Ob]:
    try:
        return get_mapper_from_module(config, suffix="_ob")

    except NotImplementedError:
        # For objects we allow things to default to the identity
        return lambda x: x


def get_box_mapper(config: dict) -> Callable[[Box], Diagram]:
    return get_mapper_from_module(config)


def get_mapper_from_module(config: dict, suffix=""):
    module_name = f".fine.{config['dataset']}.{config['group']}_{config['input_action']}_{config['output_action']}"

    try:
        module = importlib.import_module(module_name, package=__package__)
    except ModuleNotFoundError:
        raise NotImplementedError(
            f"Could not find module `{module_name}'. Does configuration exist?"
        )

    map_name = f"{config['backbone']}_{config['gamma']}{suffix}"
    map_name = map_name.replace("-", "")

    try:
        unwrapped_result = getattr(module, map_name)
    except AttributeError:
        raise NotImplementedError(
            f"Could not find function `{map_name}' in module `{module_name}'. Does configuration exist?"
        )

    def result(x):
        output = unwrapped_result(x)
        return output if output is not None else x

    return result
