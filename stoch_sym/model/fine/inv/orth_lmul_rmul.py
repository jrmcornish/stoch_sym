from discopy.markov import Box, Diagram

from ..shared import (
    orth_haar,
    sequential,
    orth_right_matmul_by_transpose,
    noise_outsourced,
)


def emlp_none(box: Box) -> Diagram | None:
    match box.name:
        case "f":
            return sequential(
                ["transpose", "flatten", "f_emlp", "make_square"], box.dom, box.cod
            )

        case _:
            return group_component(box)


def mlp_haar(box: Box) -> Diagram | None:
    match box.name:
        case "f":
            return sequential(["flatten", "f_mlp", "make_square"], box.dom, box.cod)

        case "gamma":
            return orth_haar(box.cod, box.dom)

        case _:
            return group_component(box)


def mlp_mlphaar(box: Box) -> Diagram | None:
    match box.name:
        case "f":
            return sequential(["flatten", "f_mlp", "make_square"], box.dom, box.cod)

        case "gamma_1":
            return noise_outsourced(
                ["flatten", "gamma_mlp", "make_square", "qr"], box.dom, box.cod
            )

        case "gamma_0":
            return orth_haar(box.cod, box.dom)

        case _:
            return group_component(box)


def mlp_emlp(box: Box) -> Diagram | None:
    match box.name:
        case "f":
            return sequential(["flatten", "f_mlp", "make_square"], box.dom, box.cod)

        case "gamma":
            return noise_outsourced(
                [
                    "append",
                    "transpose",
                    "flatten",
                    "gamma_emlp",
                    "make_square",
                    "transpose",
                    "qr",
                ],
                box.dom,
                box.cod,
            )

        case _:
            return group_component(box)


def group_component(box: Box) -> Diagram | None:
    match box.name:
        case "mul":
            return Box("matmul", box.dom, box.cod)

        case "inv":
            return Box("transpose", box.dom, box.cod)

        case "act_X":
            return Box("matmul", box.dom, box.cod)

        case "act_Y":
            return orth_right_matmul_by_transpose(box.dom[0], box.dom[1])

        case "haar":
            return orth_haar(box.cod, box.dom)
