from discopy.markov import Box, Diagram

from ..shared import (
    new_ty,
    orth_haar,
    sequential,
    orth_right_matmul_by_transpose,
    noise_outsourced,
)


# TODO: Move this (and other similar ones) into their own file, where the action is trivial
def mlp_none(box: Box) -> Diagram | None:
    match box.name:
        case "f":
            return sequential(["flatten", "f_mlp"], box.dom, box.cod)

        case _:
            return group_component(box)


def emlp_none(box: Box) -> Diagram | None:
    match box.name:
        case "f":
            return sequential(["transpose", "flatten", "f_emlp"], box.dom, box.cod)

        case _:
            return group_component(box)


def mlp_haar(box: Box) -> Diagram | None:
    match box.name:
        case "f":
            return sequential(["flatten", "f_mlp"], box.dom, box.cod)

        case "gamma_0":
            return orth_haar(box.cod, box.dom)

        case _:
            return group_component(box)


def mlp_mlphaar(box: Box) -> Diagram | None:
    match box.name:
        case "f":
            return sequential(["flatten", "f_mlp"], box.dom, box.cod)

        case "gamma_0":
            return noise_outsourced(
                ["flatten", "gamma_mlp", "make_square", "qr"], box.dom, box.cod
            )

        case "gamma_1":
            return orth_haar(box.cod, box.dom)

        case _:
            return group_component(box)


def mlp_emlp(box: Box) -> Diagram | None:
    match box.name:
        case "f":
            return sequential(["flatten", "f_mlp"], box.dom, box.cod)

        case "gamma_0":
            # TODO: Could make a component "vec" that does transpose and flatten
            transpose = Box("transpose", box.dom, new_ty())
            flatten = Box("flatten", transpose.cod, new_ty())
            backbone = noise_outsourced(
                [
                    "cat",
                    "gamma_emlp",
                    "make_square",
                    "transpose",
                    "qr",
                ],
                flatten.cod,
                box.cod,
            )

            return transpose >> flatten >> backbone

        case _:
            return group_component(box)


def group_component(box: Box) -> Diagram | None:
    match box.name:
        case "mul":
            return Box("matmul", box.dom, box.cod)

        case "inv":
            return Box("transpose", box.dom, box.cod)

        case "act_X":
            G, Xy = box.dom
            X = new_ty()
            y = new_ty()
            pop = Box("pop", Xy, X @ y)
            matmul = Box("matmul", G @ X, X)
            insert = Box("insert", X @ y, Xy)
            return (G @ pop) >> (matmul @ y) >> insert

        case "act_Y":
            G, y_hat = box.dom
            U = new_ty()
            unsqueeze = Box("unsqueeze", y_hat, U)
            matmul = Box("matmul", G @ U, U)
            squeeze = Box("squeeze", U, y_hat)
            return (G @ unsqueeze) >> matmul >> squeeze

        case "haar":
            return orth_haar(box.cod, box.dom)
