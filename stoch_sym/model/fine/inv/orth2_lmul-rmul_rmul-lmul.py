from discopy.monoidal import Ob
from discopy.markov import Box, Diagram

from ..shared import (
    orth2_right_matmul_by_transpose_and_left_matmul,
    orth2_left_matmul_and_right_matmul_by_transpose,
    orth2_mul,
    orth2_inv,
    orth2_haar,
    sequential,
    orth_right_matmul_by_transpose,
    noise_outsourced,
    new_ty,
)


def mlp_none_ob(ob: Ob) -> Diagram | None:
    match ob.name:
        case "G":
            return ob @ ob


# TODO: Move this (and other similar ones) into their own file, where the action is trivial
def mlp_none(box: Box) -> Diagram | None:
    match box.name:
        case "f":
            return sequential(["flatten", "f_mlp", "make_square"], box.dom, box.cod)

        case _:
            return group_component(box)


def mlp_haar_ob(ob: Ob) -> Diagram | None:
    match ob.name:
        case "G":
            return ob @ ob


def mlp_haar(box: Box) -> Diagram | None:
    match box.name:
        case "f":
            return sequential(["flatten", "f_mlp", "make_square"], box.dom, box.cod)

        # TODO: Rename (everywhere) to match paper
        case "gamma_0":
            X = box.dom
            G = box.cod
            return orth2_haar(G, X)

        case _:
            return group_component(box)


def mlp_mlphaar_ob(ob: Ob) -> Diagram | None:
    match ob.name:
        case "G":
            return ob @ ob


def mlp_mlphaar(box: Box) -> Diagram | None:
    match box.name:
        case "f":
            return sequential(["flatten", "f_mlp", "make_square"], box.dom, box.cod)

        # TODO: Rename (everywhere) to match paper
        case "gamma_0":
            U = new_ty()
            backbone = noise_outsourced(["flatten", "gamma_mlp"], box.dom, U)
            split = Box("split", U, U @ U)
            pr = sequential(["make_square", "qr"], U, box.cod)
            return backbone >> split >> (pr @ pr)

        case "gamma_1":
            X = box.dom
            G = box.cod
            return orth2_haar(G, X)

        case _:
            return group_component(box)


def group_component(box: Box) -> Diagram | None:
    match box.name:
        case "mul":
            (G, _) = box.dom
            return orth2_mul(G)

        case "inv":
            (G,) = box.dom
            return orth2_inv(G)

        case "act_X":
            G, X = box.dom
            return orth2_left_matmul_and_right_matmul_by_transpose(G, X)

        case "act_Y":
            G, Y = box.dom
            return orth2_right_matmul_by_transpose_and_left_matmul(G, Y)

        case "haar":
            X = box.dom
            G = box.cod
            return orth2_haar(G, X)
