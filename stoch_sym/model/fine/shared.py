from discopy.monoidal import Ty
from discopy.markov import Box, Copy, Diagram, Id, Swap, Discard


def orth2_left_matmul_and_right_matmul_by_transpose(G: Ty, X: Ty) -> Diagram:
    matmul = Box("matmul", G @ X, X)
    rmatmul = orth_right_matmul_by_transpose(G, X)
    return Swap(G, G) @ X >> G @ matmul >> rmatmul


def orth2_right_matmul_by_transpose_and_left_matmul(G: Ty, X: Ty) -> Diagram:
    return Swap(G, G) @ X >> orth2_left_matmul_and_right_matmul_by_transpose(G, X)


def orth2_inv(G: Ty) -> Diagram:
    inv = Box("transpose", G, G)
    return inv @ inv


def orth2_mul(G: Ty) -> Diagram:
    matmul = Box("matmul", G @ G, G)
    return (G @ Swap(G, G) @ G) >> (matmul @ matmul)


def orth2_trivial_action(G: Ty, X: Ty) -> Diagram:
    return Discard(G) @ Discard(G) @ X


def orth2_haar(G: Ty, X: Ty) -> Diagram:
    return Copy(X) >> (orth_haar(G, X) @ orth_haar(G, X))


def orth_conjugation(G: Ty, X: Ty) -> Diagram:
    return (Copy(G) @ X) >> orth2_left_matmul_and_right_matmul_by_transpose(G, X)


def orth_right_matmul_by_transpose(G: Ty, X: Ty) -> Diagram:
    return Swap(G, X) >> (X @ Box("transpose", G, G)) >> Box("matmul", X @ G, X)


def orth_trivial_action(G: Ty, Y: Ty) -> Diagram:
    # TODO: Replace with Discard(G) (or just inline)
    return Copy(G, n=0) @ Y


def orth_haar(G: Ty, X: Ty) -> Diagram:
    return Box("orthogonal_haar", X, G)


# TODO: Rename G argument
def noise_outsourced(layers: list[str], X: Ty, G: Ty) -> Diagram:
    U = new_ty()

    gaussian = Box("gaussian", X, U)
    network = sequential(layers, X @ U, G)

    return Copy(X) >> X @ gaussian >> network


def sequential(layers: list[str], input_ty: Ty, output_ty: Ty) -> Diagram:
    x = input_ty
    result = Id(input_ty)
    for layer in layers[:-1]:
        y = new_ty()
        result = result >> Box(layer, x, y)
        x = y

    result = result >> Box(layers[-1], x, output_ty)

    return result


_ty_counter = 0


def new_ty(prefix="u") -> Ty:
    global _ty_counter
    ty = Ty(prefix + str(_ty_counter))
    _ty_counter += 1
    return ty
