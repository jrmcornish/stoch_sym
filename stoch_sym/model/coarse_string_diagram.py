from discopy.monoidal import Ty
from discopy.markov import Box, Copy, Diagram, Id, Swap


# TODO: Could do a version where we allow multiple samples of g per row of x?
# NOTE: This metric only really makes sense if model is a DeterministicKernel
# (otherwise it may not be zero even if model is equivariant - e.g. consider a
# situation where model(y|x) = Normal(y; 0, I) for all x, and G is e.g.
# orthogonal group)
# NOTE: this is not scale invariant!
def get_equi_score_string_diagram() -> Diagram:
    G = Ty("G")
    X = Ty("X")
    Y = Ty("Y")
    R = Ty("R")
    f = Box("ave(k)", X, Y)
    loss = Box("mse", Y @ Y, R)
    act_X = get_act(G, X)
    act_Y = get_act(G, Y)
    haar = Box("haar", X, G)

    return (
        Copy(X)
        >> (haar @ X)
        >> (Copy(G) @ Copy(X))
        >> G @ Swap(G, X) @ X
        >> act_X @ G @ f
        >> f @ act_Y
        >> loss
    )


def get_coarse_string_diagram(gamma_type: str) -> Diagram:
    X = Ty("X")
    Y = Ty("Y")
    f = Box("f", X, Y)

    if gamma_type == "none":
        return f

    G = Ty("G")

    gamma = get_gamma_string_diagram(
        gamma=new_gamma_box(G), depth=get_depth(gamma_type)
    )

    return sym(f, G, Id(G), gamma)


def get_depth(gamma_type):
    match gamma_type:
        case "haar" | "emlp":
            return 0

        case "mlp-haar":
            return 1

        case _:
            raise NotImplementedError


def get_gamma_string_diagram(gamma, depth: int) -> Diagram:
    if depth == 0:
        return gamma

    G = gamma.cod

    return sym(gamma, G, Id(G), get_gamma_string_diagram(new_gamma_box(G), depth - 1))


def sym(f: Diagram, G: Ty, s: Box, gamma: Diagram) -> Diagram:
    k_plus = adjunction(f, G, s)
    return precompose(k_plus, gamma)


def adjunction(
    f: Diagram,
    G: Ty,
    s: Box,
) -> Diagram:
    X = f.dom
    Y = f.cod

    act_X = get_act(G, X)
    act_Y = get_act(G, Y)
    inv = Box("inv", G, G)

    return (s >> Copy(G)) @ X >> G @ (inv @ X >> act_X >> f) >> act_Y


def get_act(G: Ty, X: Ty) -> Box:
    match G.name, X.name:
        case ("G", "G") | ("H", "H"):
            return Box("mul", G @ G, G)

        case ("H", "G"):
            return Box("incl", G, X) @ X >> Box(f"mul", X @ X, X)

        case _:
            return Box(f"act_{X.name}", G @ X, X)


def precompose(k_plus: Diagram, gamma: Diagram) -> Diagram:
    _, X = k_plus.dom
    return Copy(X) >> gamma @ X >> k_plus


_gamma_count = -1


def new_gamma_box(cod: Ty) -> Box:
    global _gamma_count
    _gamma_count += 1
    return Box(f"gamma_{_gamma_count}", Ty("X"), cod)
