import numpy as np

def OST(theta, timestep, M, B, C, sol):
    """
    B = [B_n, B_{n+1}]
    C = [C_n, C_{n+1}]
    sol = φ_n
    """
    M   = float(M)
    sol = float(sol)

    Bn, Bn1 = B
    Cn, Cn1 = C

    LHS = M - theta * timestep * Bn1
    RHS = M * sol + timestep * ((1 - theta) * (Bn * sol + Cn) + theta * Cn1)

    return LHS, RHS

