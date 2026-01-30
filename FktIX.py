import numpy as np

# def OST(theta, timestep, M, B, C, sol):
#     """
#     Einschritt-θ-Verfahren

#     Parameters:
#     theta     : θ
#     timestep  : Δt
#     M         : M
#     B         : [B_n, B_{n+1}]
#     C         : [C_n, C_{n+1}]
#     sol       : φ_n
#     """
#     dt = timestep
#     Bn, Bn1 = B
#     Cn, Cn1 = C

#     LHS = M - theta * dt * Bn1
#     RHS = M * sol + dt * ((1-theta)*(Bn*sol + Cn) + theta*Cn1)

#     return LHS, RHS
def OST(theta, timestep, M, B, C, sol):
    """
    One-Step-Theta (OST) method, scalar version.

    LHS = M - theta*dt*B
    RHS = (M + (1-theta)*dt*B)*sol + dt*(theta*C + (1-theta)*C)
    """
    LHS = M - theta * timestep * B
    RHS = (M + (1 - theta) * timestep * B) * sol + timestep * (theta * C + (1 - theta) * C)
    return LHS, RHS
