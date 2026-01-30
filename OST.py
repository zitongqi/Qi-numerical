# def OST(theta, timestep, M, B, C, sol):
#     LHS = M - theta * timestep * B
#     RHS = (M + (1 - theta) * timestep * B) * sol \
#           + timestep * (theta * C + (1 - theta) * C)
#     return LHS, RHS

def OST(theta, timestep, M, B, C, sol):
    """
    One-Step-Theta method.
    Supports scalar B/C or list/array B=[Bn,Bn1], C=[Cn,Cn1].
    """

    # allow both scalar and [Bn, Bn1]
    if isinstance(B, (list, tuple)):
        Bn = float(B[0])
        Bn1 = float(B[1])
    else:
        Bn = float(B)
        Bn1 = float(B)

    if isinstance(C, (list, tuple)):
        Cn = float(C[0])
        Cn1 = float(C[1])
    else:
        Cn = float(C)
        Cn1 = float(C)

    # theta method:
    # (M - theta*dt*B^{n+1}) u^{n+1} = (M + (1-theta)*dt*B^{n}) u^{n} + dt( theta*C^{n+1} + (1-theta)*C^{n})
    LHS = M - theta * timestep * Bn1
    RHS = (M + (1 - theta) * timestep * Bn) * sol + timestep * (theta * Cn1 + (1 - theta) * Cn)

    return LHS, RHS
