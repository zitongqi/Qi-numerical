def AB2(timestep, M, B, C, sol):
    """
    Adams-Bashforth 2. Ordnung

    Parameters:
    timestep : Δt
    M        : M
    B        : [B_n, B_{n-1}]
    C        : [C_n, C_{n-1}]
    sol      : [φ_n, φ_{n-1}]
    """
    dt = timestep
    Bn, Bnm1 = B
    Cn, Cnm1 = C
    phin, phinm1 = sol

    LHS = M
    RHS = (
        M * phin
        + dt * (1.5*(Bn*phin + Cn) - 0.5*(Bnm1*phinm1 + Cnm1))
    )

    return LHS, RHS
