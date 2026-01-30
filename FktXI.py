def AM3(timestep, M, B, C, sol):
    """
    Adams-Moulton 3. Ordnung

    Parameters:
    timestep : Δt
    M        : M
    B        : [B_{n+1}, B_n, B_{n-1}]
    C        : [C_{n+1}, C_n, C_{n-1}]
    sol      : [φ_n, φ_{n-1}]
    """
    dt = timestep
    Bn1, Bn, Bnm1 = B
    Cn1, Cn, Cnm1 = C
    phin, phinm1 = sol

    LHS = M - (5/12)*dt*Bn1
    RHS = (
        M*phin
        + dt * (
            (2/3)*(Bn*phin + Cn)
            - (1/12)*(Bnm1*phinm1 + Cnm1)
            + (5/12)*Cn1
        )
    )

    return LHS, RHS
