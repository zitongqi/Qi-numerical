def BDF2(timestep, M, B, C, sol):
    """
    BDF2-Verfahren

    Parameters:
    timestep : Δt
    M        : M
    B        : B_{n+1}
    C        : C_{n+1}
    sol      : [φ_n, φ_{n-1}]
    """
    dt = timestep
    phin, phinm1 = sol

    LHS = (3/(2*dt))*M - B
    RHS = (2/dt)*M*phin - (1/(2*dt))*M*phinm1 + C

    return LHS, RHS
