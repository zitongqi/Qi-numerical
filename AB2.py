def AB2(timestep, M, B, C, sol):
    """
    Adams-Bashforth 2 (AB2) time integration
    Python translation of AB2.m

    Parameters
    ----------
    timestep : float
    M : float
        Mass term (scalar)
    B : list or array of length 2
        [B^n, B^{n-1}]
    C : list or array of length 2
        [C^n, C^{n-1}]
    sol : list or array of length 2
        [u^n, u^{n-1}]
    """

    LHS = M

    RHS = (
        M * sol[0]
        + timestep / 2.0
        * (
            3.0 * (B[0] * sol[0] + C[0])
            - B[1] * sol[1]
            - C[1]
        )
    )

    return LHS, RHS
