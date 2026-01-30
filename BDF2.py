def BDF2(timestep, M, B, C, sol):
    LHS = 3 * M / 2 - timestep * B
    RHS = 2 * M * sol[0] - M * sol[1] / 2 + timestep * C
    return LHS, RHS
