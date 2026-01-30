def AM3(timestep, M, B, C, sol):
    LHS = M - 5 * timestep * B / 12
    RHS = (
        (M + 8 * timestep * B / 12) * sol[0]
        - timestep * B * sol[1] / 12
        + (5 * timestep * C + 8 * timestep * C - timestep * C) / 12
    )
    return LHS, RHS
