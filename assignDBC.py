def assignDBC(sysmat, rhs, dbc):
    """
    Apply Dirichlet boundary conditions

    Parameters
    ----------
    sysmat : (N,N) ndarray
        Global system matrix
    rhs : (N,) ndarray
        Global right-hand side
    dbc : (n,2) ndarray
        dbc[:,0] = node indices (0-based)
        dbc[:,1] = prescribed values
    """

    for i in range(dbc.shape[0]):
        node = int(dbc[i, 0])
        value = dbc[i, 1]

        sysmat[node, :] = 0.0
        sysmat[node, node] = 1.0
        rhs[node] = value

    return sysmat, rhs
