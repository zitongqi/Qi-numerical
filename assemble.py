import numpy as np

def assemble(elemat, elevec, sysmat, rhs, ele):
    """
    Assemble element matrix/vector into global system

    Parameters
    ----------
    elemat : (4,4) ndarray
        Element matrix
    elevec : (4,1) or (4,) ndarray
        Element vector
    sysmat : (N,N) ndarray
        Global system matrix
    rhs : (N,) ndarray
        Global right-hand side
    ele : array-like of length 4
        Element node indices (0-based!)
    """

    a = np.zeros_like(sysmat)
    b = np.zeros_like(rhs)

    for i in range(4):
        for j in range(4):
            a[ele[i], ele[j]] = elemat[i, j]
        b[ele[i]] = elevec[i]

    sysmat = sysmat + a
    rhs = rhs + b

    return sysmat, rhs
