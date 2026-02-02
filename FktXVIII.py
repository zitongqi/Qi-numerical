# assemble.py
import numpy as np

def assemble(elemat, elevec, sysmat, rhs, ele):
    """
    Fkt. XVIII (Aufgabenblatt 7)装配矩阵

    elemat : (nen,nen) element matrix A^(e)
    elevec : (nen,)    element vector f^(e)
    sysmat : (N,N)     global matrix A
    rhs    : (N,)      global vector f
    ele    : (nen,)    global node indices (as row vector), 1-based like in sheet

    Returns:
      sysmat, rhs (assembled)
    """
    elemat = np.asarray(elemat, dtype=float)
    elevec = np.asarray(elevec, dtype=float).reshape(-1)
    ele = np.asarray(ele, dtype=int).reshape(-1)

    nen = len(ele)

    for a in range(nen):
        A = ele[a] - 1  # to 0-based
        rhs[A] += elevec[a]
        for b in range(nen):
            B = ele[b] - 1
            sysmat[A, B] += elemat[a, b]

    return sysmat, rhs
