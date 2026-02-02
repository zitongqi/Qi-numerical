# assignDBC.py
import numpy as np

def assignDBC(sysmat, rhs, dbc):
    """
    Fkt. XIX (Aufgabenblatt 7)将已知的节点温度直接替换

    sysmat : (N,N) global matrix A
    rhs    : (N,)  global vector f
    dbc    : (ndbc,2) matrix: [node_index, value]
             node_index is 1-based as in sheet

    Returns:
      sysmat, rhs after applying Dirichlet BC
    """
    dbc = np.asarray(dbc, dtype=float)

    for k in range(dbc.shape[0]):
        idx = int(dbc[k, 0]) - 1
        val = float(dbc[k, 1])

        sysmat[idx, :] = 0.0
        sysmat[idx, idx] = 1.0
        rhs[idx] = val

    return sysmat, rhs
