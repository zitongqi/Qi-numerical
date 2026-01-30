# =========================
# Fkt. VII: Isoparametrische Abbildung x(ξ,η)
# =========================
import numpy as np

def _N_quad4(xi: float, eta: float) -> np.ndarray:
    """
    Bilineare Ansatzfunktionen (Q4) auf [-1,1]x[-1,1]
    Knotenreihenfolge:
      1: (-1,-1), 2: ( 1,-1), 3: ( 1, 1), 4: (-1, 1)
    Output:
      N shape (4,)
    """
    N1 = 0.25 * (1.0 - xi) * (1.0 - eta)
    N2 = 0.25 * (1.0 + xi) * (1.0 - eta)
    N3 = 0.25 * (1.0 + xi) * (1.0 + eta)
    N4 = 0.25 * (1.0 - xi) * (1.0 + eta)
    return np.array([N1, N2, N3, N4], dtype=float)

def getxPos(nodes, xi: float, eta: float) -> np.ndarray:
    """
    Input:
      nodes: array-like shape (4,2)  (Zeile: Knoten i; Spalte: x,y)
      xi, eta: Position im Referenzelement
    Output:
      xvec: Spaltenvektor shape (2,1) mit [x; y]
    """
    nodes = np.asarray(nodes, dtype=float)
    if nodes.shape != (4, 2):
        raise ValueError("getxPos: 'nodes' must have shape (4,2)")

    N = _N_quad4(xi, eta)  # (4,)
    x = float(np.dot(N, nodes[:, 0]))
    y = float(np.dot(N, nodes[:, 1]))
    return np.array([[x], [y]], dtype=float)
