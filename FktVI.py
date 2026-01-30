# =========================
# Fkt. VI: 2D Gauß-Gewichte auf Ωref = [-1,1]x[-1,1]
# =========================
import numpy as np
from FktIV import gw  # 如果你文件名是 fkt4.py，就改成 from fkt4 import gw

def gw2dref(n: int) -> np.ndarray:
    """
    Input:
        n : Anzahl der Integrationspunkte in einer Richtung (n = 1,2,3)
    Output:
        w2d : numpy array shape (n*n, 1)  (Spaltenvektor)
    """
    w_1d = np.asarray(gw(n), dtype=float)

    # Tensorprodukt: w_ij = w_i * w_j
    w2d = []
    for w_eta in w_1d:
        for w_xi in w_1d:
            w2d.append(w_xi * w_eta)

    return np.asarray(w2d, dtype=float).reshape(-1, 1)
