# =========================
# Fkt. V: 2D Gauß-Punkte auf Ωref = [-1,1]x[-1,1]
# =========================
import numpy as np
from FktIII import gx  # 如果你文件名是 fkt3.py，就改成 from fkt3 import gx

def gx2dref(n: int) -> np.ndarray:
    """
    Input:
        n : Anzahl der Integrationspunkte in einer Richtung (n = 1,2,3)
    Output:
        pts : numpy array shape (n*n, 2)
              Zeile i: [xi, eta]
    """
    xi_1d = np.asarray(gx(n), dtype=float)
    eta_1d = np.asarray(gx(n), dtype=float)

    pts = []
    for eta in eta_1d:
        for xi in xi_1d:
            pts.append([xi, eta])

    return np.asarray(pts, dtype=float)
