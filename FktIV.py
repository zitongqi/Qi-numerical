# =========================
# Fkt. IV: Gauß-Gewichte
# =========================
import numpy as np

def gw(n):
    """
    Rückgabewert:
    w : 1D numpy array der Gauß-Gewichte
    """
    if n == 1:
        w = np.array([2.0])

    elif n == 2:
        w = np.array([1.0, 1.0])

    elif n == 3:
        w = np.array([5.0/9.0, 8.0/9.0, 5.0/9.0])

    else:
        raise ValueError("gw(n): n must be 1, 2 or 3")

    return w
