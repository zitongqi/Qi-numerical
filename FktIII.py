# =========================
# Fkt. III: Gauß-Punkte
# =========================
import numpy as np

def gx(n):
    """
    Rückgabewert:
    xi : 1D numpy array der Gauß-Integrationspunkte auf [-1, 1]
    """
    if n == 1:
        xi = np.array([0.0])

    elif n == 2:
        a = 1.0 / np.sqrt(3.0)
        xi = np.array([-a, a])

    elif n == 3:
        b = np.sqrt(3.0 / 5.0)
        xi = np.array([-b, 0.0, b])

    else:
        raise ValueError("gx(n): n must be 1, 2 or 3")

    return xi
