# fkt1_linquadref.py
import numpy as np

def linquadref(xi: float, eta: float) -> np.ndarray:
    """
    Fkt. I: Return bilinear Lagrange shape functions N_i(xi,eta) for Q1 element.
    Node order:
        1: (-1,-1), 2: ( 1,-1), 3: ( 1, 1), 4: (-1, 1)
    Returns:
        N (shape (4,)): [N1, N2, N3, N4]
    """
    N1 = 0.25 * (1.0 - xi) * (1.0 - eta)
    N2 = 0.25 * (1.0 + xi) * (1.0 - eta)
    N3 = 0.25 * (1.0 + xi) * (1.0 + eta)
    N4 = 0.25 * (1.0 - xi) * (1.0 + eta)
    return np.array([N1, N2, N3, N4], dtype=float)

if __name__ == "__main__":
    # Sheet tests
    print("Test Fkt. I: linquadref")
    print("(xi,eta)=(0.0,0.0) ->", linquadref(0.0, 0.0))
    print("(xi,eta)=(0.577,-0.577) ->", linquadref(0.577, -0.577))
