# fkt2_linquadderivref.py
import numpy as np

def linquadderivref(xi: float, eta: float) -> np.ndarray:
    """
    Fkt. II: Return derivatives of bilinear Lagrange shape functions for Q1 element.
    Returns:
        dN (shape (4,2)):
            dN[i,0] = ∂N_i/∂xi
            dN[i,1] = ∂N_i/∂eta
        Rows: i=1..4, Cols: [xi, eta]
    """
    dN1_dxi  = -0.25 * (1.0 - eta)
    dN1_deta = -0.25 * (1.0 - xi)

    dN2_dxi  =  0.25 * (1.0 - eta)
    dN2_deta = -0.25 * (1.0 + xi)

    dN3_dxi  =  0.25 * (1.0 + eta)
    dN3_deta =  0.25 * (1.0 + xi)

    dN4_dxi  = -0.25 * (1.0 + eta)
    dN4_deta =  0.25 * (1.0 - xi)

    return np.array([
        [dN1_dxi,  dN1_deta],
        [dN2_dxi,  dN2_deta],
        [dN3_dxi,  dN3_deta],
        [dN4_dxi,  dN4_deta],
    ], dtype=float)

if __name__ == "__main__":
    # Sheet tests
    print("Test Fkt. II: linquadderivref")
    print("(xi,eta)=(0.0,0.0) ->\n", linquadderivref(0.0, 0.0))
    print("(xi,eta)=(0.577,-0.577) ->\n", linquadderivref(0.577, -0.577))
