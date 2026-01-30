import numpy as np

def linquadderivref(xi, eta):
    deriv = np.array([
        [-0.25 * (1 - eta), -0.25 * (1 - xi)],
        [ 0.25 * (1 - eta), -0.25 * (1 + xi)],
        [ 0.25 * (1 + eta),  0.25 * (1 + xi)],
        [-0.25 * (1 + eta),  0.25 * (1 - xi)]
    ], dtype=float)
    return deriv
