import numpy as np
from linquadderivref import linquadderivref

def getJacobian(nodes, xi, eta):
    """
    Compute Jacobian matrix, determinant and inverse
    Python translation of getJacobian.m
    """

    deriv = linquadderivref(xi, eta)   # shape (nnode, 2)

    j11 = 0.0
    j12 = 0.0
    j21 = 0.0
    j22 = 0.0

    for i in range(nodes.shape[0]):
        j11 += nodes[i, 0] * deriv[i, 0]
        j12 += nodes[i, 0] * deriv[i, 1]
        j21 += nodes[i, 1] * deriv[i, 0]
        j22 += nodes[i, 1] * deriv[i, 1]

    J = np.array([[j11, j12],
                  [j21, j22]], dtype=float)

    detJ = np.linalg.det(J)
    invJ = np.linalg.inv(J)

    return J, detJ, invJ
