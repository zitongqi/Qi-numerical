# evaluate_stat.py
import numpy as np

def evaluate_stat(elenodes, gpx, gpw, lam=48.0):
    """
    Fkt. XVII (Aufgabenblatt 7)

    elenodes : (4,2) array-like
               element node coordinates [[x,y],...]
    gpx      : (ngp,2) array-like
               Gauss points (xi, eta) on reference element
    gpw      : (ngp,)  array-like
               Gauss weights (already 2D weights)
    lam      : float
               thermal conductivity λ (default: 48.0)

    Returns:
      elemat : (4,4) element matrix A^(e)
      elevec : (4,)  element vector f^(e) (zero here)
    """
    elenodes = np.asarray(elenodes, dtype=float)
    gpx = np.asarray(gpx, dtype=float)
    gpw = np.asarray(gpw, dtype=float)

    elemat = np.zeros((4, 4), dtype=float)
    elevec = np.zeros(4, dtype=float)  # q̇ = 0 → no source term

    # loop over Gauss points
    for k in range(len(gpw)):
        xi, eta = gpx[k]
        w = gpw[k]

        # bilinear shape function derivatives on reference element
        dN_dxi = np.array([
            [-(1.0 - eta), -(1.0 - xi)],
            [ +(1.0 - eta), -(1.0 + xi)],
            [ +(1.0 + eta), +(1.0 + xi)],
            [-(1.0 + eta), +(1.0 - xi)],
        ], dtype=float) * 0.25

        # Jacobian
        J = dN_dxi.T @ elenodes
        detJ = np.linalg.det(J)
        invJ = np.linalg.inv(J)

        # gradients in physical coordinates
        # ∇_x N = J^{-T} ∇_ξ N
        gradN = dN_dxi @ invJ.T   # (4,2)

        # assemble element stiffness matrix
        for i in range(4):
            for j in range(4):
                elemat[i, j] += (
                    lam
                    * np.dot(gradN[i], gradN[j])
                    * detJ
                    * w
                )

    return elemat, elevec
