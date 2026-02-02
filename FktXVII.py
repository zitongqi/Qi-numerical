# evaluate_stat.py
import numpy as np

from linquadref import linquadref
from linquadderivref import linquadderivref
from getJacobian import getJacobian


def evaluate_stat(elenodes, gpx, gpw, lam=48.0):
    """
    Fkt. XVII (Aufgabenblatt 7)
    Stationary heat conduction (Q1 element)

    elenodes : (4,2) array
        Element node coordinates
    gpx : (ngp,2) array
        Gauss points (xi, eta) on reference element
    gpw : (ngp,) array
        Gauss weights (2D tensor product)
    lam : float
        Thermal conductivity

    Returns
    -------
    elemat : (4,4) ndarray
        Element stiffness matrix
    elevec : (4,) ndarray
        Element load vector (zero)
    """

    elenodes = np.asarray(elenodes, dtype=float)
    gpx = np.asarray(gpx, dtype=float)
    gpw = np.asarray(gpw, dtype=float)

    elemat = np.zeros((4, 4), dtype=float)
    elevec = np.zeros(4, dtype=float)   # no source term

    ngp = gpw.shape[0]

    # -------------------------------------------------
    # Gauss integration
    # -------------------------------------------------
    for k in range(ngp):
        xi, eta = gpx[k]
        w = gpw[k]

        # shape function derivatives on reference element
        dN_dxi = linquadderivref(xi, eta)   # (4,2)

        # Jacobian and inverse
        J, detJ, invJ = getJacobian(elenodes, xi, eta)

        # gradients in physical coordinates
        # grad N = dN/dxi * invJ
        gradN = dN_dxi @ invJ   # (4,2)

        # assemble stiffness matrix
        for i in range(4):
            for j in range(4):
                elemat[i, j] += (
                    lam
                    * np.dot(gradN[i], gradN[j])
                    * detJ
                    * w
                )

    return elemat, elevec
