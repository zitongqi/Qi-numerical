import numpy as np

from linquadref import linquadref
from linquadderivref import linquadderivref
from getJacobian import getJacobian
from OST import OST
from AB2 import AB2
from AM3 import AM3
from BDF2 import BDF2


def evaluate_instat(
    elenodes,
    gpx,
    gpw,
    elesol,
    eleosol,
    timInt_m,
    timestep,
    theta,
    firststep=None
):
    """
    Python translation of evaluate_instat.m
    """

    # -------------------------------------------------
    # parameters
    # -------------------------------------------------
    lamda = 48.0
    p = 7800.0
    c = 452.0

    # -------------------------------------------------
    # initialize matrices
    # -------------------------------------------------
    M = np.zeros((4, 4))
    B = np.zeros((4, 4))

    # -------------------------------------------------
    # element matrices
    # -------------------------------------------------
    ngp = gpw.shape[0]

    for i in range(4):
        for j in range(4):

            s1 = 0.0
            s2 = 0.0

            for k in range(ngp):
                xi  = gpx[k, 0]
                eta = gpx[k, 1]

                deriv = linquadderivref(xi, eta)   # (4,2)
                val   = linquadref(xi, eta)        # (4,)

                J, detJ, invJ = getJacobian(elenodes, xi, eta)

                # Matlab: Nv = val(i,:), Nt = val(j,:)
                Nv = val[i]        # scalar
                Nt = val[j]        # scalar

                # Matlab: gNv = deriv(i,:) * invJ
                gNv = deriv[i, :] @ invJ
                gNt = deriv[j, :] @ invJ

                s1 += p * c * Nv * Nt * detJ * gpw[k]
                s2 += lamda * np.dot(gNv, gNt) * detJ * gpw[k]

            M[i, j] = s1
            B[i, j] = -s2

    C = 0.0

    # -------------------------------------------------
    # time integration
    # -------------------------------------------------
    LHS = np.zeros((4, 4))
    RHS = np.zeros(4)

    if timInt_m == 1:          # OST
        for i in range(4):
            for j in range(4):
                a, b = OST(
                    theta,
                    timestep,
                    M[i, j],
                    [B[i, j], B[i, j]],   # B^n, B^{n+1}
                    [C, C],
                    elesol[j]
                )
                LHS[i, j] = a
                RHS[i] += b

    elif timInt_m == 2:        # AB2
        for i in range(4):
            for j in range(4):
                a, b = AB2(
                    timestep,
                    M[i, j],
                    [B[i, j], B[i, j]],
                    [C, C],
                    [elesol[j], eleosol[j]]
                )
                LHS[i, j] = a
                RHS[i] = b

    elif timInt_m == 3:        # AM3
        for i in range(4):
            for j in range(4):
                a, b = AM3(
                    timestep,
                    M[i, j],
                    B[i, j],
                    C,
                    [elesol[j], eleosol[j]]
                )
                LHS[i, j] = a
                RHS[i] = b

    elif timInt_m == 4:        # BDF2
        for i in range(4):
            for j in range(4):
                a, b = BDF2(
                    timestep,
                    M[i, j],
                    B[i, j],
                    C,
                    [elesol[j], eleosol[j]]
                )
                LHS[i, j] += a
                RHS[i] += b

    else:
        raise ValueError("Unknown time integration method")

    # -------------------------------------------------
    # output
    # -------------------------------------------------
    elemat = LHS
    elevec = RHS.reshape(-1, 1)

    return elemat, elevec
