import numpy as np
from FktI import linquadref
from FktII import linquadderivref
from FktVIII import getJacobian
from FktIX import OST
from FktX import AB2
from FktXI import AM3
from FktXII import BDF2


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
    # material parameters
    lamda = 48.0
    p = 7800.0
    c = 452.0

    M = np.zeros((4, 4))
    B = np.zeros((4, 4))
    ngp = gpw.shape[0]

    # build M and B
    for i in range(4):
        for j in range(4):
            s1 = 0.0
            s2 = 0.0

            for k in range(ngp):
                xi = gpx[k, 0]
                eta = gpx[k, 1]

                deriv = linquadderivref(xi, eta)   # (4,2)
                val = linquadref(xi, eta)          # (4,)

                J, detJ, invJ = getJacobian(elenodes, xi, eta)

                Nv = val[i]
                Nt = val[j]

                gNv = deriv[i, :] @ invJ
                gNt = deriv[j, :] @ invJ

                s1 += p * c * Nv * Nt * detJ * gpw[k]
                s2 += lamda * np.dot(gNv, gNt) * detJ * gpw[k]

            M[i, j] = s1
            B[i, j] = -s2

    C = 0.0

    LHS = np.zeros((4, 4))
    RHS = np.zeros(4)

    # ---- time integration (scalar entries) ----
    if timInt_m == 1:  # OST
        for i in range(4):
            for j in range(4):
                # ✅ 关键：OST 只传标量 B[i,j]、标量 C
                a, b = OST(theta, timestep, M[i, j], B[i, j], C, elesol[j])
                LHS[i, j] = a
                RHS[i] += b

    elif timInt_m == 2:  # AB2
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
                RHS[i] += b

    elif timInt_m == 3:  # AM3
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
                RHS[i] += b

    elif timInt_m == 4:  # BDF2
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

    return LHS, RHS.reshape(-1, 1)