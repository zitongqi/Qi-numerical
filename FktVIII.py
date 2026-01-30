# =========================
# Fkt. VIII: Jacobian J(ξ,η), det(J), inv(J)
# =========================
import numpy as np

def _dN_quad4(xi: float, eta: float):
    """
    Ableitungen der bilinearen Q4-Ansatzfunktionen nach ξ und η.
    Output:
      dN_dxi  shape (4,)
      dN_deta shape (4,)
    Knotenreihenfolge:
      1: (-1,-1), 2: ( 1,-1), 3: ( 1, 1), 4: (-1, 1)
    """
    dN1_dxi  = -0.25 * (1.0 - eta)
    dN2_dxi  =  0.25 * (1.0 - eta)
    dN3_dxi  =  0.25 * (1.0 + eta)
    dN4_dxi  = -0.25 * (1.0 + eta)

    dN1_deta = -0.25 * (1.0 - xi)
    dN2_deta = -0.25 * (1.0 + xi)
    dN3_deta =  0.25 * (1.0 + xi)
    dN4_deta =  0.25 * (1.0 - xi)

    dN_dxi  = np.array([dN1_dxi,  dN2_dxi,  dN3_dxi,  dN4_dxi], dtype=float)
    dN_deta = np.array([dN1_deta, dN2_deta, dN3_deta, dN4_deta], dtype=float)
    return dN_dxi, dN_deta

def getJacobian(nodes, xi: float, eta: float):
    """
    Input:
      nodes: array-like shape (4,2)  (Zeile: Knoten i; Spalte: x,y)
      xi, eta: Position im Referenzelement
    Output:
      J      : numpy array shape (2,2)
      detJ   : float
      invJ   : numpy array shape (2,2)
    """
    nodes = np.asarray(nodes, dtype=float)
    if nodes.shape != (4, 2):
        raise ValueError("getJacobian: 'nodes' must have shape (4,2)")

    dN_dxi, dN_deta = _dN_quad4(xi, eta)

    # dx/dxi = Σ dNi/dxi * xi_node,  dy/dxi = Σ dNi/dxi * yi_node
    dx_dxi  = float(np.dot(dN_dxi,  nodes[:, 0]))
    dy_dxi  = float(np.dot(dN_dxi,  nodes[:, 1]))
    dx_deta = float(np.dot(dN_deta, nodes[:, 0]))
    dy_deta = float(np.dot(dN_deta, nodes[:, 1]))

    J = np.array([[dx_dxi,  dx_deta],
                  [dy_dxi,  dy_deta]], dtype=float)

    detJ = float(np.linalg.det(J))
    if abs(detJ) < 1e-14:
        raise ValueError("getJacobian: det(J) is ~0 (degenerate element)")

    invJ = np.linalg.inv(J)
    return J, detJ, invJ
