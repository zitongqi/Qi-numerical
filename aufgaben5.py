# # ============================================================
# # Aufgabenblatt 5 – Aufgabe 1
# # Hauptprogramm
# # ============================================================

# import numpy as np

# # 导入你已经写好的函数
# from FktIII import gx
# from FktIV import gw


# # ============================================================
# # Integrand
# # ============================================================
# def f(x):
#     return (x**5) / (1.0 + x)**5


# # ============================================================
# # 1D Gauß-Quadratur auf [a, b]
# # ============================================================
# def gauss_1d_integral(f, a, b, n):
#     xi = gx(n)
#     wi = gw(n)

#     # affine Transformation von [-1,1] -> [a,b]
#     x = 0.5 * (b - a) * xi + 0.5 * (a + b)

#     # Gauß-Quadratur
#     I = 0.5 * (b - a) * np.sum(wi * f(x))
#     return I


# # ============================================================
# # Hauptprogramm
# # ============================================================
# if __name__ == "__main__":

#     # Intervall und exakte Lösung
#     a = 0.0
#     b = 4.0
#     I_exact = 0.556543771162832

#     # --------------------------------------------------------
#     # Mittelpunktregel
#     # --------------------------------------------------------
#     x_mid = 0.5 * (a + b)
#     I_mid = (b - a) * f(x_mid)

#     # --------------------------------------------------------
#     # Trapezregel
#     # --------------------------------------------------------
#     I_trap = 0.5 * (b - a) * (f(a) + f(b))

#     # --------------------------------------------------------
#     # Ausgabe
#     # --------------------------------------------------------
#     print("==========================================")
#     print("Aufgabenblatt 5 – Aufgabe 1")
#     print("==========================================\n")

#     print(f"Exakte Integration        : {I_exact:.12f}\n")

#     print(f"Mittelpunktregel          : {I_mid:.6f}")
#     print(f"Trapezregel               : {I_trap:.6f}\n")

#     for n in [1, 2, 3]:
#         I_g = gauss_1d_integral(f, a, b, n)
#         print(f"Gauß-Quadratur (n = {n})    : {I_g:.6f}")

#     print("\n==========================================")


# ============================================================
# Aufgabenblatt 5 – Aufgabe 2
# Mehrdimensionale Gauß-Quadratur
# Berechnung von m_12 = ∫_Ωe N1 * N2 dΩe
# ============================================================

import numpy as np

# --- Import deiner Funktionen ---
from FktV import gx2dref
from FktVI import gw2dref
from FktVII import getxPos
from FktVII import N_quad4
from FktVIII import getJacobian




# ============================================================
# Hauptprogramm
# ============================================================
if __name__ == "__main__":

    # --------------------------------------------------------
    # Knotenkoordinaten des Elements
    # Reihenfolge:
    # 1: (-1,-1), 2: (1,-1), 3: (1,1), 4: (-1,1)
    # --------------------------------------------------------
    nodes = np.array([
        [2.0, 1.0],  # node 1
        [4.0, 1.0],  # node 2
        [4.0, 3.0],  # node 3
        [2.0, 2.0],  # node 4
    ], dtype=float)

    print("==============================================")
    print("Aufgabenblatt 5 – Aufgabe 2")
    print("Berechnung von m_12 = ∫ N1 * N2 dΩ")
    print("==============================================\n")

    # --------------------------------------------------------
    # Gauss-Integration mit n = 1,2,3
    # --------------------------------------------------------
    for n in [1, 2, 3]:

        gauss_pts = gx2dref(n)    # (n*n, 2)
        gauss_w   = gw2dref(n)    # (n*n, 1)

        m12 = 0.0

        for k in range(len(gauss_pts)):

            xi  = gauss_pts[k, 0]
            eta = gauss_pts[k, 1]
            w   = gauss_w[k, 0]

            # Ansatzfunktionen im Referenzelement
            N = N_quad4(xi, eta)

            # Jacobian
            J, detJ, invJ = getJacobian(nodes, xi, eta)

            # Beitrag zum Integral
            m12 += N[0] * N[1] * detJ * w

        print(f"Gauß-Quadratur n = {n} x {n} :  m12 = {m12:.6f}")

    print("\n==============================================")
