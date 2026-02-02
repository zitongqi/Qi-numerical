# #E1
# import numpy as np


# # ==================================================
# # Newton-Verfahren für eine Variable
# # ==================================================
# def newton_1d(f, df, x0, tol=1e-12, itermax=100):
#     """
#     Newton-Verfahren zur Lösung von f(x) = 0

#     Parameter:
#     f       : Funktion f(x)
#     df      : Ableitung f'(x)
#     x0      : Startwert
#     tol     : Toleranz für |f(x)|
#     itermax : maximale Iterationszahl

#     Rückgabe:
#     x       : Näherung der Nullstelle
#     k       : Anzahl der Iterationen
#     """
#     x = x0

#     for k in range(itermax):
#         fx = f(x)

#         # Abbruchkriterium
#         if abs(fx) < tol:
#             return x, k

#         dfx = df(x)
#         if abs(dfx) < 1e-14:
#             raise ValueError("Ableitung zu klein – Newton-Verfahren bricht ab")

#         x = x - fx / dfx

#     raise RuntimeError("Newton-Verfahren konvergiert nicht innerhalb itermax")


# # ==================================================
# # Aufgabe 1a: f(x) = x^3 + 3^x, Startwert x0 = 0
# # ==================================================
# def f_a(x):
#     return x**3 + 3**x

# def df_a(x):
#     return 3*x**2 + 3**x * np.log(3)


# # ==================================================
# # Aufgabe 1b: f(x) = arctan(x)
# # ==================================================
# def f_b(x):
#     return np.arctan(x)

# def df_b(x):
#     return 1.0 / (1.0 + x**2)


# # ==================================================
# # Hauptprogramm
# # ==================================================
# if __name__ == "__main__":

#     print("======================================")
#     print("Aufgabenblatt 10 – Aufgabe 1")
#     print("Newton-Verfahren (1D)")
#     print("======================================\n")

#     # ---------- Aufgabe 1a ----------
#     print("Aufgabe 1a:")
#     print("f(x) = x^3 + 3^x, Startwert x0 = 0.0")

#     xN, it = newton_1d(f_a, df_a, x0=0.0)
#     print(f"Ergebnis: x_N = {xN:.12f}")
#     print(f"Iterationen: {it}\n")

#     # ---------- Aufgabe 1b ----------
#     print("Aufgabe 1b:")
#     print("f(x) = arctan(x)\n")

#     # Startwert x0 = 2.0
#     print("Startwert x0 = 2.0")
#     try:
#         xN, it = newton_1d(f_b, df_b, x0=2.0)
#         print(f"Ergebnis: x_N = {xN:.12f}")
#         print(f"Iterationen: {it}\n")
#     except Exception as e:
#         print("Newton-Verfahren bricht ab:")
#         print(e, "\n")

#     # Startwert x0 = 1.0
#     print("Startwert x0 = 1.0")
#     xN, it = newton_1d(f_b, df_b, x0=1.0)
#     print(f"Ergebnis: x_N = {xN:.12f}")
#     print(f"Iterationen: {it}\n")



import numpy as np

from linquadref import linquadref
from linquadderivref import linquadderivref
from getJacobian import getJacobian
from gx2dref import gx2dref
from gw2dref import gw2dref
from assemble import assemble
from assignDBC import assignDBC
from Fkt_0 import plot_temperature_trisurf_interp


# =========================================================
# Problem parameters (given in sheet)
# =========================================================
lam = 48.0
c1 = 1e6
c2 = 1e3

tol = 1e-8
itermax = 50


# =========================================================
# Mesh (0-based)
# =========================================================
b = 0.3
h = 0.3
r = 0.1

dx = b / 3.0
dy = h / 3.0

nodes = np.array([
    [0.0, 0.0], [dx, 0.0], [2*dx, 0.0], [b, 0.0],
    [0.0, dy],  [dx, dy],  [2*dx, dy],  [b, dy],
    [0.0, 2*dy],[dx, 2*dy],[2*dx, 2*dy],
    [b - r*np.sin(np.pi/6), h - r*np.cos(np.pi/6)],  # node 12 (idx 11)
    [b, h - r],                                       # node 13 (idx 12)
    [b - r*np.cos(np.pi/6), h - r*np.sin(np.pi/6)],   # node 14 (idx 13)
    [0.0, h], [dx, h], [b/2, h], [b - r, h]           # 15..18 (idx 14..17)
], dtype=float)

elements = np.array([
    [0, 1, 5, 4],
    [1, 2, 6, 5],
    [2, 3, 7, 6],
    [4, 5, 9, 8],
    [5, 6, 10, 9],
    [6, 10, 13, 11],
    [6, 7, 12, 11],
    [8, 9, 15, 14],
    [9, 10, 16, 15],
    [10, 13, 17, 16]
], dtype=int)

nnodes = nodes.shape[0]


# =========================================================
# Nonlinear source
# q(T) = c1 * exp(-c2/T)
# dq/dT = c1 * exp(-c2/T) * (c2/T^2)
# =========================================================
def qdot(T):
    return c1 * np.exp(-c2 / T)

def dqdot_dT(T):
    return c1 * np.exp(-c2 / T) * (c2 / (T**2))


# =========================================================
# IMPORTANT: Fix element orientation so detJ > 0 (at center)
# This is the #1 reason for "too cold" results and Part B "already safe".
# =========================================================
def fix_element_orientation(nodes, elements):
    fixed = elements.copy()
    for eidx in range(fixed.shape[0]):
        ele = fixed[eidx]
        elenodes = nodes[ele, :]
        _, detJ, _ = getJacobian(elenodes, 0.0, 0.0)  # check at element center
        if detJ < 0:
            # reverse order: [0,1,2,3] -> [0,3,2,1]
            fixed[eidx] = ele[[0, 3, 2, 1]]
    return fixed


# =========================================================
# Element routine for Newton:
# 给定一个单元当前的温度 Te，计算这个单元在 Newton 法里对应的 Jacobian 矩阵 Je 和残差向量 re
# residual re(T) = Ke*Te - fe(T)
# jacobian Je(T) = Ke - dfe/dT
# =========================================================
def eval_elem_newton(elenodes, Te, gpx, gpw):
    Ke = np.zeros((4, 4), dtype=float) #单元刚度矩阵（线性的那一部分）
    fe = np.zeros(4, dtype=float)      #非线性体热源向量
    dfe = np.zeros((4, 4), dtype=float)

    ngp = gpw.shape[0]

    for k in range(ngp):
        xi, eta = gpx[k]
        w = gpw[k]

        N = linquadref(xi, eta)                 # (4,)
        dN_ref = linquadderivref(xi, eta)       # (4,2)

        J, detJ, invJ = getJacobian(elenodes, xi, eta)
        if detJ <= 0:
            raise ValueError(f"detJ <= 0 in element. Check element ordering. detJ={detJ}")

        gradN = dN_ref @ invJ                   # (4,2)

        # stiffness part: ∫ lam * gradNi·gradNj
        Ke += lam * (gradN @ gradN.T) * detJ * w

        # nonlinear load: ∫ Ni * q(T)
        Tgp = float(N @ Te)
        q = qdot(Tgp)
        dq = dqdot_dT(Tgp)

        fe += (N * q) * detJ * w
        dfe += (dq * np.outer(N, N)) * detJ * w

    re = Ke @ Te - fe
    Je = Ke - dfe
    return Je, re


# =========================================================
# Solve nonlinear stationary for a given cooling temperature
# 用 Newton方法解整个结构的非线性稳态温度场
# 给定 T^k
# 逐元素算 Ke, fe(T^k), dfe/dT
# 组装全局 F(T^k) = K*T^k - f(T^k)
# 组装 Jacobian J(T^k) = K - df/dT
# 对固定温度节点施加 ΔT=0
# 解 J ΔT = -F
# T^{k+1} = T^k + ΔT
# 重新强制 T = 边界温度
# =========================================================
def solve_for_Tcool(Tcool, elements, verbose=True):

    # initial guess: all 300 except Dirichlet nodes
    T = np.full(nnodes, 300.0, dtype=float)

    bottom_nodes = np.array([0, 1, 2, 3], dtype=int)          # y=0 -> 600K
    notch_nodes  = np.array([11, 12, 13, 17], dtype=int)      # notch boundary -> Tcool

    fixed_nodes = np.unique(np.hstack([bottom_nodes, notch_nodes]))

    T[bottom_nodes] = 600.0
    T[notch_nodes]  = float(Tcool)

    gpx = gx2dref(3)
    gpw = gw2dref(3).flatten()

    # dbc for Newton increment system: ΔT = 0 on fixed nodes
    dbc_delta = np.column_stack([fixed_nodes, np.zeros_like(fixed_nodes, dtype=float)])

    for it in range(itermax):

        Jglob = np.zeros((nnodes, nnodes), dtype=float)
        rglob = np.zeros(nnodes, dtype=float)

        # assemble
        for ele in elements:
            elenodes = nodes[ele, :]
            Te = T[ele]

            Je, re = eval_elem_newton(elenodes, Te, gpx, gpw)

            # IMPORTANT: re must be 1D (4,), not (4,1)
            Jglob, rglob = assemble(Je, re, Jglob, rglob, ele)

        res = np.linalg.norm(rglob, 2)
        if verbose:
            print(f"Newton {it:02d}: ||r||2 = {res:.3e}")

        if res < tol:
            break

        rhs = -rglob.copy()
        Jm, rhsm = assignDBC(Jglob, rhs, dbc_delta)   # enforce ΔT=0 on fixed nodes

        dT = np.linalg.solve(Jm, rhsm)
        T += dT

        # re-enforce actual Dirichlet on T
        T[bottom_nodes] = 600.0
        T[notch_nodes]  = float(Tcool)

    return T


# =========================================================
# Main
# =========================================================
if __name__ == "__main__":

    # Fix element ordering first (still 0-based, for FEM)
    elements_fixed = fix_element_orientation(nodes, elements)

    # -----------------------------------------------------
    # Prepare elements for plotting (1-based!)
    # -----------------------------------------------------
    elements_plot = elements_fixed + 1

    print("=== Part A: nonlinear stationary with Tcool=300K ===")
    T = solve_for_Tcool(300.0, elements_fixed, verbose=True)

    print("\nTemperatures at nodes 15-18:")
    for idx in [14, 15, 16, 17]:
        print(f"T{idx+1} = {T[idx]:.10f} K")

    # ===== Plot for Part A (as in solution) =====
    plot_temperature_trisurf_interp(
        nodes,
        elements_plot,
        T,
        Tmin=300,
        Tmax=600,
        title="Temperaturverteilung stationär nichtlinear"
    )

    print("\n=== Part B: find T* so that max(T on y=h) <= 450K ===")
    top_nodes = np.where(np.abs(nodes[:, 1] - h) < 1e-12)[0]
    Tk = 450.0

    if np.max(T[top_nodes]) <= Tk:
        print("Already safe at Tcool=300K.")
        Tbest = T
        Tcool_best = 300.0
    else:
        Tbest = None
        Tcool_best = None

        for step in range(1, 200):
            Tcool_try = 300.0 - 10.0 * step
            Ttry = solve_for_Tcool(Tcool_try, elements_fixed, verbose=False)

            if np.max(Ttry[top_nodes]) <= Tk:
                Tbest = Ttry
                Tcool_best = Tcool_try
                break

        if Tbest is None:
            raise RuntimeError("Could not find safe Tcool within tried range.")

        print(f"Found T* = {Tcool_best:.1f} K")

        print("Safe node temps 15-18:")
        for idx in [14, 15, 16, 17]:
            print(f"T{idx+1} = {Tbest[idx]:.10f} K")

    # ===== Plot for Part B (as in solution) =====
    plot_temperature_trisurf_interp(
        nodes,
        elements_plot,
        Tbest,
        Tmin=250,
        Tmax=600,
        title=f"Temperaturverteilung stationär nichtlinear (T* = {int(Tcool_best)} K)"
    )
