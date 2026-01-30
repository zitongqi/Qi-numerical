#E1
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

# =============== optional sparse solver (recommended) ===============
try:
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

def nid(i):
    """Convert 1-based node id (as in the sheet) to 0-based index."""
    return i - 1

def nids(*args):
    """Convert multiple 1-based node ids to 0-based indices."""
    return [i - 1 for i in args]


# ============================================================
#  Robust quad preprocessing (IMPORTANT FIX)
# ============================================================

def fix_quad_order_by_angle(nodes, elem):
    """
    Reorder quad nodes into a non-self-intersecting CCW loop
    by sorting nodes around the centroid.
    """
    elem = np.array(elem, dtype=int)
    pts = nodes[elem, :]
    c = pts.mean(axis=0)

    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    order = np.argsort(ang)  # CCW around centroid
    elem2 = elem[order]

    # Ensure CCW by signed area
    x = nodes[elem2, 0]
    y = nodes[elem2, 1]
    area2 = np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    if area2 < 0:
        elem2 = elem2[[0, 3, 2, 1]]

    return elem2

def is_self_intersecting_quad(nodes, elem):
    """
    Check if quad edges self-intersect (bow-tie quad).
    Edges: 0-1,1-2,2-3,3-0
    """
    e = np.array(elem, dtype=int)
    p = nodes[e, :]

    def orient(u, v, w):
        return (v[0]-u[0])*(w[1]-u[1]) - (v[1]-u[1])*(w[0]-u[0])

    def seg_proper_intersect(a, b, c, d):
        o1 = orient(a, b, c)
        o2 = orient(a, b, d)
        o3 = orient(c, d, a)
        o4 = orient(c, d, b)
        return (o1 * o2 < 0) and (o3 * o4 < 0)

    # For a quad, self-intersection typically means (0-1) intersects (2-3) OR (1-2) intersects (3-0)
    return seg_proper_intersect(p[0], p[1], p[2], p[3]) or seg_proper_intersect(p[1], p[2], p[3], p[0])

def preprocess_elements(nodes, elements):
    """
    Apply robust reordering to all quads, and detect invalid (self-intersecting) quads.
    """
    fixed = []
    for idx, e in enumerate(elements):
        e2 = fix_quad_order_by_angle(nodes, e)

        if is_self_intersecting_quad(nodes, e2):
            print("\n=== BAD self-intersecting quad detected ===")
            print("Element index (0-based in your list):", idx)
            print("Original elem:", np.array(e, dtype=int))
            print("Reordered:", e2)
            print("Coords (reordered):\n", nodes[e2])
            raise ValueError("Self-intersecting quad: you must FIX element connectivity near notch (cannot be solved by reordering).")

        fixed.append(e2)

    return np.array(fixed, dtype=int)


# ============================================================
#  Q4 bilinear element on [-1,1]x[-1,1]
# ============================================================

def gauss_1d_n3():
    """3-point Gauss on [-1,1]."""
    a = np.sqrt(3.0 / 5.0)
    xi = np.array([-a, 0.0, a], dtype=float)
    w  = np.array([5.0/9.0, 8.0/9.0, 5.0/9.0], dtype=float)
    return xi, w

def shape_Q4(xi, eta):
    """
    Node order on reference element:
      1: (-1,-1), 2:(+1,-1), 3:(+1,+1), 4:(-1,+1)
    """
    N = np.array([
        0.25*(1-xi)*(1-eta),
        0.25*(1+xi)*(1-eta),
        0.25*(1+xi)*(1+eta),
        0.25*(1-xi)*(1+eta),
    ], dtype=float)

    dNdxi = np.array([
        -0.25*(1-eta),
        +0.25*(1-eta),
        +0.25*(1+eta),
        -0.25*(1+eta),
    ], dtype=float)

    dNdeta = np.array([
        -0.25*(1-xi),
        -0.25*(1+xi),
        +0.25*(1+xi),
        +0.25*(1-xi),
    ], dtype=float)

    return N, dNdxi, dNdeta


# ============================================================
#  Nonlinear source: q(T) = c1 * exp(-c2/T)
#  dq/dT = c1 * exp(-c2/T) * (c2/T^2)
# ============================================================

def qdot(T, c1, c2):
    return c1 * np.exp(-c2 / T)

def dqdot_dT(T, c1, c2):
    return c1 * np.exp(-c2 / T) * (c2 / (T**2))


# ============================================================
#  Assembly: residual r(T) = K*T - f(T)
#            Jacobian J(T) = K - df/dT
# ============================================================

def assemble_global(nodes, elements, T, lam, c1, c2):
    nnode = nodes.shape[0]

    if HAS_SCIPY:
        K  = sp.lil_matrix((nnode, nnode), dtype=float)
        dF = sp.lil_matrix((nnode, nnode), dtype=float)
    else:
        K  = np.zeros((nnode, nnode), dtype=float)
        dF = np.zeros((nnode, nnode), dtype=float)

    F = np.zeros(nnode, dtype=float)

    xi_1d, w_1d = gauss_1d_n3()

    for eidx, conn0 in enumerate(elements):
        # ---- IMPORTANT: reorder to CCW (robust) ----
        conn = fix_quad_order_by_angle(nodes, conn0)

        xe = nodes[conn, 0]
        ye = nodes[conn, 1]
        Te = T[conn]

        Ke  = np.zeros((4, 4), dtype=float)
        Fe  = np.zeros(4, dtype=float)
        dFe = np.zeros((4, 4), dtype=float)

        for i, xi in enumerate(xi_1d):
            for j, eta in enumerate(xi_1d):
                wgt = w_1d[i] * w_1d[j]

                N, dNdxi, dNdeta = shape_Q4(xi, eta)

                # Jacobian mapping
                J11 = np.dot(dNdxi,  xe)
                J12 = np.dot(dNdxi,  ye)
                J21 = np.dot(dNdeta, xe)
                J22 = np.dot(dNdeta, ye)

                detJ = J11*J22 - J12*J21
                if detJ <= 1e-14:
                    print("\n=== detJ failure ===")
                    print("Element index:", eidx)
                    print("Original conn:", np.array(conn0, dtype=int))
                    print("Reordered conn:", conn)
                    print("Coords:\n", nodes[conn])
                    raise ValueError(f"Element has non-positive detJ={detJ}. This usually means the 4 chosen nodes do not form a valid quad.")

                invJT = (1.0/detJ) * np.array([[ J22, -J21],
                                               [-J12,  J11]], dtype=float)

                dN = np.vstack((dNdxi, dNdeta)).T     # (4,2) in (xi,eta)
                gradN = dN @ invJT                    # (4,2) in (x,y)

                Tgp  = np.dot(N, Te)
                qgp  = qdot(Tgp, c1, c2)
                dqgp = dqdot_dT(Tgp, c1, c2)

                Ke  += lam * (gradN @ gradN.T) * detJ * wgt
                Fe  += (N * qgp) * detJ * wgt
                dFe += (dqgp * np.outer(N, N)) * detJ * wgt

        # assemble to global
        for a in range(4):
            A = conn[a]
            F[A] += Fe[a]
            for b in range(4):
                B = conn[b]
                if HAS_SCIPY:
                    K[A, B]  += Ke[a, b]
                    dF[A, B] += dFe[a, b]
                else:
                    K[A, B]  += Ke[a, b]
                    dF[A, B] += dFe[a, b]

    if HAS_SCIPY:
        K = K.tocsr()
        dF = dF.tocsr()
        r = K @ T - F
        J = K - dF
        return J, r
    else:
        r = K @ T - F
        J = K - dF
        return J, r


# ============================================================
#  Dirichlet treatment for Newton system: J dT = -r
# ============================================================

def apply_dirichlet(J, rhs, dirichlet_nodes):
    if HAS_SCIPY:
        J = J.tolil()
        for i in dirichlet_nodes:
            J.rows[i] = [i]
            J.data[i] = [1.0]
            rhs[i] = 0.0
        J = J.tocsr()
        return J, rhs
    else:
        for i in dirichlet_nodes:
            J[i, :] = 0.0
            J[:, i] = 0.0
            J[i, i] = 1.0
            rhs[i] = 0.0
        return J, rhs


# ============================================================
#  Newton solver
# ============================================================

def newton_solve(nodes, elements, lam, c1, c2, dirichlet, T_init=None,
                 tol=1e-8, itmax=50, verbose=True):
    nnode = nodes.shape[0]

    T = np.full(nnode, 300.0, dtype=float)
    if T_init is not None:
        T[:] = T_init

    dir_nodes = np.array(sorted(dirichlet.keys()), dtype=int)
    for i, val in dirichlet.items():
        T[i] = float(val)

    for k in range(itmax):
        J, r = assemble_global(nodes, elements, T, lam, c1, c2)

        norm_r = np.linalg.norm(r, 2)
        if verbose:
            print(f"Newton iter {k:02d}: ||r||2 = {norm_r:.3e}")

        if norm_r < tol:
            return T, k, norm_r

        rhs = -r.copy()
        Jm, rhsm = apply_dirichlet(J, rhs, dir_nodes)

        if HAS_SCIPY:
            dT = spla.spsolve(Jm, rhsm)
        else:
            dT = np.linalg.solve(Jm, rhsm)

        T += dT

        for i, val in dirichlet.items():
            T[i] = float(val)

    return T, itmax, np.linalg.norm(r, 2)


# ============================================================
#  Plot
# ============================================================

def plot_temperature_3d(nodes, elements, T, title="Temperature distribution"):
    # --- triangulate quads ---
    tris = []
    for e in elements:
        a, b, c, d = e
        tris.append([a, b, c])
        tris.append([a, c, d])
    tris = np.array(tris, dtype=int)

    x = nodes[:, 0]
    y = nodes[:, 1]
    z = T

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")

    # --- key: color by temperature ---
    surf = ax.plot_trisurf(
        x, y, z,
        triangles=tris,
        cmap="hot",          # 和答案一致（黑→红→黄）
        vmin=300, vmax=600,  # 明确温度范围（非常重要）
        linewidth=0.2,
        antialiased=True
    )

    # --- colorbar ---
    cbar = fig.colorbar(surf, ax=ax, shrink=0.65, pad=0.1)
    cbar.set_label("T [K]")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("T [K]")
    ax.set_title(title)

    plt.tight_layout()
    plt.show()



# ============================================================
#  Helper: find nodes on y=h
# ============================================================

def nodes_on_y(nodes, y_value, tol=1e-12):
    return np.where(np.abs(nodes[:, 1] - y_value) < tol)[0]


# ============================================================
#  Example mesh (WARNING: your sheet connectivity near notch may differ)
# ============================================================

def build_example_mesh_b_h_r(b=0.3, h=0.3, r=0.1):
    """
    Node numbering exactly matches the provided figure (1..18).
    In code we store nodes as 0-based array: node i -> index (i-1).

    Special node coordinates from the sheet:
      x12 = [b - r sin(pi/6), h - r cos(pi/6)]
      x13 = [b, h - r]
      x14 = [b - r cos(pi/6), h - r sin(pi/6)]
      x17 = [b/2, h]
      x18 = [b - r, h]

    All other nodes lie on structured grid with spacing b/3 and h/3.
    """
    dx = b / 3.0
    dy = h / 3.0

    coords = {}

    # --- structured grid nodes (as in the figure) ---
    # y = 0: nodes 1..4
    coords[1] = (0.0,   0.0)
    coords[2] = (dx,    0.0)
    coords[3] = (2*dx,  0.0)
    coords[4] = (3*dx,  0.0)

    # y = h/3: nodes 5..8
    coords[5] = (0.0,   dy)
    coords[6] = (dx,    dy)
    coords[7] = (2*dx,  dy)
    coords[8] = (3*dx,  dy)

    # y = 2h/3: nodes 9..11 and 13
    coords[9]  = (0.0,   2*dy)
    coords[10] = (dx,    2*dy)
    coords[11] = (2*dx,  2*dy)

    # node 13 is special but lies at (b, h-r) which equals (b, 2h/3) here if r=0.1,h=0.3
    coords[13] = (b, h - r)

    # y = h: nodes 15,16,17,18 (17,18 special)
    coords[15] = (0.0, h)
    coords[16] = (dx,  h)

    # --- special nodes (from sheet) ---
    coords[12] = (b - r*np.sin(np.pi/6.0), h - r*np.cos(np.pi/6.0))
    coords[14] = (b - r*np.cos(np.pi/6.0), h - r*np.sin(np.pi/6.0))
    coords[17] = (b/2.0, h)
    coords[18] = (b - r, h)

    # build nodes array [0..17] corresponding to node IDs [1..18]
    nodes = np.zeros((18, 2), dtype=float)
    for nid in range(1, 19):
        if nid not in coords:
            raise ValueError(f"Missing coordinate for node {nid}. Check numbering vs figure.")
        nodes[nid-1, :] = coords[nid]

    return nodes


def example_elements_connectivity():
    """
    Element connectivity written in 1-based node numbering
    (exactly as in the sheet).
    Internally converted to 0-based indices.
    """
    elems_1based = [
        [1, 2, 6, 5],        # e1
        [2, 3, 7, 6],        # e2
        [3, 4, 8, 7],        # e3
        [5, 6, 10, 9],       # e4
        [6, 7, 11, 10],      # e5
        [7, 11, 14, 12],     # e6
        [7, 8, 13, 12],      # e7
        [9, 10, 16, 15],     # e8
        [10, 11, 17, 16],    # e9
        [11, 14, 18, 17],    # e10
    ]

    # convert to 0-based for computation
    elems_0based = [nids(*e) for e in elems_1based]
    return np.array(elems_0based, dtype=int)



def plot_mesh_2d(nodes, elements, title="Mesh connectivity (1-based)"):
    plt.figure(figsize=(6,6))

    # draw elements
    for eidx, e in enumerate(elements, start=1):  # 元素从 1 开始编号
        e = np.array(e, dtype=int)
        pts = nodes[e, :]
        pts2 = np.vstack([pts, pts[0]])
        plt.plot(pts2[:,0], pts2[:,1], "-k", linewidth=1)

        c = pts.mean(axis=0)
        plt.text(c[0], c[1], f"e{eidx}", color="red", fontsize=10)

    # draw node ids (1-based!)
    for i, (x, y) in enumerate(nodes, start=1):
        plt.text(x, y, str(i), color="blue", fontsize=9)

    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.grid(True)
    plt.show()


def element_min_detJ(nodes, elem):
    xi_1d, _ = gauss_1d_n3()
    conn = fix_quad_order_by_angle(nodes, elem)

    xe = nodes[conn,0]
    ye = nodes[conn,1]

    min_detJ = 1e100
    for xi in xi_1d:
        for eta in xi_1d:
            _, dNdxi, dNdeta = shape_Q4(xi, eta)
            J11 = np.dot(dNdxi,  xe)
            J12 = np.dot(dNdxi,  ye)
            J21 = np.dot(dNdeta, xe)
            J22 = np.dot(dNdeta, ye)
            detJ = J11*J22 - J12*J21
            min_detJ = min(min_detJ, detJ)
    return min_detJ

def print_detJ_report(nodes, elements):
    report = []
    for i, e in enumerate(elements):
        dj = element_min_detJ(nodes, e)
        report.append((i, dj, np.array(e, dtype=int)))

    report.sort(key=lambda x: x[1])

    print("\n=== detJ report (smallest first) ===")
    for i, dj, e in report:
        print(f"elem {i:02d}: min detJ = {dj:.3e}, conn = {e}")

# ============================================================
#  Run
# ============================================================

def main():
    b = 0.3
    h = 0.3
    r = 0.1

    lam = 48.0
    c1  = 1e6
    c2  = 1e3

    nodes = build_example_mesh_b_h_r(b=b, h=h, r=r)
    elements = example_elements_connectivity()

    # ======================================================
    # 🔴 STEP 1: 画 2D 网格（最重要）
    # ======================================================
    plot_mesh_2d(nodes, elements, title="Original mesh connectivity (before preprocess)")

    # ======================================================
    # 🔴 STEP 2: detJ 报告（预处理前）
    # ======================================================
    print_detJ_report(nodes, elements)

    # ======================================================
    # 🔴 STEP 3: 预处理（CCW + bow-tie 检测）
    # ======================================================
    elements = preprocess_elements(nodes, elements)

    # ======================================================
    # 🔴 STEP 4: 再画一次网格（确认）
    # ======================================================
    plot_mesh_2d(nodes, elements, title="Mesh connectivity AFTER preprocess")

    print_detJ_report(nodes, elements)

    # ======================================================
    # Dirichlet BC
    # ======================================================
    bottom_nodes = nodes_on_y(nodes, 0.0, tol=1e-12)
    bottom_nodes = nids(1, 2, 3, 4)          # y = 0
    notch_nodes  = nids(12, 13, 14, 18)      # cooling boundary


    def solve_for_Tcool(Tcool, verbose=False):
        dirichlet = {int(i): 600.0 for i in bottom_nodes}
        for i in notch_nodes:
            dirichlet[int(i)] = float(Tcool)

        Tsol, iters, res = newton_solve(
            nodes, elements, lam, c1, c2,
            dirichlet=dirichlet,
            tol=1e-8, itmax=50, verbose=verbose
        )
        return Tsol

    # ======================================================
    # Part A
    # ======================================================
    T = solve_for_Tcool(300.0, verbose=True)

    for nid_1 in [15, 16, 17, 18]:
        print(f"T{nid_1} = {T[nid_1-1]:.10f} K")

    plot_temperature_3d(nodes, elements, T,
        title="Stationary nonlinear temperature (Tcool=300K)")

    # ======================================================
    # Part B
    # ======================================================
    top_nodes = nodes_on_y(nodes, h, tol=1e-12)
    Tk = 450.0

    if np.max(T[top_nodes]) > Tk:
        Tbest = None
        Tcool_best = None
        for trial in range(1, 200):
            Tcool_trial = 300.0 - 10.0 * trial
            Ttrial = solve_for_Tcool(Tcool_trial, verbose=False)
            if np.max(Ttrial[top_nodes]) <= Tk:
                Tbest = Ttrial
                Tcool_best = Tcool_trial
                break

        if Tbest is None:
            raise RuntimeError("Could not find Tcool within the trial range.")

        print(f"Found T* = {Tcool_best:.1f} K so that max(T on y=h) <= {Tk} K")
    else:
        print(f"Already safe: max(T on y=h) <= {Tk} K with Tcool=300 K")
        Tbest = T
        Tcool_best = 300.0

    # ---- output node temperatures using 1-based numbering ----
    for i in [15, 16, 17, 18]:
        print(f"(safe) T{i} = {Tbest[nid(i)]:.10f} K")

    plot_temperature_3d(
        nodes, elements, Tbest,
        title=f"Safe configuration (Tcool={Tcool_best:.0f}K)"
    )


if __name__ == "__main__":
    main()
