# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.tri as mtri

# from FktXVII import evaluate_stat
# from FktXVIII import assemble
# from FktXIX import assignDBC

# # ----------------------------
# # Gauss points/weights on [-1,1]
# # ----------------------------
# def gx2dref(n: int) -> np.ndarray:
#     """2D Gauss points (tensor product), shape (n*n, 2): (xi, eta)."""
#     x, _ = np.polynomial.legendre.leggauss(n)
#     pts = []
#     for eta in x:
#         for xi in x:
#             pts.append([xi, eta])
#     return np.array(pts, dtype=float)

# def gw2dref(n: int) -> np.ndarray:
#     """2D Gauss weights (tensor product), shape (n*n,)."""
#     _, w = np.polynomial.legendre.leggauss(n)
#     W = []
#     for j in range(n):
#         for i in range(n):
#             W.append(w[i] * w[j])
#     return np.array(W, dtype=float)


# # ----------------------------
# # helper: build mesh for the sheet
# # ----------------------------
# def build_mesh(b=0.3, h=0.3, r=0.02):
#     """
#     Returns:
#       nodes    : (18,2) array, node coords
#       elements : (10,4) int array, 1-based node indices (quad elements)
#       dbc      : (ndbc,2) array, [node_index, value]
#     """
#     nodes = np.zeros((18, 2), dtype=float)

#     # structured grid spacing b/3, h/3
#     dx = b / 3.0
#     dy = h / 3.0

#     # nodes 1..11 (structured)
#     nodes[0]  = [0.0, 0.0]      # 1
#     nodes[1]  = [dx, 0.0]       # 2
#     nodes[2]  = [2*dx, 0.0]     # 3
#     nodes[3]  = [b, 0.0]        # 4

#     nodes[4]  = [0.0, dy]       # 5
#     nodes[5]  = [dx, dy]        # 6
#     nodes[6]  = [2*dx, dy]      # 7
#     nodes[7]  = [b, dy]         # 8

#     nodes[8]  = [0.0, 2*dy]     # 9
#     nodes[9]  = [dx, 2*dy]      # 10
#     nodes[10] = [2*dx, 2*dy]    # 11

#     # special nodes 12,13,14,17,18 from sheet
#     nodes[11] = [b - r*np.sin(np.pi/6), h - r*np.cos(np.pi/6)]  # 12
#     nodes[12] = [b,               h - r]                        # 13
#     nodes[13] = [b - r*np.cos(np.pi/6), h - r*np.sin(np.pi/6)]  # 14

#     nodes[14] = [0.0, h]         # 15
#     nodes[15] = [dx, h]          # 16
#     nodes[16] = [b/2.0, h]       # 17
#     nodes[17] = [b - r, h]       # 18

#     # 10 quad elements (1-based indices), following the mesh picture
#     elements = np.array([
#         [ 1,  2,  6,  5],   # e1
#         [ 2,  3,  7,  6],   # e2
#         [ 3,  4,  8,  7],   # e3

#         [ 5,  6, 10,  9],   # e4
#         [ 6,  7, 11, 10],   # e5

#         [ 7, 12, 14, 11],   # e6  ✅ 按图：7-12-14-11
#         [ 7,  8, 13, 12],   # e7  ✅ 按图：7-8-13-12

#         [ 9, 10, 16, 15],   # e8
#         [10, 11, 17, 16],   # e9
#         [11, 14, 18, 17],   # e10 ✅ 按图：11-14-18-17
#     ], dtype=int)



#     # Dirichlet BCs:
#     # Γ1D: y=0 => nodes 1..4 are 600 K
#     # Γ2D: cooling notch boundary => nodes 12,13,14,18 are 300 K
#     dbc = np.array([
#         [ 1, 600.0],
#         [ 2, 600.0],
#         [ 3, 600.0],
#         [ 4, 600.0],
#         [12, 300.0],
#         [13, 300.0],
#         [14, 300.0],
#         [18, 300.0],
#     ], dtype=float)

#     return nodes, elements, dbc


# # ----------------------------
# # helper: 3D plot (split each quad into 2 triangles)
# # ----------------------------
# def plot_temperature_3d(nodes, elements, T):
#     import matplotlib.pyplot as plt
#     import matplotlib.tri as mtri
#     from matplotlib import cm
#     from matplotlib.colors import Normalize

#     x = nodes[:, 0]
#     y = nodes[:, 1]

#     # split each quad into two triangles
#     tris = []
#     for ele in elements:
#         n1, n2, n3, n4 = ele - 1
#         tris.append([n1, n2, n3])
#         tris.append([n1, n3, n4])

#     tri = mtri.Triangulation(x, y, np.array(tris))

#     # --- IMPORTANT PART ---
#     Tmin = 300.0
#     Tmax = 600.0
#     norm = Normalize(vmin=Tmin, vmax=Tmax)
#     cmap = cm.hot   # 和题目最接近（黑-红-黄）

#     fig = plt.figure(figsize=(8, 6))
#     ax = fig.add_subplot(111, projection="3d")

#     surf = ax.plot_trisurf(
#         tri,
#         T,
#         cmap=cmap,
#         norm=norm,
#         linewidth=0.3,
#         edgecolor="k",
#         antialiased=True
#     )

#     # colorbar with physical temperature values
#     cbar = fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1)
#     cbar.set_label("T [K]")
#     cbar.set_ticks([300, 350, 400, 450, 500, 550, 600])

#     ax.set_xlabel("x")
#     ax.set_ylabel("y")
#     ax.set_zlabel("T(x,y)")

#     # view angle similar to solution sheet
#     ax.view_init(elev=25, azim=-120)

#     ax.set_title("Stationäre Temperaturverteilung (FEM)")
#     plt.tight_layout()
#     plt.show()



# def main():
#     # problem constants
#     b = 0.3
#     h = 0.3
#     r = 0.02
#     lam = 48.0

#     # mesh & BC
#     nodes, elements, dbc = build_mesh(b=b, h=h, r=r)

#     # Gauss quadrature required by sheet: n=2 points per direction
#     gpx = gx2dref(2)
#     gpw = gw2dref(2)

#     # global system
#     N = nodes.shape[0]
#     A = np.zeros((N, N), dtype=float)
#     f = np.zeros(N, dtype=float)

#     # loop over elements -> evaluate -> assemble
#     for e in range(elements.shape[0]):
#         ele = elements[e]                # 1-based
#         elenodes = nodes[ele - 1, :]     # (4,2)

#         # Fkt XVII: element matrix/vector
#         A_e, f_e = evaluate_stat(elenodes, gpx, gpw, lam=lam)

#         # Fkt XVIII: assembly
#         A, f = assemble(A_e, f_e, A, f, ele)

#     # Fkt XIX: Dirichlet BC
#     A, f = assignDBC(A, f, dbc)

#     # solve
#     T = np.linalg.solve(A, f)

#     # print requested node temperatures
#     for nid in [15, 16, 17, 18]:
#         print(f"T{nid} = {T[nid-1]:.12f} K")

#     # 3D plot
#     plot_temperature_3d(nodes, elements, T)



# if __name__ == "__main__":
#     main()


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from FktXVII import evaluate_stat
from FktXVIII import assemble
from FktXIX import assignDBC


# ============================================================
# Gauss points / weights (2D tensor product)
# ============================================================
def gx2dref(n):
    x, _ = np.polynomial.legendre.leggauss(n)
    pts = []
    for eta in x:
        for xi in x:
            pts.append([xi, eta])
    return np.array(pts, dtype=float)


def gw2dref(n):
    _, w = np.polynomial.legendre.leggauss(n)
    W = []
    for j in range(n):
        for i in range(n):
            W.append(w[i] * w[j])
    return np.array(W, dtype=float)


# ============================================================
# Build mesh
# ============================================================
def build_mesh(b=0.3, h=0.3, r=0.02):
    nodes = np.zeros((18, 2))

    dx = b / 3.0
    dy = h / 3.0

    # structured grid
    nodes[0]  = [0.0, 0.0]
    nodes[1]  = [dx, 0.0]
    nodes[2]  = [2*dx, 0.0]
    nodes[3]  = [b, 0.0]

    nodes[4]  = [0.0, dy]
    nodes[5]  = [dx, dy]
    nodes[6]  = [2*dx, dy]
    nodes[7]  = [b, dy]

    nodes[8]  = [0.0, 2*dy]
    nodes[9]  = [dx, 2*dy]
    nodes[10] = [2*dx, 2*dy]

    # notch geometry
    nodes[11] = [b - r*np.sin(np.pi/6), h - r*np.cos(np.pi/6)]  # 12
    nodes[12] = [b,               h - r]                        # 13
    nodes[13] = [b - r*np.cos(np.pi/6), h - r*np.sin(np.pi/6)]  # 14

    nodes[14] = [0.0, h]        # 15
    nodes[15] = [dx, h]         # 16
    nodes[16] = [b/2.0, h]      # 17
    nodes[17] = [b - r, h]      # 18

    elements = np.array([
        [ 1,  2,  6,  5],
        [ 2,  3,  7,  6],
        [ 3,  4,  8,  7],
        [ 5,  6, 10,  9],
        [ 6,  7, 11, 10],
        [ 7, 12, 14, 11],
        [ 7,  8, 13, 12],
        [ 9, 10, 16, 15],
        [10, 11, 17, 16],
        [11, 14, 18, 17],
    ], dtype=int)

    # Dirichlet BCs
    dbc = np.array([
        [ 1, 600.0],
        [ 2, 600.0],
        [ 3, 600.0],
        [ 4, 600.0],
        [12, 300.0],
        [13, 300.0],
        [14, 300.0],
        [18, 300.0],
    ], dtype=float)

    return nodes, elements, dbc


# ============================================================
# Update only notch nodes
# ============================================================
def update_notch_nodes(nodes, b, h, r):
    nodes = nodes.copy()
    nodes[11] = [b - r*np.sin(np.pi/6), h - r*np.cos(np.pi/6)]
    nodes[12] = [b, h - r]
    nodes[13] = [b - r*np.cos(np.pi/6), h - r*np.sin(np.pi/6)]
    nodes[17] = [b - r, h]
    return nodes


# ============================================================
# Solve FEM system
# ============================================================
def solve_temperature(nodes, elements, dbc, lam):
    gpx = gx2dref(2)
    gpw = gw2dref(2)

    N = nodes.shape[0]
    A = np.zeros((N, N))
    f = np.zeros(N)

    for ele in elements:
        elenodes = nodes[ele - 1]
        A_e, f_e = evaluate_stat(elenodes, gpx, gpw, lam=lam)
        A, f = assemble(A_e, f_e, A, f, ele)

    A, f = assignDBC(A, f, dbc)
    T = np.linalg.solve(A, f)
    return T


# ============================================================
# 3D plot
# ============================================================
def plot_temperature_3d(nodes, elements, T):
    from matplotlib.colors import Normalize
    from matplotlib import cm

    x = nodes[:, 0]
    y = nodes[:, 1]

    tris = []
    for ele in elements:
        n1, n2, n3, n4 = ele - 1
        tris.append([n1, n2, n3])
        tris.append([n1, n3, n4])

    tri = mtri.Triangulation(x, y, np.array(tris))

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_trisurf(
        tri, T,
        cmap=cm.hot,
        norm=Normalize(vmin=300, vmax=600),
        linewidth=0.3,
        edgecolor="k",
        antialiased=True
    )

    cbar = fig.colorbar(surf, ax=ax, shrink=0.6)
    cbar.set_label("T [K]")
    cbar.set_ticks([300, 350, 400, 450, 500, 550, 600])

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("T(x,y)")
    ax.view_init(elev=25, azim=-120)
    ax.set_title("Stationäre Temperaturverteilung (FEM)")

    plt.tight_layout()
    plt.show()


# ============================================================
# MAIN PROGRAM
# ============================================================
def main():
    b = 0.3
    h = 0.3
    lam = 48.0
    Tk = 450.0

    r = 0.02
    dr = 0.01

    nodes0, elements, dbc = build_mesh(b, h, r)

    step = 0
    while True:
        step += 1

        nodes = update_notch_nodes(nodes0, b, h, r)
        T = solve_temperature(nodes, elements, dbc, lam)

        Tmax_top = np.max(T[[14, 15, 16, 17]])

        print(
            f"step {step:2d} | r = {r:.2f} m | "
            f"T15={T[14]:.2f}, T16={T[15]:.2f}, "
            f"T17={T[16]:.2f}, T18={T[17]:.2f}"
        )

        if Tmax_top <= Tk:
            print(f"\n✅ Minimale sichere Ausnehmung: r* = {r:.2f} m\n")
            plot_temperature_3d(nodes, elements, T)
            break

        r += dr


if __name__ == "__main__":
    main()
