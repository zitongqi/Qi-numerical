# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# def quadplot(nodes, elements, sol):
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection="3d")

#     colors = [
#         "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
#         "#9467bd", "#8c564b", "#17becf", "#bcbd22"
#     ]
#     c = 0

#     def xyz(nid):
#         x, y = nodes[nid - 1]
#         z = sol[nid - 1]
#         return (x, y, z)

#     # ===============================
#     # 🔴 答案指定的固定三角化方式
#     # ===============================
#     for elem in elements:
#         n1, n2, n3, n4 = elem

#         tri1 = [xyz(n1), xyz(n2), xyz(n4)]
#         tri2 = [xyz(n2), xyz(n3), xyz(n4)]


#         for tri in (tri1, tri2):
#             poly = Poly3DCollection(
#                 [tri],
#                 facecolor=colors[c % len(colors)],
#                 edgecolor="k",
#                 linewidth=1.2,
#                 alpha=0.95
#             )
#             ax.add_collection3d(poly)
#             c += 1

#     # 坐标轴设置（与答案一致）
#     ax.set_xlim(-1, 1)
#     ax.set_ylim(-1, 1)
#     ax.set_zlim(0, 2.5)

#     ax.set_xticks([-1, 0, 1])
#     ax.set_yticks([-1, 0, 1])
#     ax.set_zticks([0, 0.5, 1, 1.5, 2, 2.5])

#     ax.set_xlabel("x")
#     ax.set_ylabel("y")
#     ax.set_zlabel("f(x,y)")
#     ax.set_title("3D Quadplot (nodes + elements + solution)")

#     # 视角：不影响拼接，只是展示
#     ax.view_init(elev=20, azim=45)

#     plt.show()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import Normalize
from matplotlib import cm


# -------------------------------------------------
# helper: subdivide one triangle with barycentric interpolation
# -------------------------------------------------
def _refine_triangle(coords, values, nsub):
    """
    Subdivide a triangle and interpolate nodal values.

    coords : (3,2) array
        Triangle node coordinates
    values : (3,) array
        Nodal values (e.g. temperature)
    nsub : int
        Subdivision level
    """
    pts = []
    vals = []

    for i in range(nsub + 1):
        for j in range(nsub + 1 - i):
            k = nsub - i - j

            l1 = i / nsub
            l2 = j / nsub
            l3 = k / nsub

            x = l1 * coords[0] + l2 * coords[1] + l3 * coords[2]
            t = l1 * values[0] + l2 * values[1] + l3 * values[2]

            pts.append(x)
            vals.append(t)

    return np.array(pts), np.array(vals)


# -------------------------------------------------
# main plot function
# -------------------------------------------------
def plot_temperature_trisurf_interp(
    nodes,
    elements,
    T,
    Tmin,
    Tmax,
    nsub=5,
    title="Temperature distribution (interpolated)",
    cmap="hot",
    view=(25, -120),
):
    """
    Matlab-style interpolated FEM surface plot.
    Interpolation is done INSIDE EACH TRIANGLE.

    Parameters
    ----------
    nodes : (nnode,2) ndarray
        Node coordinates
    elements : (nelem,4) ndarray
        Q4 elements (1-based indexing)
    T : (nnode,) ndarray
        Nodal temperature values
    Tmin, Tmax : float
        Colorbar limits
    nsub : int
        Triangle subdivision level (>=2, typical: 5~8)
    """

    nodes = np.asarray(nodes, dtype=float)
    elements = np.asarray(elements, dtype=int)
    T = np.asarray(T, dtype=float)

    X_all = []
    Y_all = []
    Z_all = []
    tris_all = []

    idx = 0

    # -------------------------------------------------
    # loop over elements
    # -------------------------------------------------
    for elem in elements:
        n1, n2, n3, n4 = elem - 1  # to 0-based

        # Q4 -> two triangles
        tri_conn = [
            (n1, n2, n3),
            (n1, n3, n4),
        ]

        for (i, j, k) in tri_conn:
            coords = nodes[[i, j, k], :]
            values = T[[i, j, k]]

            pts, vals = _refine_triangle(coords, values, nsub)

            X_all.extend(pts[:, 0])
            Y_all.extend(pts[:, 1])
            Z_all.extend(vals)

            # local triangulation of refined triangle
            local_tri = mtri.Triangulation(
                pts[:, 0], pts[:, 1]
            ).triangles

            tris_all.extend(local_tri + idx)
            idx += pts.shape[0]

    X_all = np.array(X_all)
    Y_all = np.array(Y_all)
    Z_all = np.array(Z_all)
    tris_all = np.array(tris_all)

    tri = mtri.Triangulation(X_all, Y_all, tris_all)

    # -------------------------------------------------
    # plot
    # -------------------------------------------------
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    norm = Normalize(vmin=Tmin, vmax=Tmax)
    cmap = cm.get_cmap(cmap)

    surf = ax.plot_trisurf(
        tri,
        Z_all,
        cmap=cmap,
        norm=norm,
        edgecolor="none",
        linewidth=0.0,
        antialiased=True,
    )

    cbar = fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label("T [K]")
    cbar.set_ticks(np.linspace(Tmin, Tmax, 7))

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("T(x,y)")
    ax.set_title(title)
    ax.view_init(elev=view[0], azim=view[1])

    plt.tight_layout()
    plt.show()
