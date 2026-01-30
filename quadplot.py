# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.tri as mtri

# def quadplot(nodes, elements, sol):
#     """
#     Plot Q4 elements by splitting each quad into two triangles
#     Python translation of quadplot.m

#     Parameters
#     ----------
#     nodes : (N,2) ndarray
#         Node coordinates
#     elements : (Ne,4) ndarray
#         Element connectivity (0-based!)
#     sol : (N,) ndarray
#         Nodal solution values
#     """

#     n_elem = elements.shape[0]

#     # -------------------------------------------------
#     # node coordinates and solution
#     # -------------------------------------------------
#     x = nodes[:, 0]
#     y = nodes[:, 1]
#     z = sol

#     # -------------------------------------------------
#     # build triangle connectivity
#     # each quad -> 2 triangles
#     # -------------------------------------------------
#     T = np.zeros((2 * n_elem, 3), dtype=int)

#     for i in range(n_elem):
#         # Matlab: [elements(i,1:2), elements(i,4)]
#         T[2*i, :]   = [elements[i, 0], elements[i, 1], elements[i, 3]]

#         # Matlab: [elements(i,2:4)]
#         T[2*i+1, :] = [elements[i, 1], elements[i, 2], elements[i, 3]]

#     # -------------------------------------------------
#     # triangulation & plot
#     # -------------------------------------------------
#     triang = mtri.Triangulation(x, y, T)

#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection="3d")
#     ax.plot_trisurf(triang, z, cmap="viridis")

#     ax.set_xlabel("x")
#     ax.set_ylabel("y")
#     ax.set_zlabel("solution")

#     plt.tight_layout()
#     plt.show()


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

def quadplot(nodes, elements, sol):
    x = nodes[:, 0]
    y = nodes[:, 1]
    z = sol

    n_elem = elements.shape[0]
    T = np.zeros((2 * n_elem, 3), dtype=int)

    for i in range(n_elem):
        T[2*i]   = [elements[i, 0], elements[i, 1], elements[i, 3]]
        T[2*i+1] = [elements[i, 1], elements[i, 2], elements[i, 3]]

    triang = mtri.Triangulation(x, y, T)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_trisurf(triang, z, cmap="hot")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("T(x,y,t)")
    ax.set_title("Temperatur zum Zeitpunkt: t = 5000 s, OST (Δt = 500 s)")

    fig.colorbar(surf, ax=ax, shrink=0.6)
    plt.show()
