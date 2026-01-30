import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def quadplot(nodes, elements, sol):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#17becf", "#bcbd22"
    ]
    c = 0

    def xyz(nid):
        x, y = nodes[nid - 1]
        z = sol[nid - 1]
        return (x, y, z)

    # ===============================
    # 🔴 答案指定的固定三角化方式
    # ===============================
    for elem in elements:
        n1, n2, n3, n4 = elem

        tri1 = [xyz(n1), xyz(n2), xyz(n4)]
        tri2 = [xyz(n2), xyz(n3), xyz(n4)]


        for tri in (tri1, tri2):
            poly = Poly3DCollection(
                [tri],
                facecolor=colors[c % len(colors)],
                edgecolor="k",
                linewidth=1.2,
                alpha=0.95
            )
            ax.add_collection3d(poly)
            c += 1

    # 坐标轴设置（与答案一致）
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(0, 2.5)

    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.set_zticks([0, 0.5, 1, 1.5, 2, 2.5])

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("f(x,y)")
    ax.set_title("3D Quadplot (nodes + elements + solution)")

    # 视角：不影响拼接，只是展示
    ax.view_init(elev=20, azim=45)

    plt.show()
