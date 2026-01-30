#E1
# import numpy as np
# import matplotlib.pyplot as plt

# def lineintersection(P1, P2, y_h=2.0):
#     """
#     返回过 P1,P2 的直线与水平线 y=y_h 的交点 x 坐标
#     P1, P2: (x,y) tuple or array-like
#     """
#     x1, y1 = P1
#     x2, y2 = P2

#     # 斜率（这里正是数值问题会出现的地方：y2-y1 可能由于舍入变成 0）
#     m = (y2 - y1) / (x2 - x1)

#     if m == 0.0:
#         return np.nan   # 表示“数值上失败”

#     # y = y1 + m (x - x1)  =>  x = x1 + (y_h - y1)/m
#     return x1 + (y_h - y1) / m


# def main():
#     # 1) 生成 delta：10^-20 到 10^5，logspace 是最合适的“覆盖多个数量级”的分布
#     deltas = np.logspace(-20, 5, 400)

#     x_ex = 1.0  # 解析真值

#     x_num = np.empty_like(deltas)
#     for i, d in enumerate(deltas):
#         P1 = (0.0, 1.0)
#         P2 = (float(d), 1.0 + float(d))
#         x_num[i] = lineintersection(P1, P2, y_h=2.0)

#     # 2) 误差
#     err = np.abs(x_ex - x_num)

#     # 3) 过滤掉 inf/nan（小 delta 时可能出现）
#     mask = np.isfinite(err) & (err > 0)
#     deltas_plot = deltas[mask]
#     err_plot = err[mask]

#     # 4) 画图：双对数
#     plt.figure()
#     plt.loglog(deltas_plot, err_plot)
#     plt.xlabel(r'$\delta$')
#     plt.ylabel(r'$|x_{ex} - x_{num}|$')
#     plt.title('Absolute error of intersection x-coordinate (log-log)')
#     plt.grid(True, which='both')
#     plt.show()

#     # 可选：打印最小/最大误差出现在哪些delta附近
#     if len(err_plot) > 0:
#         idx_max = np.argmax(err_plot)
#         idx_min = np.argmin(err_plot)
#         print("max error =", err_plot[idx_max], "at delta =", deltas_plot[idx_max])
#         print("min error =", err_plot[idx_min], "at delta =", deltas_plot[idx_min])

# if __name__ == "__main__":
#     main()


#E2
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# 真实函数及其导数（用于对比）
# =========================================================

def f_exact(x):
    return (x / (1.0 + x))**5

def df_exact(x):
    return 5 * x**4 / (1.0 + x)**6


# =========================================================
# Lagrange 基函数 L_i(x)
# =========================================================

def LagrangeBasis(x, i, x_nodes):
    """
    L_i(x) = prod_{k!=i} (x - x_k)/(x_i - x_k)
    x 可以是标量或 numpy 数组
    """
    L = 1.0
    xi = x_nodes[i]

    for k, xk in enumerate(x_nodes):
        if k != i:
            L *= (x - xk) / (xi - xk)

    return L


# =========================================================
# Lagrange 基函数导数 L_i'(x)
# =========================================================

def LagrangeDerivBasis(x, i, x_nodes):
    """
    L_i'(x) = sum_{m!=i} [ 1/(x_i-x_m) * prod_{k!=i,m} (x-x_k)/(x_i-x_k) ]
    """
    dL = 0.0
    xi = x_nodes[i]

    for m, xm in enumerate(x_nodes):
        if m == i:
            continue

        term = 1.0 / (xi - xm)

        for k, xk in enumerate(x_nodes):
            if k != i and k != m:
                term *= (x - xk) / (xi - xk)

        dL += term

    return dL


# =========================================================
# Lagrange 插值多项式 P_n(x)
# =========================================================

def LagrangePolynomial(x, x_nodes, f_nodes):
    P = 0.0
    for i in range(len(x_nodes)):
        P += f_nodes[i] * LagrangeBasis(x, i, x_nodes)
    return P


# =========================================================
# Lagrange 插值多项式导数 P_n'(x)
# =========================================================

def LagrangeDerivPolynomial(x, x_nodes, f_nodes):
    dP = 0.0
    for i in range(len(x_nodes)):
        dP += f_nodes[i] * LagrangeDerivBasis(x, i, x_nodes)
    return dP


# =========================================================
# 主程序：完成 a) b) c)
# =========================================================

def main():

    x_eval = 0.6  # 要求计算的点
    x_plot = np.linspace(0.0, 4.0, 400)

    # -----------------------------
    # a) 一阶插值（用 x=0,1）
    # -----------------------------
    x_nodes_1 = np.array([0.0, 1.0])
    f_nodes_1 = np.array([0.0, 0.03125])

    f_L1 = LagrangePolynomial(x_eval, x_nodes_1, f_nodes_1)
    df_L1 = LagrangeDerivPolynomial(x_eval, x_nodes_1, f_nodes_1)

    print("a) Grad 1:")
    print(f"f_L1(0.6)  = {f_L1:.8f}")
    print(f"f'_L1(0.6) = {df_L1:.8f}")
    print()

    # -----------------------------
    # b) 四阶插值（用给定 5 个点）
    # -----------------------------
    x_nodes_4 = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    f_nodes_4 = np.array([
    0.0,
    0.03125,
    0.131687242798,
    0.2373046875,
    0.32768
    ])

    f_L4 = LagrangePolynomial(x_eval, x_nodes_4, f_nodes_4)
    df_L4 = LagrangeDerivPolynomial(x_eval, x_nodes_4, f_nodes_4)

    print("b) Grad 4:")
    print(f"f_L4(0.6)  = {f_L4:.8f}")
    print(f"f'_L4(0.6) = {df_L4:.8f}")
    print()

    # -----------------------------
    # c) 八十阶插值（81 个等距点）
    # -----------------------------
    x_nodes_80 = np.linspace(0.0, 4.0, 81)
    f_nodes_80 = f_exact(x_nodes_80)

    f_L80 = LagrangePolynomial(x_eval, x_nodes_80, f_nodes_80)
    df_L80 = LagrangeDerivPolynomial(x_eval, x_nodes_80, f_nodes_80)

    print("c) Grad 80:")
    print(f"f_L80(0.6)  = {f_L80:.8f}")
    print(f"f'_L80(0.6) = {df_L80:.8f}")
    print()

    print("Exakt:")
    print(f"f(0.6)  = {f_exact(x_eval):.8f}")
    print(f"f'(0.6) = {df_exact(x_eval):.8f}")

    # =====================================================
    # 作图（以 Grad 4 为例，和题目示例一致）
    # =====================================================

    f_plot_L4 = LagrangePolynomial(x_plot, x_nodes_4, f_nodes_4)
    df_plot_L4 = LagrangeDerivPolynomial(x_plot, x_nodes_4, f_nodes_4)

    plt.figure(figsize=(8, 5))
    plt.plot(x_plot, f_plot_L4, label="Polynom P4")
    plt.plot(x_plot, df_plot_L4, label="erste Ableitung P4")
    plt.plot(x_plot, f_exact(x_plot), "k--", label="exakte Funktion f(x)")
    plt.plot(x_plot, df_exact(x_plot), "k-.", label="exakte Ableitung df/dx")
    plt.scatter(x_nodes_4, f_nodes_4, color="black", zorder=5, label="Stützstellen")

    plt.xlabel("x")
    plt.ylabel("Wert")
    plt.title("Lagrangesches Interpolationspolynom vom Grad 4")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
