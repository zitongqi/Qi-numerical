# E1 错误
import numpy as np

from FktXX import evaluate_instat          # Q4 element routine
from FktV import gx2dref                   # Gauss points (n*n,2)
from FktVI import gw2dref                  # Gauss weights (n*n,1)
from FktXVIII import assemble              # global assembly
from assignDBC import assignDBC            # your Dirichlet routine
from quadplot import quadplot              # plotting

# -------------------------------------------------
# geometry
# -------------------------------------------------
r = 0.02
b = 0.3
h = 0.3

nodes = np.array([
    [0, 0],
    [b/3, 0],
    [2*b/3, 0],
    [b, 0],
    [0, h/3],
    [b/3, h/3],
    [2*b/3, h/3],
    [b, h/3],
    [0, 2*h/3],
    [b/3, 2*h/3],
    [2*b/3, 2*h/3],
    [b - r*np.sin(np.pi/6), h - r*np.cos(np.pi/6)],
    [b, h - r],
    [b - r*np.cos(np.pi/6), h - r*np.sin(np.pi/6)],
    [0, h],
    [b/3, h],
    [b/2, h],
    [b - r, h]
], dtype=float)

elements = np.array([
    [1, 2, 6, 5],
    [2, 3, 7, 6],
    [3, 4, 8, 7],
    [5, 6, 10, 9],
    [6, 7, 11, 10],
    [11, 7, 12, 14],
    [7, 8, 13, 12],
    [9, 10, 16, 15],
    [10, 11, 17, 16],
    [17, 11, 14, 18]
], dtype=int) - 1  # 0-based

# -------------------------------------------------
# task parameters
# -------------------------------------------------
Tk = 450.0
T0 = 300.0
ts = 5000
timestep = 500      # dt=500 -> solution expects t*=3500
theta = 0.5
timInt_m = 1        # OST

t = np.arange(0, ts + timestep, timestep)

# -------------------------------------------------
# DBC
# -------------------------------------------------
dbc = np.array([
    [1, 600],
    [2, 600],
    [3, 600],
    [4, 600],
    [12, 300],
    [13, 300],
    [14, 300],
    [18, 300],
], dtype=float)
dbc[:, 0] -= 1  # 0-based

# monitor nodes 15..18 (Matlab) => 14..17 (Python)
monitor = np.array([14, 15, 16, 17], dtype=int)

# -------------------------------------------------
# init
# -------------------------------------------------
ndof = 18
T = np.zeros((ndof, len(t)), dtype=float)
T[:, 0] = T0

gpx = gx2dref(2)
gpw = gw2dref(2).reshape(-1)   # (n*n,)

def solve_one_step(sol_n, sol_nm1, firststep_flag):
    # IMPORTANT: use column vector rhs like Matlab (ndof,1)
    sysmat = np.zeros((ndof, ndof), dtype=float)
    rhs = np.zeros((ndof, 1), dtype=float)

    for e in range(elements.shape[0]):
        elem = elements[e, :]

        elemat, elevec = evaluate_instat(
            nodes[elem, :],
            gpx,
            gpw,
            sol_n[elem],
            sol_nm1[elem],
            timInt_m,
            timestep,
            theta,
            firststep=firststep_flag
        )

        # IMPORTANT: convert elevec (4,1) -> (4,)
        elevec_1d = np.asarray(elevec, dtype=float).reshape(-1)

        # If your assemble expects (4,1) you can also pass elevec directly,
        # but then rhs MUST be (ndof,1). Here we pass (4,) to avoid warnings.
        sysmat, rhs = assemble(elemat, elevec_1d, sysmat, rhs, elem)

    sysmat, rhs = assignDBC(sysmat, rhs, dbc)

    # solve -> return 1D vector
    return np.linalg.solve(sysmat, rhs).reshape(-1)

# -------------------------------------------------
# step 1: boundary acts "from next timestep"
# -------------------------------------------------
T[:, 1] = solve_one_step(T[:, 0], T[:, 0], firststep_flag=0)

t_star = None
j_star = None

if np.max(T[monitor, 1]) > Tk:
    t_star = t[1]
    j_star = 1

# -------------------------------------------------
# time loop
# -------------------------------------------------
for j in range(2, len(t)):
    T[:, j] = solve_one_step(T[:, j-1], T[:, j-2], firststep_flag=1)

    if np.max(T[monitor, j]) > Tk:
        t_star = t[j]
        j_star = j
        break

# -------------------------------------------------
# output
# -------------------------------------------------
if t_star is None:
    print(f"Tk={Tk} K not exceeded up to t={ts} s.")
    print("T15..T18 at t_end:", T[monitor, -1])
    quadplot(nodes, elements, T[:, -1])
else:
    T15, T16, T17, T18 = T[monitor, j_star]
    print(f"First exceedance of Tk={Tk} K at t* = {t_star:.1f} s (step {j_star})")
    print(f"T15 = {T15:.12f}")
    print(f"T16 = {T16:.12f}")
    print(f"T17 = {T17:.12f}")
    print(f"T18 = {T18:.12f}")

    quadplot(nodes, elements, T[:, j_star])














# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.tri as mtri

# from FktXX import evaluate_instat          # 你的 Q4 元素函数
# from FktV import gx2dref                   # 2D Gauss 点
# from FktVI import gw2dref                  # 2D Gauss 权重


# # ============================================================
# # T3（三角形）元素：instationär（CST）
# # ============================================================
# def evaluate_instat_T3(xy, Tn, dt, theta, rho=7800.0, c=452.0, lam=48.0):
#     xy = np.asarray(xy, float).reshape(3, 2)
#     Tn = np.asarray(Tn, float).reshape(3)

#     x1, y1 = xy[0]
#     x2, y2 = xy[1]
#     x3, y3 = xy[2]

#     det = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
#     A = 0.5 * abs(det)
#     if A <= 0:
#         raise ValueError("Triangle area is zero/negative. Check node ordering.")

#     b1 = y2 - y3
#     b2 = y3 - y1
#     b3 = y1 - y2
#     c1 = x3 - x2
#     c2 = x1 - x3
#     c3 = x2 - x1

#     grad = np.array([[b1, c1],
#                      [b2, c2],
#                      [b3, c3]], dtype=float) / (2.0 * A)

#     # consistent mass
#     M = rho * c * A / 12.0 * np.array([[2, 1, 1],
#                                        [1, 2, 1],
#                                        [1, 1, 2]], dtype=float)

#     # stiffness
#     K = lam * A * (grad @ grad.T)

#     Ae = (1.0 / dt) * M + theta * K
#     fe = ((1.0 / dt) * M - (1.0 - theta) * K) @ Tn
#     return Ae, fe


# # ============================================================
# # Dirichlet 施加（强制法）
# # ============================================================
# def apply_dirichlet(A, f, dbc):
#     A = A.copy()
#     f = f.copy()
#     for i, val in dbc:
#         f -= A[:, i] * val
#         A[:, i] = 0.0
#         A[i, :] = 0.0
#         A[i, i] = 1.0
#         f[i] = val
#     return A, f


# # ============================================================
# # 网格：18 节点（按题图公式）
# # ============================================================
# def build_mesh():
#     b = 0.3
#     h = 0.3
#     r = 0.02
#     dx = b / 3.0
#     dy = h / 3.0

#     X = np.zeros((19, 2), dtype=float)  # 1..18 used

#     X[1]  = [0.0, 0.0]
#     X[2]  = [dx, 0.0]
#     X[3]  = [2*dx, 0.0]
#     X[4]  = [3*dx, 0.0]

#     X[5]  = [0.0, dy]
#     X[6]  = [dx, dy]
#     X[7]  = [2*dx, dy]
#     X[8]  = [3*dx, dy]

#     X[9]  = [0.0, 2*dy]
#     X[10] = [dx, 2*dy]
#     X[11] = [2*dx, 2*dy]

#     X[15] = [0.0, 3*dy]
#     X[16] = [dx, 3*dy]

#     X[12] = [b - r*np.sin(np.pi/6.0), h - r*np.cos(np.pi/6.0)]
#     X[13] = [b, h - r]
#     X[14] = [b - r*np.cos(np.pi/6.0), h - r*np.sin(np.pi/6.0)]
#     X[17] = [b/2.0, h]
#     X[18] = [b - r, h]

#     return X


# def build_elements():
#     """
#     如果你之前主程序已经能跑出接近参考值（483/463/441/300），
#     就继续用同一套连通性即可。
#     """
#     elems = []

#     elems.append(("Q4", [1, 2, 6, 5]))    # e1
#     elems.append(("Q4", [2, 3, 7, 6]))    # e2
#     elems.append(("Q4", [3, 4, 8, 7]))    # e3

#     elems.append(("Q4", [5, 6, 10, 9]))   # e4
#     elems.append(("Q4", [6, 7, 11, 10]))  # e5

#     elems.append(("Q4", [9, 10, 16, 15]))   # e8
#     elems.append(("Q4", [10, 11, 17, 16]))  # e9

#     elems.append(("Q4", [7, 8, 13, 11]))     # e7
#     elems.append(("T3", [11, 14, 17]))       # e6
#     elems.append(("T3", [17, 14, 18]))       # e10

#     return elems


# # ============================================================
# # 画 3D 温度图
# # ============================================================
# def plot_3d_temperature(X, elems, T, title):
#     tris = []
#     for etype, conn in elems:
#         if etype == "T3":
#             tris.append([conn[0]-1, conn[1]-1, conn[2]-1])
#         else:
#             n1, n2, n3, n4 = [c-1 for c in conn]
#             tris.append([n1, n2, n3])
#             tris.append([n1, n3, n4])

#     pts = X[1:19, :]  # (18,2)
#     tri = mtri.Triangulation(pts[:, 0], pts[:, 1], np.array(tris, dtype=int))

#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection="3d")
#     ax.plot_trisurf(tri, T, linewidth=0.2)
#     ax.set_xlabel("x")
#     ax.set_ylabel("y")
#     ax.set_zlabel("T [K]")
#     ax.set_title(title)
#     plt.show()


# # ============================================================
# # 主逻辑：找 t*
# # ============================================================
# def main():
#     # -------- 可调参数 --------
#     theta = 0.5         # 题目给
#     dt = 500.0          # 题目给（你也可以改成 250 / 1000 来测试）
#     Tk = 450.0          # 关键温度
#     max_steps = 200     # 防止无限跑（200*500=100000s 足够）
#     gauss_n = 2         # 题目说 2 点/方向

#     # 网格与元素
#     X = build_mesh()
#     elems = build_elements()

#     # Gauss
#     gpx = gx2dref(gauss_n)
#     gpw = gw2dref(gauss_n).flatten()

#     nn = 18
#     # 初值：全域 300K
#     T_old = np.ones(nn, dtype=float) * 300.0

#     # 顶边节点（y=h）
#     top_nodes = [15, 16, 17, 18]  # 题目也用这几个输出
#     top_idx = [n-1 for n in top_nodes]

#     # Dirichlet 边界：
#     bottom_nodes = [1, 2, 3, 4]          # y=0
#     notch_nodes = [12, 13, 14, 18]       # 缺口边界保持 300K（与前半题一致）

#     # 关键：题目说 “Ab dem nächsten Zeitschritt … y=0 wird T=600K”
#     # => step=0 时（t=0）全部 300
#     # => 从 step=1 开始施加 bottom=600
#     def get_dbc(step):
#         dbc = []
#         # notch 永远 300K
#         for n in notch_nodes:
#             dbc.append((n-1, 300.0))
#         # bottom 从下一步开始 600K
#         if step >= 1:
#             for n in bottom_nodes:
#                 dbc.append((n-1, 600.0))
#         return dbc

#     # 初始检查（t=0 顶边一定 <=450）
#     print(f"t=0: max(top)={T_old[top_idx].max():.3f} K")

#     # 时间步进，找第一次超过 Tk 的步
#     t_star = None
#     T_star = None
#     step_star = None

#     for step in range(1, max_steps + 1):
#         A = np.zeros((nn, nn), dtype=float)
#         f = np.zeros(nn, dtype=float)

#         # 组装
#         for etype, conn in elems:
#             idx = [n-1 for n in conn]

#             if etype == "Q4":
#                 xy = X[conn, :]
#                 Tn = T_old[idx]
#                 # eleosol 在 OST 测试里不需要，这里给同维零即可
#                 Tnm1 = np.zeros_like(Tn)

#                 Ae, fe = evaluate_instat(
#                     xy, gpx, gpw, Tn, Tnm1,
#                     timInt_m=1,
#                     timestep=dt,
#                     theta=theta,
#                     firststep=(1 if step == 1 else 0)
#                 )

#                 for a in range(4):
#                     ia = idx[a]
#                     f[ia] += fe[a]
#                     for b in range(4):
#                         ib = idx[b]
#                         A[ia, ib] += Ae[a, b]

#             elif etype == "T3":
#                 xy = X[conn, :]
#                 Tn = T_old[idx]

#                 Ae, fe = evaluate_instat_T3(
#                     xy, Tn, dt=dt, theta=theta
#                 )

#                 for a in range(3):
#                     ia = idx[a]
#                     f[ia] += fe[a]
#                     for b in range(3):
#                         ib = idx[b]
#                         A[ia, ib] += Ae[a, b]
#             else:
#                 raise ValueError("Unknown element type")

#         # Dirichlet
#         dbc = get_dbc(step)
#         A_bc, f_bc = apply_dirichlet(A, f, dbc)

#         # solve
#         T_new = np.linalg.solve(A_bc, f_bc)

#         t = step * dt
#         top_max = T_new[top_idx].max()

#         print(f"step {step:3d}, t={t:8.1f}s, max(top)={top_max:8.3f} K")

#         # 找第一次超过 Tk
#         if top_max > Tk:
#             t_star = t
#             T_star = T_new.copy()
#             step_star = step
#             break

#         T_old = T_new

#     if t_star is None:
#         print(f"\nNo exceedance found up to {max_steps*dt:.1f} s.")
#         return

#     print("\n==================== RESULT ====================")
#     print(f"First exceedance of Tk={Tk}K at step {step_star}, t* = {t_star:.1f} s")
#     print("Temperatures at top nodes:")
#     for n in top_nodes:
#         print(f"  T[{n}] = {T_star[n-1]:.12f} K")

#     # 画 3D
#     plot_3d_temperature(
#         X, elems, T_star,
#         title=f"Temperature field at first exceedance: t* = {t_star:.0f}s (dt={dt:.0f}s, theta={theta})"
#     )


# if __name__ == "__main__":
#     main()
