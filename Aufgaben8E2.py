# import numpy as np

# from evaluate_instat import evaluate_instat
# from gx2dref import gx2dref
# from gw2dref import gw2dref
# from assemble import assemble
# from assignDBC import assignDBC
# from quadplot import quadplot

# # -------------------------------------------------
# # parameters / geometry
# # -------------------------------------------------
# r = 0.02
# b = 0.3
# h = 0.3

# nodes = np.array([
#     [0, 0],
#     [b/3, 0],
#     [2*b/3, 0],
#     [b, 0],
#     [0, h/3],
#     [b/3, h/3],
#     [2*b/3, h/3],
#     [b, h/3],
#     [0, 2*h/3],
#     [b/3, 2*h/3],
#     [2*b/3, 2*h/3],
#     [b - r*np.sin(np.pi/6), h - r*np.cos(np.pi/6)],
#     [b, h - r],
#     [b - r*np.cos(np.pi/6), h - r*np.sin(np.pi/6)],
#     [0, h],
#     [b/3, h],
#     [b/2, h],
#     [b - r, h]
# ], dtype=float)

# elements = np.array([
#     [1, 2, 6, 5],
#     [2, 3, 7, 6],
#     [3, 4, 8, 7],
#     [5, 6, 10, 9],
#     [6, 7, 11, 10],
#     [11, 7, 12, 14],
#     [7, 8, 13, 12],
#     [9, 10, 16, 15],
#     [10, 11, 17, 16],
#     [17, 11, 14, 18]
# ], dtype=int) - 1  # 0-based

# # -------------------------------------------------
# # DBC:
# #  - y = 0: nodes 1..4 -> 600 K (from next time step)
# #  - keep the other fixed nodes as in sheet (12,13,14,18) -> 300 K
# # -------------------------------------------------
# dbc = np.array([
#     [1, 600],
#     [2, 600],
#     [3, 600],
#     [4, 600],
#     [12, 300],
#     [13, 300],
#     [14, 300],
#     [18, 300]
# ], dtype=float)
# dbc[:, 0] -= 1  # 0-based

# # -------------------------------------------------
# # time integration (as in solution: dt=500 -> t*=3500 corresponds to step 7)
# # -------------------------------------------------
# Tk = 450.0
# ts = 5000
# timestep = 500
# theta = 0.5
# timInt_m = 1  # OST

# t = np.arange(0, ts + timestep, timestep)

# # -------------------------------------------------
# # initialization
# # -------------------------------------------------
# ndof = 18
# T = np.zeros((ndof, len(t)))
# T[:, 0] = 300.0  # T0 everywhere

# # nodes on y = h you want to monitor: (15,16,17,18) in Matlab => (14..17) in Python
# monitor = np.array([14, 15, 16, 17], dtype=int)

# # -------------------------------------------------
# # helper: assemble + solve one time step
# # -------------------------------------------------
# def solve_step(j, sol_n, sol_nm1, firststep_flag):
#     sysmat = np.zeros((ndof, ndof))
#     rhs = np.zeros(ndof)

#     for i in range(elements.shape[0]):
#         elem = elements[i, :]
#         elemat, elevec = evaluate_instat(
#             nodes[elem, :],
#             gx2dref(2),
#             gw2dref(2),
#             sol_n[elem],
#             sol_nm1[elem],
#             timInt_m,
#             timestep,
#             theta,
#             firststep=firststep_flag
#         )
#         sysmat, rhs = assemble(elemat, elevec, sysmat, rhs, elem)

#     sysmat, rhs = assignDBC(sysmat, rhs, dbc)
#     return np.linalg.solve(sysmat, rhs)

# # -------------------------------------------------
# # first step: compute T(:,1) -> T at t = 500 s
# # (this is "ab dem nächsten Zeitschritt" -> boundary y=0=600 acts from here)
# # -------------------------------------------------
# T[:, 1] = solve_step(
#     j=1,
#     sol_n=T[:, 0],
#     sol_nm1=T[:, 0],
#     firststep_flag=0
# )

# # check threshold from step 1 onward
# t_star = None
# j_star = None

# if np.max(T[monitor, 1]) > Tk:
#     t_star = t[1]
#     j_star = 1

# # -------------------------------------------------
# # time loop
# # -------------------------------------------------
# for j in range(2, len(t)):
#     T[:, j] = solve_step(
#         j=j,
#         sol_n=T[:, j-1],
#         sol_nm1=T[:, j-2],
#         firststep_flag=1
#     )

#     if np.max(T[monitor, j]) > Tk:
#         t_star = t[j]
#         j_star = j
#         break

# # -------------------------------------------------
# # output
# # -------------------------------------------------
# if t_star is None:
#     print(f"No exceedance of Tk={Tk} K up to t={ts} s.")
#     print("Final monitored temps (T15..T18) =", T[monitor, -1])
#     quadplot(nodes, elements, T[:, -1])
# else:
#     # Matlab nodes 15..18 => Python indices 14..17
#     T15, T16, T17, T18 = T[monitor, j_star]

#     print(f"First exceedance of Tk={Tk} K at t* = {t_star:.1f} s (step j = {j_star})")
#     print(f"T15 = {T15:.12f}")
#     print(f"T16 = {T16:.12f}")
#     print(f"T17 = {T17:.12f}")
#     print(f"T18 = {T18:.12f}")

#     # plot temperature field at t*
#     quadplot(nodes, elements, T[:, j_star])


import numpy as np

from evaluate_instat import evaluate_instat
from gx2dref import gx2dref
from gw2dref import gw2dref
from assemble import assemble
from assignDBC import assignDBC
from Fkt_0 import plot_temperature_trisurf_interp


# =================================================
# 0) Geometry / Mesh (given by sheet)
# =================================================
r = 0.02
b = 0.3
h = 0.3

# nodes (18 x 2)
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

# elements (10 x 4), stored 0-based in Python
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
], dtype=int) - 1

ndof = nodes.shape[0]  # 18 DOF for scalar temperature


# =================================================
# 1) Problem setup (tasks)
# =================================================
T0 = 300.0         # initial temperature everywhere at t=0
T_hot = 600.0      # y=0 boundary temperature, applied from next time step
T_cold = 300.0     # notch boundary (always)
Tk = 450.0         # critical temperature at y=h
ts = 5000.0        # simulation end time

# monitor nodes on y=h: Matlab (15,16,17,18) -> Python (14,15,16,17)
monitor = np.array([14, 15, 16, 17], dtype=int)

# Node sets for Dirichlet boundaries (0-based indices)
nodes_y0 = np.array([0, 1, 2, 3], dtype=int)          # nodes 1..4 -> y=0
nodes_notch = np.array([11, 12, 13, 17], dtype=int)   # nodes 12,13,14,18 -> notch


def dbc_for_time_step(j: int) -> np.ndarray:
    """
    Task 3 (step loading):
      - At t=0 (j=0): do NOT apply the 600K on y=0.
      - From the next time step (j>=1): apply y=0 => 600K.
      - Notch boundary always 300K.
    Returns dbc array with rows [node_index, value] (node_index is 0-based).
    """
    rows = []
    # notch is always fixed at 300K
    for n in nodes_notch:
        rows.append([n, T_cold])

    # step load: y=0 becomes 600K only from j>=1
    if j >= 1:
        for n in nodes_y0:
            rows.append([n, T_hot])

    return np.array(rows, dtype=float)


# =================================================
# 2) One-step solver (assemble + apply DBC + solve)
# =================================================
gpx = gx2dref(2)
gpw = gw2dref(2)

def solve_step(sol_n: np.ndarray, sol_nm1: np.ndarray,
               timestep: float, timInt_m: int, theta: float,
               j: int, firststep_flag: int) -> np.ndarray:
    """
    Solve one time step producing T^{n+1}.
    sol_n   : temperature at time level n  (global size ndof)
    sol_nm1 : temperature at time level n-1(global size ndof)
    j       : resulting time index (so we can apply correct time-dependent DBC)
    """
    sysmat = np.zeros((ndof, ndof), dtype=float)
    rhs = np.zeros(ndof, dtype=float)

    for e in range(elements.shape[0]):
        elem = elements[e, :]  # 4 global node indices of this element

        elemat, elevec = evaluate_instat(
            nodes[elem, :],
            gpx,
            gpw,
            sol_n[elem],        # elesol  (T^n at element nodes)
            sol_nm1[elem],      # eleosol (T^{n-1} at element nodes)
            timInt_m,
            timestep,
            theta,
            firststep=firststep_flag
        )
        sysmat, rhs = assemble(elemat, elevec, sysmat, rhs, elem)

    # apply time-dependent Dirichlet BC for this step index j
    dbc = dbc_for_time_step(j)
    sysmat, rhs = assignDBC(sysmat, rhs, dbc)

    return np.linalg.solve(sysmat, rhs)


# =================================================
# 3) Run one case: (Δt, method) -> find first exceedance time t*
# =================================================
def run_case(timestep: float, timInt_m: int, theta: float,
             ts: float, Tk: float, monitor: np.ndarray):
    """
    Returns:
      t_star (float or None), j_star (int or None), T_star (ndof array or None)
    """
    t = np.arange(0.0, ts + timestep, timestep)
    nt = len(t)

    # initial condition at t=0
    Tn0 = np.full(ndof, T0, dtype=float)

    # If nt == 1, no stepping possible
    if nt < 2:
        return None, None, None

    # --- first step: compute T at j=1 (t = Δt)
    # for start-up we set sol_nm1 = sol_n = T0
    T1 = solve_step(
        sol_n=Tn0,
        sol_nm1=Tn0,
        timestep=timestep,
        timInt_m=timInt_m,
        theta=theta,
        j=1,
        firststep_flag=0
    )

    # check threshold at first computed step
    if np.max(T1[monitor]) > Tk:
        return t[1], 1, T1

    # --- general time loop: keep only last two states
    Tnm1 = Tn0     # T^{n-1}
    Tn = T1        # T^{n}

    for j in range(2, nt):
        Tnp1 = solve_step(
            sol_n=Tn,
            sol_nm1=Tnm1,
            timestep=timestep,
            timInt_m=timInt_m,
            theta=theta,
            j=j,
            firststep_flag=1
        )

        if np.max(Tnp1[monitor]) > Tk:
            return t[j], j, Tnp1

        Tnm1, Tn = Tn, Tnp1

    return None, None, None


# =================================================
# 4) Task 1+2+3+4: automatic study + report + plot one selected case
# =================================================
# Methods to test (Fkt. IX..XII)
# NOTE: theta is only used for OST; for other methods it is ignored in your evaluate_instat,
#       but we pass a float anyway.
methods = [
    ("OST",  1, 0.5),
    ("AB2",  2, 0.0),
    ("AM3",  3, 0.0),
    ("BDF2", 4, 0.0),
]

# time steps to test
dt_list = [100, 250, 500, 1000]

# choose one case to create the 3D plot at t*
PLOT_METHOD = "OST"
PLOT_DT = 500

results = []

print("=================================================")
print("Parameter study: varying Δt and time integrator")
print(f"Critical temperature Tk = {Tk} K monitored at nodes 15..18 (y=h)")
print("Step loading: y=0 becomes 600K from the NEXT time step (j>=1)")
print("=================================================\n")

for method_name, timInt_m, theta in methods:
    for dt in dt_list:
        t_star, j_star, T_star = run_case(
            timestep=float(dt),
            timInt_m=timInt_m,
            theta=float(theta),
            ts=float(ts),
            Tk=float(Tk),
            monitor=monitor
        )

        results.append((method_name, dt, t_star, j_star))

        t_str = "never" if t_star is None else f"{t_star:.0f}"
        print(f"[{method_name:4s}] Δt = {dt:4d} s  ->  first exceed t* = {t_str:>5s} s")

print("\n=================================================")
print("Summary table")
print("Method   Δt[s]   t*[s]")
print("----------------------")
for method_name, dt, t_star, _ in results:
    t_str = "never" if t_star is None else f"{t_star:.0f}"
    print(f"{method_name:6s} {dt:6d} {t_str:>8s}")
print("=================================================\n")

# -------------------------------------------------
# Task 4: For the selected case, print node temps & plot 3D field at t*
# -------------------------------------------------
sel = None
for method_name, dt, t_star, j_star in results:
    if method_name == PLOT_METHOD and dt == PLOT_DT:
        sel = (method_name, dt, t_star, j_star)
        break

if sel is None:
    raise RuntimeError("Selected plot case not found in results list.")

method_name, dt, t_star, j_star = sel

print(f"Selected case for Task 4 plot: {method_name}, Δt = {dt} s")

if t_star is None:
    print(f"No exceedance of Tk={Tk} K up to t={ts} s for this case.")
else:
    # rerun once to get T at t*
    # (we rerun because we did not store all T histories in the study loop)
    timInt_m = next(m[1] for m in methods if m[0] == method_name)
    theta = next(m[2] for m in methods if m[0] == method_name)

    t_star2, j_star2, T_star = run_case(
        timestep=float(dt),
        timInt_m=int(timInt_m),
        theta=float(theta),
        ts=float(ts),
        Tk=float(Tk),
        monitor=monitor
    )

    # safety
    if t_star2 is None:
        print("Unexpected: plot case rerun found no exceedance.")
    else:
        # Matlab nodes 15..18 => Python indices 14..17
        T15, T16, T17, T18 = T_star[monitor]
        print(f"First exceedance of Tk={Tk} K at t* = {t_star2:.0f} s (step j={j_star2})")
        print(f"T15 = {T15:.12f}")
        print(f"T16 = {T16:.12f}")
        print(f"T17 = {T17:.12f}")
        print(f"T18 = {T18:.12f}")

        # 3D plot using Fkt_0 function (expects elements 1-based)
        plot_temperature_trisurf_interp(
            nodes,
            elements + 1,
            T_star,
            Tmin=300.0,
            Tmax=600.0,
            nsub=6,
            title=f"Temperature at t*={t_star2:.0f}s, {method_name}, Δt={dt}s",
            cmap="hot",
            view=(25, -120),
        )
