import numpy as np

from evaluate_instat import evaluate_instat
from gx2dref import gx2dref
from gw2dref import gw2dref
from assemble import assemble
from assignDBC import assignDBC
from quadplot import quadplot

# -------------------------------------------------
# parameters / geometry
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
# DBC:
#  - y = 0: nodes 1..4 -> 600 K (from next time step)
#  - keep the other fixed nodes as in sheet (12,13,14,18) -> 300 K
# -------------------------------------------------
dbc = np.array([
    [1, 600],
    [2, 600],
    [3, 600],
    [4, 600],
    [12, 300],
    [13, 300],
    [14, 300],
    [18, 300]
], dtype=float)
dbc[:, 0] -= 1  # 0-based

# -------------------------------------------------
# time integration (as in solution: dt=500 -> t*=3500 corresponds to step 7)
# -------------------------------------------------
Tk = 450.0
ts = 5000
timestep = 500
theta = 0.5
timInt_m = 1  # OST

t = np.arange(0, ts + timestep, timestep)

# -------------------------------------------------
# initialization
# -------------------------------------------------
ndof = 18
T = np.zeros((ndof, len(t)))
T[:, 0] = 300.0  # T0 everywhere

# nodes on y = h you want to monitor: (15,16,17,18) in Matlab => (14..17) in Python
monitor = np.array([14, 15, 16, 17], dtype=int)

# -------------------------------------------------
# helper: assemble + solve one time step
# -------------------------------------------------
def solve_step(j, sol_n, sol_nm1, firststep_flag):
    sysmat = np.zeros((ndof, ndof))
    rhs = np.zeros(ndof)

    for i in range(elements.shape[0]):
        elem = elements[i, :]
        elemat, elevec = evaluate_instat(
            nodes[elem, :],
            gx2dref(2),
            gw2dref(2),
            sol_n[elem],
            sol_nm1[elem],
            timInt_m,
            timestep,
            theta,
            firststep=firststep_flag
        )
        sysmat, rhs = assemble(elemat, elevec, sysmat, rhs, elem)

    sysmat, rhs = assignDBC(sysmat, rhs, dbc)
    return np.linalg.solve(sysmat, rhs)

# -------------------------------------------------
# first step: compute T(:,1) -> T at t = 500 s
# (this is "ab dem nächsten Zeitschritt" -> boundary y=0=600 acts from here)
# -------------------------------------------------
T[:, 1] = solve_step(
    j=1,
    sol_n=T[:, 0],
    sol_nm1=T[:, 0],
    firststep_flag=0
)

# check threshold from step 1 onward
t_star = None
j_star = None

if np.max(T[monitor, 1]) > Tk:
    t_star = t[1]
    j_star = 1

# -------------------------------------------------
# time loop
# -------------------------------------------------
for j in range(2, len(t)):
    T[:, j] = solve_step(
        j=j,
        sol_n=T[:, j-1],
        sol_nm1=T[:, j-2],
        firststep_flag=1
    )

    if np.max(T[monitor, j]) > Tk:
        t_star = t[j]
        j_star = j
        break

# -------------------------------------------------
# output
# -------------------------------------------------
if t_star is None:
    print(f"No exceedance of Tk={Tk} K up to t={ts} s.")
    print("Final monitored temps (T15..T18) =", T[monitor, -1])
    quadplot(nodes, elements, T[:, -1])
else:
    # Matlab nodes 15..18 => Python indices 14..17
    T15, T16, T17, T18 = T[monitor, j_star]

    print(f"First exceedance of Tk={Tk} K at t* = {t_star:.1f} s (step j = {j_star})")
    print(f"T15 = {T15:.12f}")
    print(f"T16 = {T16:.12f}")
    print(f"T17 = {T17:.12f}")
    print(f"T18 = {T18:.12f}")

    # plot temperature field at t*
    quadplot(nodes, elements, T[:, j_star])
