
import numpy as np

from evaluate_instat import evaluate_instat
from gx2dref import gx2dref
from gw2dref import gw2dref
from assemble import assemble
from assignDBC import assignDBC
from Fkt_0 import plot_temperature_trisurf_interp
# -------------------------------------------------
# parameters
# -------------------------------------------------
r = 0.02
b = 0.3
h = 0.3

# -------------------------------------------------
# nodes (18 x 2)
# -------------------------------------------------
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

# -------------------------------------------------
# elements (10 x 4), 0-based
# -------------------------------------------------
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

# -------------------------------------------------
# Dirichlet boundary conditions (0-based)
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
dbc[:, 0] -= 1

# -------------------------------------------------
# time parameters (题目要求)
# -------------------------------------------------
ts = 5000
timestep = 500          # Δt = 500 s
theta = 0.5             # OST
timInt_m = 1            # OST method

t = np.arange(0, ts + timestep, timestep)

# -------------------------------------------------
# initialization
# -------------------------------------------------
ndof = 18
T = np.zeros((ndof, len(t)))
T[:, 0] = 300.0         # initial condition

# =================================================
# first time step
# =================================================
sysmat = np.zeros((ndof, ndof))
rhs = np.zeros(ndof)

for i in range(elements.shape[0]):
    elem = elements[i, :]
    elemat, elevec = evaluate_instat(
        nodes[elem, :],
        gx2dref(2),
        gw2dref(2),
        T[elem, 0],
        T[elem, 0],
        timInt_m,
        timestep,
        theta,
        firststep=0
    )
    sysmat, rhs = assemble(elemat, elevec, sysmat, rhs, elem)

sysmat, rhs = assignDBC(sysmat, rhs, dbc)
T[:, 1] = np.linalg.solve(sysmat, rhs)

# =================================================
# time loop
# =================================================
for j in range(2, len(t)):
    sysmat = np.zeros((ndof, ndof))
    rhs = np.zeros(ndof)

    for i in range(elements.shape[0]):
        elem = elements[i, :]
        elemat, elevec = evaluate_instat(
            nodes[elem, :],
            gx2dref(2),
            gw2dref(2),
            T[elem, j-1],
            T[elem, j-2],
            timInt_m,
            timestep,
            theta,
            firststep=1
        )
        sysmat, rhs = assemble(elemat, elevec, sysmat, rhs, elem)

    sysmat, rhs = assignDBC(sysmat, rhs, dbc)
    T[:, j] = np.linalg.solve(sysmat, rhs)

# =================================================
# results at t = 5000 s
# =================================================
print("Temperatures at t = 5000 s (OST, Δt = 500 s):")
print(f"T15 = {T[14, -1]:.6f}")
print(f"T16 = {T[15, -1]:.6f}")
print(f"T17 = {T[16, -1]:.6f}")
print(f"T18 = {T[17, -1]:.6f}")

# plot temperature field
plot_temperature_trisurf_interp(
    nodes,
    elements + 1,          # ⚠️ 关键：转回 1-based
    T[:, -1],
    Tmin=300,
    Tmax=600,
    nsub=6,
    title="Temperature at t = 5000 s (OST, Δt = 500 s)",
    cmap="hot",
    view=(25, -120),
)
