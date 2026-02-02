
import numpy as np
import time
import matplotlib.pyplot as plt

from FktXIV import solveGauss
from FktXV import solveG
from FktXVI import solveCG


# --------------------------------------------------
# Hilfsfunktion: tridiagonale Matrix
# --------------------------------------------------
def build_matrix(n, phi):
    A = np.zeros((n, n))
    for i in range(n):
        A[i, i] = phi
        if i > 0:
            A[i, i - 1] = -2.0
        if i < n - 1:
            A[i, i + 1] = -2.0
    return A


# ==================================================
# Parameter
# ==================================================
rtol = 1e-8
itermax = 5000
x0 = None   # wird je nach n gesetzt


# ==================================================
# TEIL 1: Einfluss von phi (n = 300)
# ==================================================
print("\n===== Einfluss von phi (n = 300) =====")

n = 300
b = np.ones(n)
x0 = np.zeros(n)

phi_values = [
    10.0, 6.0, 5.1, 5.01, 5.001,
    5.00001, 5.0000001, 5.000000001
]

results_phi = []

for phi in phi_values:
    print(f"\nphi = {phi}")
    A = build_matrix(n, phi)

    # ---------- Gauss ----------
    t0 = time.time()
    try:
        solveGauss(A.copy(), b.copy())
        t_gauss = time.time() - t0
    except Exception:
        t_gauss = np.nan

    # ---------- Gradient ----------
    t0 = time.time()
    _, it_g = solveG(A, b, x0, rtol, itermax)
    t_g = time.time() - t0

    # ---------- Conjugate Gradient ----------
    t0 = time.time()
    _, it_cg = solveCG(A, b, x0, rtol, itermax)
    t_cg = time.time() - t0

    # ---------- NumPy ----------
    t0 = time.time()
    np.linalg.solve(A, b)
    t_np = time.time() - t0

    results_phi.append((phi, it_g, it_cg, t_g, t_cg, t_gauss, t_np))

    print(f"  Gradient : {it_g:4d} it, {t_g:.4f} s")
    print(f"  CG       : {it_cg:4d} it, {t_cg:.4f} s")
    print(f"  Gauss    : {t_gauss:.4f} s")
    print(f"  NumPy    : {t_np:.4f} s")


# ==================================================
# TEIL 2: Einfluss der Systemgröße n (phi = 5.01)
# ==================================================
print("\n===== Einfluss der Systemgröße n (phi = 5.01) =====")

phi = 5.01
n_values = [300, 500, 800, 1200]

results_n = []

for n in n_values:
    print(f"\nn = {n}")
    A = build_matrix(n, phi)
    b = np.ones(n)
    x0 = np.zeros(n)

    # ---------- Gauss ----------
    t0 = time.time()
    try:
        solveGauss(A.copy(), b.copy())
        t_gauss = time.time() - t0
    except Exception:
        t_gauss = np.nan

    # ---------- CG ----------
    t0 = time.time()
    solveCG(A, b, x0, rtol, itermax)
    t_cg = time.time() - t0

    results_n.append((n, t_gauss, t_cg))

    print(f"  Gauss : {t_gauss:.4f} s")
    print(f"  CG    : {t_cg:.4f} s")


# ==================================================
# Plot 1: Iterationen vs phi
# ==================================================
phis = [r[0] for r in results_phi]
iters_g = [r[1] for r in results_phi]
iters_cg = [r[2] for r in results_phi]

plt.figure()
plt.semilogx(phis, iters_g, 'o-', label='Gradient')
plt.semilogx(phis, iters_cg, 's-', label='CG')
plt.xlabel(r'$\phi$')
plt.ylabel('Iterationszahl')
plt.title('Iterationszahl vs. $\phi$ (n = 300)')
plt.legend()
plt.grid(True)
plt.show()


# ==================================================
# Plot 2: Laufzeit vs n
# ==================================================
ns = [r[0] for r in results_n]
t_gauss = [r[1] for r in results_n]
t_cg = [r[2] for r in results_n]

plt.figure()
plt.plot(ns, t_gauss, 'o-', label='Gauss')
plt.plot(ns, t_cg, 's-', label='CG')
plt.xlabel('Systemgröße n')
plt.ylabel('Zeit [s]')
plt.title('Lösungszeit vs. Systemgröße n ($\phi=5.01$)')
plt.legend()
plt.grid(True)
plt.show()
