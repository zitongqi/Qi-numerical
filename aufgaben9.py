import numpy as np
import time

from FktXIV import solveGauss
from FktXV import solveG
from FktXVI import solveCG


# --------------------------------------------------
# Hilfsfunktion: tridiagonale Matrix erzeugen
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
# TEIL 1: Einfluss von phi (n = 300)
# ==================================================
print("\n===== Einfluss von phi (n = 300) =====")

n = 300
phi_values = [
    10.0, 6.0, 5.1, 5.01, 5.001,
    5.00001, 5.0000001, 5.000000001
]

rtol = 1e-8
itermax = 5000

b = np.ones(n)
x0 = np.zeros(n)

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
    x = x0.copy()
    r = b - A @ x
    it = 0
    while np.linalg.norm(r) > rtol and it < itermax:
        Ar = A @ r
        alpha = (r @ r) / (r @ Ar)
        x = x + alpha * r
        r = r - alpha * Ar
        it += 1
    t_g = time.time() - t0
    it_g = it

    # ---------- Conjugate Gradient ----------
    t0 = time.time()
    x = x0.copy()
    r = b - A @ x
    p = r.copy()
    it = 0
    while np.linalg.norm(r) > rtol and it < itermax:
        Ap = A @ p
        alpha = (r @ r) / (p @ Ap)
        x = x + alpha * p
        r_new = r - alpha * Ap
        beta = (r_new @ r_new) / (r @ r)
        p = r_new + beta * p
        r = r_new
        it += 1
    t_cg = time.time() - t0
    it_cg = it

    # ---------- NumPy ----------
    t0 = time.time()
    np.linalg.solve(A, b)
    t_np = time.time() - t0

    results_phi.append((phi, it_g, t_g, it_cg, t_cg, t_gauss, t_np))

    print(f"  Gradient : {it_g:3d} it, {t_g:.4f} s")
    print(f"  CG       : {it_cg:3d} it, {t_cg:.4f} s")
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

    # ---------- Conjugate Gradient ----------
    t0 = time.time()
    x = x0.copy()
    r = b - A @ x
    p = r.copy()
    it = 0
    while np.linalg.norm(r) > rtol and it < itermax:
        Ap = A @ p
        alpha = (r @ r) / (p @ Ap)
        x = x + alpha * p
        r_new = r - alpha * Ap
        beta = (r_new @ r_new) / (r @ r)
        p = r_new + beta * p
        r = r_new
        it += 1
    t_cg = time.time() - t0

    results_n.append((n, t_gauss, t_cg))

    print(f"  Gauss : {t_gauss:.4f} s")
    print(f"  CG    : {t_cg:.4f} s")


# ==================================================
# Zusammenfassung für Analyse
# ==================================================
print("\n===== Zusammenfassung: Skalierung =====")
for n, tg, tcg in results_n:
    print(f"n={n:4d} | Gauss: {tg:.4f}s | CG: {tcg:.4f}s")

# ==================================================
# Plot 1: Iterationszahl vs phi (Gradient vs CG)
# Entspricht PPT Seite 8
# ==================================================
import matplotlib.pyplot as plt

phis = [r[0] for r in results_phi]
iters_grad = [r[1] for r in results_phi]
iters_cg = [r[3] for r in results_phi]

plt.figure()
plt.semilogx(phis, iters_grad, 'o-', label='Gradient')
plt.semilogx(phis, iters_cg, 's-', label='CG')

plt.xlabel(r'$\phi$')
plt.ylabel('Iterationszahl')
plt.title('Iterationszahl vs. $\phi$ (n = 300)')
plt.legend()
plt.grid(True)

plt.show()

# ==================================================
# Plot 2: Lösungszeit vs n (Gauss vs CG)
# Entspricht PPT Seite 12
# ==================================================
ns = [r[0] for r in results_n]
time_gauss = [r[1] for r in results_n]
time_cg = [r[2] for r in results_n]

plt.figure()
plt.plot(ns, time_gauss, 'o-', label='Gauss')
plt.plot(ns, time_cg, 's-', label='CG')

plt.xlabel('Systemgröße n')
plt.ylabel('Zeit [s]')
plt.title('Lösungszeit vs. Systemgröße n ($\phi = 5.01$)')
plt.legend()
plt.grid(True)

plt.show()
