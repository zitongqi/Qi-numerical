# # ab4_aufgabe1.py
# import numpy as np

# def f(x):
#     """Given function: f(x) = (x/(1+x))^5"""
#     return (x / (1.0 + x))**5

# def fprime_exact(x):
#     """
#     Exact derivative of f(x) = (x/(1+x))^5.
#     Let g(x)=x/(1+x) => g'(x)=1/(1+x)^2
#     f'(x)=5*g(x)^4*g'(x)
#          = 5*(x/(1+x))^4 * 1/(1+x)^2
#          = 5*x^4 / (1+x)^6
#     """
#     return 5.0 * x**4 / (1.0 + x)**6

# # -----------------------------
# # Finite-difference formulas
# # -----------------------------
# def fd_2p_forward(x0, h):
#     """Two-point (forward) formula: (f(x0+h)-f(x0))/h"""
#     return (f(x0 + h) - f(x0)) / h

# def fd_3p_endpoint(x0, h):
#     """Three-point endpoint formula: (-3f0 + 4f1 - f2) / (2h)"""
#     f0 = f(x0)
#     f1 = f(x0 + h)
#     f2 = f(x0 + 2*h)
#     return (-3*f0 + 4*f1 - f2) / (2*h)

# def fd_3p_midpoint(x0, h):
#     """Three-point midpoint formula: (f(x0+h)-f(x0-h))/(2h)"""
#     return (f(x0 + h) - f(x0 - h)) / (2*h)

# def fd_5p_midpoint(x0, h):
#     """Five-point midpoint formula: (f(x0-2h)-8f(x0-h)+8f(x0+h)-f(x0+2h))/(12h)"""
#     fm2 = f(x0 - 2*h)
#     fm1 = f(x0 - h)
#     fp1 = f(x0 + h)
#     fp2 = f(x0 + 2*h)
#     return (fm2 - 8*fm1 + 8*fp1 - fp2) / (12*h)

# def main():
#     x0 = 0.6
#     h = 0.1

#     exact = fprime_exact(x0)

#     methods = [
#         ("2P (forward)", fd_2p_forward),
#         ("3PE (endpoint)", fd_3p_endpoint),
#         ("3PM (midpoint)", fd_3p_midpoint),
#         ("5PM (midpoint)", fd_5p_midpoint),
#     ]

#     print("Aufgabenblatt 4 - Aufgabe 1: Finite-Differenzen-Approximation")
#     print(f"f(x) = (x/(1+x))^5,  x0={x0},  h={h}")
#     print(f"Exact f'(x0) = {exact:.16f}")
#     print()

#     for name, func in methods:
#         approx = func(x0, h)
#         err = abs(exact - approx)
#         print(f"{name:15s}:  approx = {approx:.12f}   abs error = {err:.3e}")

# if __name__ == "__main__":
#     main()


# ab4_aufgabe2.py
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------
# Given function and exact derivative
# -------------------------------------------------
def f(x):
    return (x / (1.0 + x))**5

def fprime_exact(x):
    return 5.0 * x**4 / (1.0 + x)**6


# -------------------------------------------------
# Finite-difference formulas
# -------------------------------------------------
def fd_2p_forward(x0, h):
    return (f(x0 + h) - f(x0)) / h

def fd_3p_endpoint(x0, h):
    return (-3*f(x0) + 4*f(x0 + h) - f(x0 + 2*h)) / (2*h)

def fd_3p_midpoint(x0, h):
    return (f(x0 + h) - f(x0 - h)) / (2*h)

def fd_5p_midpoint(x0, h):
    return (f(x0 - 2*h) - 8*f(x0 - h) + 8*f(x0 + h) - f(x0 + 2*h)) / (12*h)


# -------------------------------------------------
# Main routine for Aufgabe 2
# -------------------------------------------------
def convergence_plot(x0):
    # step sizes
    h_vals = np.logspace(0, -5, 200)

    exact = fprime_exact(x0)

    err_2p  = []
    err_3pe = []
    err_3pm = []
    err_5pm = []

    for h in h_vals:
        err_2p.append(abs(exact - fd_2p_forward(x0, h)))
        err_3pe.append(abs(exact - fd_3p_endpoint(x0, h)))
        err_3pm.append(abs(exact - fd_3p_midpoint(x0, h)))
        err_5pm.append(abs(exact - fd_5p_midpoint(x0, h)))

    err_2p  = np.array(err_2p)
    err_3pe = np.array(err_3pe)
    err_3pm = np.array(err_3pm)
    err_5pm = np.array(err_5pm)

    # reference curves (scaled for visibility)
    h1 = h_vals
    h2 = h_vals**2
    h4 = h_vals**4

    # -------------------------------------------------
    # Plot
    # -------------------------------------------------
    plt.figure(figsize=(7, 5))
    plt.loglog(h_vals, err_2p,  label="2P (O(h))")
    plt.loglog(h_vals, err_3pe, label="3PE (O(h²))")
    plt.loglog(h_vals, err_3pm, label="3PM (O(h²))")
    plt.loglog(h_vals, err_5pm, label="5PM (O(h⁴))")

    plt.loglog(h_vals, h1,  "k--", label="h")
    plt.loglog(h_vals, h2,  "k-.", label="h²")
    plt.loglog(h_vals, h4,  "k:",  label="h⁴")

    plt.xlabel("h")
    plt.ylabel(r"$|f'(x_0) - f'_h(x_0)|$")
    plt.title(f"Konvergenz Finite Differenzen bei x0 = {x0}")
    plt.legend()
    plt.grid(True, which="both", ls=":")

    plt.show()


def main():
    # two evaluation points required by the task
    convergence_plot(0.6)
    convergence_plot(2.0)


if __name__ == "__main__":
    main()

