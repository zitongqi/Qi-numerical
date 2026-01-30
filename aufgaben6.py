#E1
# import numpy as np
# import matplotlib.pyplot as plt

# # -----------------------------
# # Problem definition
# # -----------------------------
# def f(t, phi):
#     return t**2 * np.exp(-5*t) - 6*phi

# def phi_exact(t):
#     return np.exp(-5*t)*(t**2 - 2*t + 2) - 2*np.exp(-6*t)

# # -----------------------------
# # Time integration methods
# # -----------------------------
# def euler_explicit(dt, T=2.0):
#     t = np.arange(0, T+dt, dt)
#     phi = np.zeros_like(t)
#     for n in range(len(t)-1):
#         phi[n+1] = phi[n] + dt*f(t[n], phi[n])
#     return t, phi

# def euler_implicit(dt, T=2.0):
#     t = np.arange(0, T+dt, dt)
#     phi = np.zeros_like(t)
#     for n in range(len(t)-1):
#         tn1 = t[n+1]
#         phi[n+1] = (phi[n] + dt*tn1**2*np.exp(-5*tn1)) / (1 + 6*dt)
#     return t, phi

# def trapezoidal(dt, T=2.0):
#     t = np.arange(0, T+dt, dt)
#     phi = np.zeros_like(t)
#     for n in range(len(t)-1):
#         tn = t[n]
#         tn1 = t[n+1]
#         fn = tn**2*np.exp(-5*tn)
#         fn1 = tn1**2*np.exp(-5*tn1)
#         phi[n+1] = ((1-3*dt)*phi[n] + 0.5*dt*(fn + fn1)) / (1+3*dt)
#     return t, phi

# # -----------------------------
# # Plot results
# # -----------------------------
# for dt in [0.1, 0.2, 0.4]:
#     t_ex = np.linspace(0, 2, 400)
#     phi_ex = phi_exact(t_ex)

#     t1, phi1 = euler_explicit(dt)
#     t2, phi2 = euler_implicit(dt)
#     t3, phi3 = trapezoidal(dt)

#     plt.figure(figsize=(7,4))
#     plt.plot(t1, phi1, '-o', label='Euler Expl.')
#     plt.plot(t3, phi3, '-o', label='Trapez')
#     plt.plot(t2, phi2, '-o', label='Euler Impl.')
#     plt.plot(t_ex, phi_ex, 'k--', linewidth=2, label='exakt')

#     plt.title(f"Lösungen mit Zeitschrittlänge: {dt}")
#     plt.xlabel("t")
#     plt.ylabel(r"$\phi(t)$")
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()



#E2
import numpy as np
import matplotlib.pyplot as plt
from FktIX import OST
from FktX import AB2
from FktXI import AM3
from FktXII import BDF2


# -------------------------------------------------
# Problem definition (from Aufgabe 1)
# -------------------------------------------------
def B_func(t):
    return -6.0

def C_func(t):
    return t**2 * np.exp(-5*t)

def phi_exact(t):
    return np.exp(-5*t)*(t**2 - 2*t + 2) - 2*np.exp(-6*t)

M = 1.0
t_end = 2.0
phi0 = 0.0

# -------------------------------------------------
# Time stepping driver
# -------------------------------------------------
def solve(method, dt, theta=None):
    t = np.arange(0, t_end + dt, dt)
    N = len(t)

    phi = np.zeros(N)
    phi[0] = phi0

    # --- First step for multistep methods: Trapezregel ---
    if method in ["AB2", "AM3", "BDF2"]:
        LHS, RHS = OST(
            theta=0.5,
            timestep=dt,
            M=M,
            B=[B_func(t[0]), B_func(t[1])],
            C=[C_func(t[0]), C_func(t[1])],
            sol=phi[0]
        )
        phi[1] = RHS / LHS
        start = 1
    else:
        start = 0

    # --- Main time loop ---
    for n in range(start, N-1):
        if method == "OST":
            LHS, RHS = OST(
                theta, dt, M,
                [B_func(t[n]), B_func(t[n+1])],
                [C_func(t[n]), C_func(t[n+1])],
                phi[n]
            )

        elif method == "AB2":
            LHS, RHS = AB2(
                dt, M,
                [B_func(t[n]), B_func(t[n-1])],
                [C_func(t[n]), C_func(t[n-1])],
                [phi[n], phi[n-1]]
            )

        elif method == "AM3":
            LHS, RHS = AM3(
                dt, M,
                [B_func(t[n+1]), B_func(t[n]), B_func(t[n-1])],
                [C_func(t[n+1]), C_func(t[n]), C_func(t[n-1])],
                [phi[n], phi[n-1]]
            )

        elif method == "BDF2":
            LHS, RHS = BDF2(
                dt, M,
                B_func(t[n+1]),
                C_func(t[n+1]),
                [phi[n], phi[n-1]]
            )

        phi[n+1] = RHS / LHS

    return t, phi

# -------------------------------------------------
# Run simulations and plot
# -------------------------------------------------
for dt in [0.1, 0.2, 0.4]:
    t_ex = np.linspace(0, 2, 400)
    phi_ex = phi_exact(t_ex)

    plt.figure(figsize=(7,4))

    # Einschritt-θ
    for theta, label in zip([0.0, 0.5, 1.0],
                            ["Euler expl.", "Trapez", "Euler impl."]):
        t, phi = solve("OST", dt, theta)
        plt.plot(t, phi, marker='o', label=label)

    # Mehrschrittverfahren
    for method in ["AB2", "AM3", "BDF2"]:
        t, phi = solve(method, dt)
        plt.plot(t, phi, marker='o', label=method)

    plt.plot(t_ex, phi_ex, 'k--', linewidth=2, label="exakt")

    plt.title(f"Lösungen mit Zeitschrittlänge Δt = {dt}")
    plt.xlabel("t")
    plt.ylabel(r"$\phi(t)$")
    plt.legend()
    plt.grid(True)

    # 科学计数法：×10^-3
    plt.ticklabel_format(style='sci', axis='y', scilimits=(-3, -3))

    # 只在 Δt = 0.4 时，限制 y 轴范围（和老师图一致）
    if dt == 0.4:
        plt.ylim(0.0, 9e-3)

    plt.tight_layout()
    plt.show()

