# main_aufgabe3.py
import numpy as np

from FktI import linquadref
from FktII import linquadderivref

def approx_f(xi: float, eta: float, f_nodes: np.ndarray) -> float:
    """
    f(xi,eta) = sum_i N_i(xi,eta) * f_i
    """
    N = linquadref(xi, eta)          # (4,)
    return float(N @ f_nodes)

def approx_df(xi: float, eta: float, f_nodes: np.ndarray) -> tuple[float, float]:
    """
    df/dxi  = sum_i (∂N_i/∂xi)  * f_i
    df/deta = sum_i (∂N_i/∂eta) * f_i
    """
    dN = linquadderivref(xi, eta)    # (4,2)
    df_dxi  = float(dN[:, 0] @ f_nodes)
    df_deta = float(dN[:, 1] @ f_nodes)
    return df_dxi, df_deta

def main():
    # Given nodal values from the sheet:
    # Node order: 1(-1,-1), 2(1,-1), 3(1,1), 4(-1,1)
    f_nodes = np.array([0.0, 1.0, 3.0, 1.0], dtype=float)

    # Points to evaluate
    pts = [(0.0, 0.0), (0.577, -0.577)]

    print("=== Step 1: Test Fkt. I outputs (shape functions) ===")
    for xi, eta in pts:
        print(f"(xi,eta)=({xi:.3f},{eta:.3f}) -> N =", linquadref(xi, eta))
    print()

    print("=== Step 2: Approximate f(xi,eta) ===")
    for xi, eta in pts:
        f_val = approx_f(xi, eta, f_nodes)
        print(f"f_L({xi:.3f},{eta:.3f}) = {f_val:.8f}")
    print("Expected (Lsg.): f_L(0,0)=1.25 ; f_L(0.577,-0.577)=1.16676775")
    print()

    print("=== Step 3: Test Fkt. II outputs (shape function derivatives) ===")
    for xi, eta in pts:
        print(f"(xi,eta)=({xi:.3f},{eta:.3f}) -> dN =\n{linquadderivref(xi, eta)}")
    print()

    print("=== Step 4: Approximate derivatives df/dxi and df/deta ===")
    for xi, eta in pts:
        df_dxi, df_deta = approx_df(xi, eta, f_nodes)
        print(f"(xi,eta)=({xi:.3f},{eta:.3f})  df/dxi={df_dxi:.8f}  df/deta={df_deta:.8f}")
    print("Expected (Lsg.):")
    print("  at (0,0): df/dxi=0.75 ; df/deta=0.75")
    print("  at (0.577,-0.577): df/dxi=0.60575 ; df/deta=0.89425")

if __name__ == "__main__":
    main()
