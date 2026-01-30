import numpy as np

def solveCG(A, b, x0, rtol, itermax):
    """
    Löst Ax = b mit der Konjugierten-Gradienten-Methode
    (A muss symmetrisch positiv definit sein)
    """
    x = x0.astype(float)
    r = b - A @ x
    p = r.copy()

    k = 0
    while np.linalg.norm(r, 2) > rtol and k < itermax:
        Ap = A @ p
        alpha = (r @ r) / (p @ Ap)

        x = x + alpha * p
        r_new = r - alpha * Ap

        beta = (r_new @ r_new) / (r @ r)
        p = r_new + beta * p

        r = r_new
        k += 1

    return x, k


# # Test
# if __name__ == "__main__":
#     A = np.array([[10.0, 2.0, 10.0],
#                   [2.0, 40.0, 8.0],
#                   [10.0, 8.0, 60.0]])
#     b = np.array([1.0, 1.0, 2.0])
#     x0 = np.zeros(3)

#     x, it = solveCG(A, b, x0, 1e-7, 1000)
#     print(x)
#     print("Iterationen:", it)
