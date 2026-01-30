import numpy as np

def solveG(A, b, x0, rtol, itermax):
    """
    Löst Ax = b mit der Gradienten-Methode
    """
    x = x0.astype(float)
    r = b - A @ x

    k = 0
    while np.linalg.norm(r, 2) > rtol and k < itermax:
        Ar = A @ r
        alpha = (r @ r) / (r @ Ar)

        x = x + alpha * r
        r = r - alpha * Ar

        k += 1

    return x, k


# # Test
# if __name__ == "__main__":
#     A = np.array([[10.0, 2.0, 10.0],
#                   [2.0, 40.0, 8.0],
#                   [10.0, 8.0, 60.0]])
#     b = np.array([1.0, 1.0, 2.0])
#     x0 = np.zeros(3)

#     x, it = solveG(A, b, x0, 1e-7, 1000)
#     print(x)
#     print("Iterationen:", it)
