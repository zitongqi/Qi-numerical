import numpy as np

def solveGauss(A, b):
    """
    Löst Ax = b mit dem Gaußschen Eliminationsverfahren
    (ohne Zeilentausch)
    """
    A = A.astype(float)
    b = b.astype(float)

    n = len(b)

    # Vorwärtselimination
    for i in range(n):
        if abs(A[i, i]) < 1e-14:
            raise ValueError("Null-Pivot aufgetreten (kein Zeilentausch erlaubt)")

        for j in range(i + 1, n):
            m = A[j, i] / A[i, i]
            A[j, i:] -= m * A[i, i:]
            b[j] -= m * b[i]

    # Rückwärtseinsetzen
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i + 1:], x[i + 1:])) / A[i, i]

    return x


# # Test
# if __name__ == "__main__":
#     A = np.array([[10.0, 2.0, 1.0],
#                   [3.0, 4.0, 4.0],
#                   [1.0, 8.0, 4.0]])
#     b = np.array([1.0, 1.0, 2.0])

#     print(solveGauss(A, b))
