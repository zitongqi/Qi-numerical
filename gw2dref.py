import numpy as np

def gw2dref(n):
    p = 0
    gaussw = []

    if n == 1:
        w1 = [2.0]
        for i in range(n):
            for j in range(n):
                gaussw.append(w1[i] * w1[j])

    if n == 2:
        w2 = [1.0, 1.0]
        for i in range(n):
            for j in range(n):
                gaussw.append(w2[i] * w2[j])

    if n == 3:
        w3 = [5.0/9.0, 8.0/9.0, 5.0/9.0]
        for i in range(n):
            for j in range(n):
                gaussw.append(w3[i] * w3[j])

    return np.array(gaussw, dtype=float)
