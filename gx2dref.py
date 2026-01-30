import numpy as np

def gx2dref(n):
    gaussx = []
    
    if n == 1:
        nx = [0.0]
        ny = [0.0]

        for i in range(n):
            for j in range(n):
                gaussx.append([nx[i], ny[j]])

    elif n == 2:
        a = 1.0 / np.sqrt(3.0)
        nx = [-a, a]
        ny = [-a, a]

        for i in range(n):
            for j in range(n):
                gaussx.append([nx[i], ny[j]])

    elif n == 3:
        a = np.sqrt(3.0 / 5.0)
        nx = [-a, 0.0, a]
        ny = [-a, 0.0, a]

        for i in range(n):
            for j in range(n):
                gaussx.append([nx[i], ny[j]])

    else:
        raise ValueError("n must be 1, 2 or 3")

    return np.array(gaussx, dtype=float)
