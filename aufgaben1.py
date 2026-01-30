# aufgabe3.py
import numpy as np
from Fkt_0 import quadplot

# ---------- nodes ----------
nodes = np.array([
    [-1, -1],  # 1
    [ 0, -1],  # 2
    [ 1, -1],  # 3
    [-1,  0],  # 4
    [ 0,  0],  # 5
    [ 1,  0],  # 6
    [-1,  1],  # 7
    [ 0,  1],  # 8
    [ 1,  1]   # 9
])

elements = [
  [1, 2, 5, 4],
  [2, 3, 6, 5],
  [4, 5, 8, 7],
  [5, 6, 9, 8]
]



# ---------- solution ----------
sol = np.array([
    2, 1, 2,
    1, 0, 1,
    2, 1, 2
])

# ---------- plot ----------
quadplot(nodes, elements, sol)
