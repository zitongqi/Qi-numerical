import numpy as np

from FktXX import evaluate_instat
from FktV import gx2dref
from FktVI import gw2dref

# -------------------------------------------------
# element geometry (Q4, arbitrary quad)
# -------------------------------------------------
elenodes = np.array([
    [0.0, 0.0],
    [1.5, 0.0],
    [1.0, 2.5],
    [0.0, 2.1]
], dtype=float)

# -------------------------------------------------
# Gauss points & weights (3x3)
# -------------------------------------------------
gpx = gx2dref(3)                 # shape (9, 2)
gpw = gw2dref(3).reshape(-1)     # shape (9,)

# -------------------------------------------------
# solution vectors
# -------------------------------------------------
elesol  = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
eleosol = np.zeros(4, dtype=float)

# -------------------------------------------------
# call element routine
# -------------------------------------------------
A_e, f_e = evaluate_instat(
    elenodes=elenodes,
    gpx=gpx,
    gpw=gpw,
    elesol=elesol,
    eleosol=eleosol,
    timInt_m=1,      # OST
    timestep=1.0,
    theta=0.66,
    firststep=True   # 目前 OST 里未用到，但接口保持一致
)

# -------------------------------------------------
# output
# -------------------------------------------------
np.set_printoptions(precision=8, suppress=True)

print("Element matrix A(e):")
print(A_e)

print("\nElement vector f(e):")
print(f_e)
