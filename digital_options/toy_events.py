import numpy as np
from optim import mcoptim
import os
import pickle
import gurobipy

class bcolors:
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    PINK = '\033[35m'
    GREY = '\033[36m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

nx1 = 3
nx2 = 3
ny1 = 3
ny2 = 3

print('Problem size', nx1, nx2, ny1, ny2)

Cons_Flag = False


X1 = np.array([1, 2, 3])
X2 = X1
Y1 = np.array([2, 3, 4])
Y2 = Y1
px_1 = np.array([0.01, 0.98, 0.01])
px_2 = np.array([0.04, 0.92, 0.04])

py_1 = np.array([0.4, 0.2, 0.4])
py_2 = np.array([0.4, 0.2, 0.4])

print('mean of X', (X1*px_1).sum(), (X2*px_2).sum())
print('mean of Y', (Y1*py_1).sum(), (Y2*py_2).sum())

cost = np.zeros((nx1, nx2, ny1, ny2))
for x1_idx in range(nx1):
    for x2_idx in range(nx2):
        for y1_idx in range(ny1):
            for y2_idx in range(ny2):
                if X1[x1_idx] <=2 and X2[x2_idx] >2 and Y2[y2_idx] >=3:
                    cost[x1_idx, x2_idx, y1_idx, y2_idx] = 10000.0


MOT_max = mcoptim(True, cost, X1, X2, Y1, Y2, px_1, px_2, py_1, py_2,
                  McCor_Causal=False, McCor_Anticausal=False, cons=Cons_Flag)

Mc_max = mcoptim(True, cost, X1, X2, Y1, Y2, px_1, px_2, py_1, py_2,
                 McCor_Causal=True, McCor_Anticausal=True, cons=Cons_Flag)

MOT_min = mcoptim(False, cost, X1, X2, Y1, Y2, px_1, px_2, py_1, py_2,
                  McCor_Causal=False, McCor_Anticausal=False, cons=Cons_Flag)

Mc_min = mcoptim(False, cost, X1, X2, Y1, Y2, px_1, px_2, py_1, py_2,
                 McCor_Causal=True, McCor_Anticausal=True, cons=Cons_Flag)

BCOT_max = mcoptim(True, cost, X1, X2, Y1, Y2, px_1, px_2, py_1, py_2,
                   McCor_Causal=False, McCor_Anticausal=False, Causal=True, Anticausal=True,
                   cons=Cons_Flag, verbose=False)

BCOT_min = mcoptim(False, cost, X1, X2, Y1, Y2, px_1, px_2, py_1, py_2,
                   McCor_Causal=False, McCor_Anticausal=False, Causal=True, Anticausal=True,
                   cons=Cons_Flag, verbose=False)


if np.abs(MOT_max - MOT_min) > 1e-5:
    ratio = (Mc_max - Mc_min) / (MOT_max - MOT_min)
    BC_ratio = (BCOT_max - BCOT_min) / (MOT_max - MOT_min)
else:
    ratio = 1.0
    BC_ratio = 1.0

print('Ratio', ratio)
print('BC Ratio', BC_ratio)
print('BCOT max', BCOT_max)
print('Mc max', Mc_max)
print('MOT max', MOT_max)
print('BCOT min', BCOT_min)
print('Mc min', Mc_min)
print('MOT min', MOT_min)
