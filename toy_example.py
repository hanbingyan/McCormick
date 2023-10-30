import numpy as np
from optim import mcoptim

X1 = np.array([11.0, 10.0, 9.0])
Y1 = np.array([24.0, 20.0, 16.0])
# support at time 2
X2 = np.array([20.0, 10.0, 0.0])
Y2 = np.array([26.0, 20.0, 14.0])
# probability
# time 1 marginals
px_1 = np.array([0.2, 0.6, 0.2])
py_1 = np.array([0.3, 0.4, 0.3])
# time 2 marginals
px_2 = np.array([0.1, 0.8, 0.1])
py_2 = np.array([0.2, 0.6, 0.2])


# number of points at each time
nx1 = X1.shape[0]
nx2 = X2.shape[0]
ny1 = Y1.shape[0]
ny2 = Y2.shape[0]
print('Problem size', nx1, nx2, ny1, ny2)

cost = np.zeros((nx1, nx2, ny1, ny2))
for x1_idx in range(nx1):
    for x2_idx in range(nx2):
        for y1_idx in range(ny1):
            for y2_idx in range(ny2):
                cost[x1_idx, x2_idx, y1_idx, y2_idx] = np.max([(X2[x2_idx] - X1[x1_idx])**2,
                                                               (Y2[y2_idx] - Y1[y1_idx])**2])






for MAX_FLAG in [True, False]:
    objval_1 = mcoptim(MAX_FLAG, cost, X1, X2, Y1, Y2, px_1, px_2, py_1, py_2,
                       McCor_Causal=False, McCor_Anticausal=False)

    objval_2 = mcoptim(MAX_FLAG, cost, X1, X2, Y1, Y2, px_1, px_2, py_1, py_2,
                       McCor_Causal=True, McCor_Anticausal=True, Causal=False, Anticausal=False)

    objval_3 = mcoptim(MAX_FLAG, cost, X1, X2, Y1, Y2, px_1, px_2, py_1, py_2,
                       McCor_Causal=False, McCor_Anticausal=False, Causal=True, Anticausal=True)

    if MAX_FLAG:
        MOT_max = objval_1
        Mc_max = objval_2
        BI_max = objval_3
    else:
        MOT_min = objval_1
        Mc_min = objval_2
        BI_min = objval_3


print('MOT', MOT_min, MOT_max)
print('McCormick', Mc_min, Mc_max)
print('Bicausal', BI_min, BI_max)

ratios = np.divide(Mc_max - Mc_min, MOT_max - MOT_min)
print('Ratios:', ratios)