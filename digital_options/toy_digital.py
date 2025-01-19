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

payoff_sample_size = nx1*nx2*ny1*ny2
Cons_Flag = False

######## result logs #########
ratio_arr = np.zeros(payoff_sample_size)
BCOT_ratio_arr = np.zeros(payoff_sample_size)
MOT_max_arr = np.zeros(payoff_sample_size)
MOT_min_arr = np.zeros(payoff_sample_size)
Mc_max_arr = np.zeros(payoff_sample_size)
Mc_min_arr = np.zeros(payoff_sample_size)
BCOT_max_arr = np.zeros(payoff_sample_size)
BCOT_min_arr = np.zeros(payoff_sample_size)


X1 = np.array([1, 2, 3])
X2 = X1
Y1 = np.array([2, 3, 4])
Y2 = Y1
px_1 = np.array([0.01, 0.98, 0.01])
px_2 = np.array([0.04, 0.92, 0.04])

py_1 = np.array([0.4, 0.2, 0.4])
py_2 = np.array([0.4, 0.2, 0.4])

for it_ in range(payoff_sample_size):

    cost = np.zeros(payoff_sample_size)
    cost[it_] = 10000.0
    cost = cost.reshape((nx1, nx2, ny1, ny2))

    try:
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
    except gurobipy.GurobiError:
        print(bcolors.RED + 'A solution does not exist. Skipped', bcolors.ENDC)
        continue
    except ValueError:
        continue
    except AttributeError:
        continue


    if np.abs(MOT_max - MOT_min) > 1e-5:
        ratio = (Mc_max - Mc_min) / (MOT_max - MOT_min)
        BC_ratio = (BCOT_max - BCOT_min) / (MOT_max - MOT_min)
    else:
        ratio = 1.0
        BC_ratio = 1.0
    if BC_ratio < 0.03:
        print(np.where(cost>0))
        print('BC Ratio', BC_ratio)
        print('BCOT max', BCOT_max)
        print('Mc max', Mc_max)
        print('MOT max', MOT_max)
        print('BCOT min', BCOT_min)
        print('Mc min', Mc_min)
        print('MOT min', MOT_min)

    ratio_arr[it_] = ratio
    BCOT_ratio_arr[it_] = BC_ratio
    MOT_max_arr[it_] = MOT_max
    BCOT_max_arr[it_] = BCOT_max
    Mc_max_arr[it_] = Mc_max
    MOT_min_arr[it_] = MOT_min
    Mc_min_arr[it_] = Mc_min
    BCOT_min_arr[it_] = BCOT_min

print('Summary')
print(bcolors.GREEN + 'McCormick Average Ratio:', np.mean(ratio_arr), bcolors.ENDC)
print(bcolors.GREEN + 'McCormick Minimum Ratio:', np.min(ratio_arr), bcolors.ENDC)

print(bcolors.GREEN + 'BCOT Average Ratio:', np.mean(BCOT_ratio_arr), bcolors.ENDC)
print(bcolors.GREEN + 'BCOT Minimum Ratio:', np.min(BCOT_ratio_arr), bcolors.ENDC)


log_dir = './logs/toy_digital'

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

with open('{}/ratios.pickle'.format(log_dir), 'wb') as fp:
    pickle.dump(ratio_arr, fp)

with open('{}/BCOT_ratios.pickle'.format(log_dir), 'wb') as fp:
    pickle.dump(BCOT_ratio_arr, fp)

with open('{}/MOT_max.pickle'.format(log_dir), 'wb') as fp:
    pickle.dump(MOT_max_arr, fp)

with open('{}/BCOT_max.pickle'.format(log_dir), 'wb') as fp:
    pickle.dump(BCOT_max_arr, fp)

with open('{}/Mc_max.pickle'.format(log_dir), 'wb') as fp:
    pickle.dump(Mc_max_arr, fp)

with open('{}/MOT_min.pickle'.format(log_dir), 'wb') as fp:
    pickle.dump(MOT_min_arr, fp)

with open('{}/Mc_min.pickle'.format(log_dir), 'wb') as fp:
    pickle.dump(Mc_min_arr, fp)

with open('{}/BCOT_min.pickle'.format(log_dir), 'wb') as fp:
    pickle.dump(BCOT_min_arr, fp)
