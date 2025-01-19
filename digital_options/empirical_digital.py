import numpy as np
import pandas as pd
import pickle
import datetime
import os
import sys
import matplotlib.pyplot as plt
from optim import mcoptim
import calibdensity
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

np.random.seed(12345)

### GILD and GSK ####
### JPM and MS ####
# ticker1 = 'GILD'
# ticker2 = 'GSK'

ticker1 = 'JPM'
ticker2 = 'MS'

### Flags to add L and U constraints ####
Cons_Flag = False

n_size = 10000

maturity_idx1 = 0
maturity_idx2 = 4

option_data = pd.read_csv('E:/GitHub/McCormick/data/2022_{}Call.csv'.format(ticker1))
cur_date = option_data['date'].unique()
scenes = len(cur_date)


######## result logs #########
log_dir = './logs/{}_{}_digital_1month'.format(ticker1, ticker2)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

d_idx = 0 # 0, 60, 124
init_date = cur_date[d_idx]
matur_date = option_data[option_data['date'] == init_date]['exdate'].unique()

dates = [datetime.datetime.strptime(ts, "%Y-%m-%d") for ts in matur_date]
dates.sort()
matur_date = [datetime.datetime.strftime(ts, "%Y-%m-%d") for ts in dates]
matur_date = np.array(matur_date)

t2_date = np.min([maturity_idx2, len(matur_date)-1])
expdate = matur_date[[maturity_idx1, t2_date]]
print(bcolors.GREEN + 'Date of data:', init_date, expdate, bcolors.ENDC)

px_1, X1, px_2, X2, _ = calibdensity.get_dist_ba(f_path='E:/GitHub/McCormick/data/2022_{}Call.csv'.format(ticker1),
                                                init_date=init_date, expiry_date=expdate,
                                                strikes_low=0.5, strikes_upper=1.5,
                                                OI_thrs1=1, OI_thrs2=1, verbose=False)

try:
    py_1, Y1, py_2, Y2, _ = calibdensity.get_dist_ba(f_path='E:/GitHub/McCormick/data/2022_{}Call.csv'.format(ticker2),
                                                     init_date=init_date, expiry_date=expdate,
                                                     strikes_low=0.5, strikes_upper=1.5,
                                                     OI_thrs1=1, OI_thrs2=1, verbose=False)
except ValueError:
    print(bcolors.RED + 'No data on the same maturity, skipped', bcolors.ENDC)
    sys.exit(1)


nx1 = X1.shape[0]
nx2 = X2.shape[0]
ny1 = Y1.shape[0]
ny2 = Y2.shape[0]
print('Problem size', nx1, nx2, ny1, ny2)

scene_size = nx1*nx2*ny1*ny2
ratio_arr = -10.0*np.ones(scene_size)
MOT_max_arr = -10.0*np.ones(scene_size)
MOT_min_arr = -10.0*np.ones(scene_size)
Mc_max_arr = -10.0*np.ones(scene_size)
Mc_min_arr = -10.0*np.ones(scene_size)


for it_ in range(scene_size):
    cost = np.zeros(scene_size)
    cost[it_] = 10000.0

    cost = cost.reshape((nx1, nx2, ny1, ny2))

    MOT_max = mcoptim(True, cost, X1, X2, Y1, Y2, px_1, px_2, py_1, py_2,
                      McCor_Causal=False, McCor_Anticausal=False, cons=Cons_Flag)

    Mc_max = mcoptim(True, cost, X1, X2, Y1, Y2, px_1, px_2, py_1, py_2,
                     McCor_Causal=True, McCor_Anticausal=True, cons=Cons_Flag)

    MOT_min = mcoptim(False, cost, X1, X2, Y1, Y2, px_1, px_2, py_1, py_2,
                      McCor_Causal=False, McCor_Anticausal=False, cons=Cons_Flag)

    Mc_min = mcoptim(False, cost, X1, X2, Y1, Y2, px_1, px_2, py_1, py_2,
                     McCor_Causal=True, McCor_Anticausal=True, cons=Cons_Flag)

    if np.abs(MOT_max - MOT_min)>1e-5:
        ratio = (Mc_max - Mc_min)/(MOT_max - MOT_min)
    else:
        ratio = 1.0

    if it_%100 == 0:
        print(it_, 'out of', scene_size, 'is done.')


    ratio_arr[it_] = ratio
    MOT_max_arr[it_] = MOT_max
    Mc_max_arr[it_] = Mc_max
    MOT_min_arr[it_] = MOT_min
    Mc_min_arr[it_] = Mc_min

ratio_arr = ratio_arr[ratio_arr != -10.0]
MOT_max_arr = MOT_max_arr[MOT_max_arr != -10.0]
Mc_max_arr = Mc_max_arr[Mc_max_arr != -10.0]
MOT_min_arr = MOT_min_arr[MOT_min_arr != -10.0]
Mc_min_arr = Mc_min_arr[Mc_min_arr != -10.0]

print('Average:', np.mean(ratio_arr))
print('Min:', np.min(ratio_arr))

with open('{}/{}_{}_{}_ratios.pickle'.format(log_dir, ticker1, ticker2, init_date), 'wb') as fp:
    pickle.dump(ratio_arr, fp)

with open('{}/{}_{}_{}_MOT_max.pickle'.format(log_dir, ticker1, ticker2, init_date), 'wb') as fp:
    pickle.dump(MOT_max_arr, fp)

with open('{}/{}_{}_{}_Mc_max.pickle'.format(log_dir, ticker1, ticker2, init_date), 'wb') as fp:
    pickle.dump(Mc_max_arr, fp)

with open('{}/{}_{}_{}_MOT_min.pickle'.format(log_dir, ticker1, ticker2, init_date), 'wb') as fp:
    pickle.dump(MOT_min_arr, fp)

with open('{}/{}_{}_{}_Mc_min.pickle'.format(log_dir, ticker1, ticker2, init_date), 'wb') as fp:
    pickle.dump(Mc_min_arr, fp)

with open('{}/{}_{}_{}_px1.pickle'.format(log_dir, ticker1, ticker2, init_date), 'wb') as fp:
    pickle.dump(px_1, fp)
with open('{}/{}_{}_{}_X1.pickle'.format(log_dir, ticker1, ticker2, init_date), 'wb') as fp:
    pickle.dump(X1, fp)

with open('{}/{}_{}_{}_px2.pickle'.format(log_dir, ticker1, ticker2, init_date), 'wb') as fp:
    pickle.dump(px_2, fp)
with open('{}/{}_{}_{}_X2.pickle'.format(log_dir, ticker1, ticker2, init_date), 'wb') as fp:
    pickle.dump(X2, fp)

with open('{}/{}_{}_{}_py1.pickle'.format(log_dir, ticker1, ticker2, init_date), 'wb') as fp:
    pickle.dump(py_1, fp)
with open('{}/{}_{}_{}_Y1.pickle'.format(log_dir, ticker1, ticker2, init_date), 'wb') as fp:
    pickle.dump(Y1, fp)

with open('{}/{}_{}_{}_py2.pickle'.format(log_dir, ticker1, ticker2, init_date), 'wb') as fp:
    pickle.dump(py_2, fp)
with open('{}/{}_{}_{}_Y2.pickle'.format(log_dir, ticker1, ticker2, init_date), 'wb') as fp:
    pickle.dump(Y2, fp)
