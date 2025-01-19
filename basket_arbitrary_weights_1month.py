import numpy as np
import pandas as pd
import pickle
import datetime
import os
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
ticker1 = 'GILD'
ticker2 = 'GSK'

### Flags to add L and U constraints ####
Cons_Flag = False

# number of samples for weights
n_sample = 100

w1_arr = np.random.randint(1, 11, size=(n_sample, 4))

# 1 month
maturity_idx1 = 0
maturity_idx2 = 4

option_data = pd.read_csv('./data/2022_{}Call.csv'.format(ticker1))
cur_date = option_data['date'].unique()
scenes = len(cur_date)

######## result logs #########
ratio_arr = np.zeros((scenes, n_sample))
MOT_max_arr = np.zeros((scenes, n_sample))
MOT_min_arr = np.zeros((scenes, n_sample))
Mc_max_arr = np.zeros((scenes, n_sample))
Mc_min_arr = np.zeros((scenes, n_sample))


for d_idx in range(scenes):

    init_date = cur_date[d_idx]
    matur_date = option_data[option_data['date'] == init_date]['exdate'].unique()

    dates = [datetime.datetime.strptime(ts, "%Y-%m-%d") for ts in matur_date]
    dates.sort()
    matur_date = [datetime.datetime.strftime(ts, "%Y-%m-%d") for ts in dates]
    matur_date = np.array(matur_date)

    t2_date = np.min([maturity_idx2, len(matur_date)-1])
    expdate = matur_date[[maturity_idx1, t2_date]]
    print(bcolors.GREEN + 'Date of data:', init_date, expdate, bcolors.ENDC)

    px_1, X1, px_2, X2, _ = calibdensity.get_dist_ba(f_path='./data/2022_{}Call.csv'.format(ticker1),
                                                    init_date=init_date, expiry_date=expdate,
                                                    strikes_low=0.5, strikes_upper=1.5,
                                                    OI_thrs1=1, OI_thrs2=1, verbose=False)

    try:
        py_1, Y1, py_2, Y2, _ = calibdensity.get_dist_ba(f_path='./data/2022_{}Call.csv'.format(ticker2),
                                                         init_date=init_date, expiry_date=expdate,
                                                         strikes_low=0.5, strikes_upper=1.5,
                                                         OI_thrs1=1, OI_thrs2=1, verbose=False)
    except ValueError:
        print(bcolors.RED + 'No data on the same maturity, skipped', bcolors.ENDC)
        ratio_arr[d_idx, :] = -10.0
        MOT_max_arr[d_idx, :] = -10.0
        Mc_max_arr[d_idx, :] = -10.0
        MOT_min_arr[d_idx, :] = -10.0
        Mc_min_arr[d_idx, :] = -10.0
        continue


    nx1 = X1.shape[0]
    nx2 = X2.shape[0]
    ny1 = Y1.shape[0]
    ny2 = Y2.shape[0]
    print('Problem size', nx1, nx2, ny1, ny2)

    cost = np.zeros((nx1, nx2, ny1, ny2))
    # print('X1', X1)
    # print('X2', X2)
    # print('Y1', Y1)
    # print('Y2', Y2)

    for it_ in range(n_sample):

        weight = w1_arr[it_, :]
        weight = weight/weight.sum()

        strike = int(((px_1*X1).sum() + (py_1*Y1).sum())/2)

        def weighted_price(x):
            y = np.dot(weight, x)
            return y

        for x1_idx in range(nx1):
            for x2_idx in range(nx2):
                for y1_idx in range(ny1):
                    for y2_idx in range(ny2):
                        ## basket option
                        avg = weighted_price(np.array([X1[x1_idx], Y1[y1_idx], X2[x2_idx], Y2[y2_idx]]))
                        cost[x1_idx, x2_idx, y1_idx, y2_idx] = np.max([avg - strike, 0.0])

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

        ratio_arr[d_idx, it_] = ratio
        MOT_max_arr[d_idx, it_] = MOT_max
        Mc_max_arr[d_idx, it_] = Mc_max
        MOT_min_arr[d_idx, it_] = MOT_min
        Mc_min_arr[d_idx, it_] = Mc_min

    print('Average:', np.mean(ratio_arr[d_idx, :]))
    print('Min:', np.min(ratio_arr[d_idx, :]))

log_dir = './logs/{}_{}_weighted_1month'.format(ticker1, ticker2)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

ratio_arr = ratio_arr[ratio_arr[:, 0] != -10.0]
MOT_max_arr = MOT_max_arr[MOT_max_arr[:, 0] != -10.0]
Mc_max_arr = Mc_max_arr[Mc_max_arr[:, 0] != -10.0]
MOT_min_arr = MOT_min_arr[MOT_min_arr[:, 0] != -10.0]
Mc_min_arr = Mc_min_arr[Mc_min_arr[:, 0] != -10.0]


with open('{}/{}_{}_ratios.pickle'.format(log_dir, ticker1, ticker2), 'wb') as fp:
    pickle.dump(ratio_arr, fp)

with open('{}/{}_{}_MOT_max.pickle'.format(log_dir, ticker1, ticker2), 'wb') as fp:
    pickle.dump(MOT_max_arr, fp)

with open('{}/{}_{}_Mc_max.pickle'.format(log_dir, ticker1, ticker2), 'wb') as fp:
    pickle.dump(Mc_max_arr, fp)

with open('{}/{}_{}_MOT_min.pickle'.format(log_dir, ticker1, ticker2), 'wb') as fp:
    pickle.dump(MOT_min_arr, fp)

with open('{}/{}_{}_Mc_min.pickle'.format(log_dir, ticker1, ticker2), 'wb') as fp:
    pickle.dump(Mc_min_arr, fp)

