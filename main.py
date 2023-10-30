import numpy as np
import pandas as pd
import pickle
import datetime
import os
import matplotlib.pyplot as plt
from optim import mcoptim
import calibdensity

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

######## result logs #########
MOT_max = []
MOT_min = []
Mc_max = []
Mc_min = []

time0 = []
time1 = []
time2 = []
strike_hist = []

density_x1 = []
support_x1 = []
density_x2 = []
support_x2 = []
density_y1 = []
support_y1 = []
density_y2 = []
support_y2 = []

### GILD and GSK ####
### JPM and MS ####
ticker1 = 'GILD'
ticker2 = 'GSK'

maturity_idx1 = 0
maturity_idx2 = 4

option_data = pd.read_csv('./data/2022_{}Call.csv'.format(ticker1))
cur_date = option_data['date'].unique()
scenes = len(cur_date)

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
    strike = int(((px_1*X1).sum() + (px_2*X2).sum() + (py_1*Y1).sum() + (py_2*Y2).sum())/4)
    print('Strike:', strike)
    for x1_idx in range(nx1):
        for x2_idx in range(nx2):
            for y1_idx in range(ny1):
                for y2_idx in range(ny2):
                    ## basket option
                    cost[x1_idx, x2_idx, y1_idx, y2_idx] = np.max([(X1[x1_idx]+Y1[y1_idx]+X2[x2_idx]+Y2[y2_idx])/4 - strike, 0])


    time0.append(init_date)
    time1.append(expdate[0])
    time2.append(expdate[1])
    strike_hist.append(strike)

    density_x1.append(px_1)
    support_x1.append(X1)
    density_x2.append(px_2)
    support_x2.append(X2)

    density_y1.append(py_1)
    support_y1.append(Y1)
    density_y2.append(py_2)
    support_y2.append(Y2)


    for MAX_FLAG in [True, False]:
        objval_0 = mcoptim(MAX_FLAG, cost, X1, X2, Y1, Y2, px_1, px_2, py_1, py_2,
                           McCor_Causal=False, McCor_Anticausal=False)


        objval_2 = mcoptim(MAX_FLAG, cost, X1, X2, Y1, Y2, px_1, px_2, py_1, py_2,
                           McCor_Causal=True, McCor_Anticausal=True)

        print(bcolors.PINK + 'MOT:', objval_0, bcolors.ENDC)
        print(bcolors.BLUE + 'McCormick:', objval_2, bcolors.ENDC)

        ### logging results ####
        if MAX_FLAG:
            MOT_max.append(objval_0)
            Mc_max.append(objval_2)
        else:
            MOT_min.append(objval_0)
            Mc_min.append(objval_2)

MOT_max = np.array(MOT_max)
MOT_min = np.array(MOT_min)
Mc_max = np.array(Mc_max)
Mc_min = np.array(Mc_min)
# print('MOT Max', MOT_max)
# print('MOT Min', MOT_min)

ratios = np.divide(Mc_max - Mc_min, MOT_max - MOT_min)
print('Number of tests:', len(ratios))
print('Average ratios:', ratios.mean())
print('Min of ratios:', ratios.min())


log_dir = './{}_{}_logs'.format(ticker1, ticker2)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

res_df = pd.DataFrame({'time0': time0, 'time1': time1, 'time2': time2, 'strike': strike_hist,
                       'MOT_max': MOT_max, 'MOT_min': MOT_min,
                       'McCormick_max': Mc_max, 'McCormick_min': Mc_min,
                       'Ratios': ratios})

res_df.to_csv('{}/{}_{}.csv'.format(log_dir, ticker1, ticker2), index=False)
# plt.figure()
# plt.hist(ratios)
# plt.legend(loc='best', fontsize=15)
# plt.savefig('{}/{}_{}.pdf'.format(log_dir, ticker1, ticker2), format='pdf',
#             dpi=1000, bbox_inches='tight', pad_inches=0.1)



with open('{}/{}_{}_px1.pickle'.format(log_dir, ticker1, ticker2), 'wb') as fp:
    pickle.dump(density_x1, fp)
with open('{}/{}_{}_X1.pickle'.format(log_dir, ticker1, ticker2), 'wb') as fp:
    pickle.dump(support_x1, fp)

with open('{}/{}_{}_px2.pickle'.format(log_dir, ticker1, ticker2), 'wb') as fp:
    pickle.dump(density_x2, fp)
with open('{}/{}_{}_X2.pickle'.format(log_dir, ticker1, ticker2), 'wb') as fp:
    pickle.dump(support_x2, fp)

with open('{}/{}_{}_py1.pickle'.format(log_dir, ticker1, ticker2), 'wb') as fp:
    pickle.dump(density_y1, fp)
with open('{}/{}_{}_Y1.pickle'.format(log_dir, ticker1, ticker2), 'wb') as fp:
    pickle.dump(support_y1, fp)

with open('{}/{}_{}_py2.pickle'.format(log_dir, ticker1, ticker2), 'wb') as fp:
    pickle.dump(density_y2, fp)
with open('{}/{}_{}_Y2.pickle'.format(log_dir, ticker1, ticker2), 'wb') as fp:
    pickle.dump(support_y2, fp)

