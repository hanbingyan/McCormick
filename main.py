import numpy as np
import pandas as pd
import pickle
import datetime
import os
import matplotlib.pyplot as plt
from gurobipy import *
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


def causal_test(MAX, cost, X1, X2, Y1, Y2, px_1, px_2, py_1, py_2, McCor_Causal, McCor_Anticausal, verbose=False):
    m = Model('Primal')
    if verbose:
        m.setParam('OutputFlag', 1)
        m.setParam('LogToConsole', 1)
    else:
        m.setParam('OutputFlag', 0)
        m.setParam('LogToConsole', 0)


    fwdx1 = (X1*px_1).sum()
    fwdx2 = (X2*px_2).sum()
    fwdy1 = (Y1*py_1).sum()
    fwdy2 = (Y2*py_2).sum()

    pix = m.addVars(nx1, nx2, lb=0.0, ub=1.0, name='transportx')
    piy = m.addVars(ny1, ny2, lb=0.0, ub=1.0, name='transporty')
    # marginal constraints
    m.addConstrs((pix.sum(i, '*') == px_1[i] for i in range(nx1)), name='x1_marginal')
    m.addConstrs((pix.sum('*', i) == px_2[i] for i in range(nx2)), name='x2_marginal')
    m.addConstrs((piy.sum(i, '*') == py_1[i] for i in range(ny1)), name='y1_marginal')
    m.addConstrs((piy.sum('*', i) == py_2[i] for i in range(ny2)), name='y2_marginal')

    pi = m.addVars(nx1, nx2, ny1, ny2, lb=0.0, ub=1.0, name='transport')
    # marginal constraints
    m.addConstrs((pi.sum(i, '*', '*', '*') == pix.sum(i, '*') for i in range(nx1)), name='x1_marginal')
    m.addConstrs((pi.sum('*', i, '*', '*') == pix.sum('*', i) for i in range(nx2)), name='x2_marginal')
    m.addConstrs((pi.sum('*', '*', i, '*') == piy.sum(i, '*') for i in range(ny1)), name='y1_marginal')
    m.addConstrs((pi.sum('*', '*', '*', i) == piy.sum('*', i) for i in range(ny2)), name='y2_marginal')

    ## joint martingale constraints
    m.addConstrs(
        ((quicksum(pi.sum(i, j, k, '*')*X2[j]*fwdx1 for j in range(nx2)) == fwdx2*X1[i]*pi.sum(i, '*', k, '*')) for i in
         range(nx1) for k in range(ny1)),
        name='martingale_at_x1')
    m.addConstrs(
        ((quicksum(pi.sum(i, '*', k, l)*Y2[l]*fwdy1 for l in range(ny2)) == fwdy2*Y1[k]*pi.sum(i, '*', k, '*')) for k in
         range(ny1) for i in range(nx1)),
        name='martingale_at_y1')


    if McCor_Causal:

        for x1id_ in range(nx1):
            for x2id_ in range(nx2):
                for y1id_ in range(ny1):
                    x1prob_ = px_1[x1id_]
                    x2prob_ = px_2[x2id_]
                    y1prob_ = py_1[y1id_]
                    q_upper = min(x1prob_, x2prob_)
                    p_upper = min(x1prob_, y1prob_)
                    m.addConstr(pi.sum(x1id_, x2id_, y1id_, '*') * x1prob_ <= q_upper * pi.sum(x1id_, '*', y1id_, '*'),
                                name='McCormick1' + '_' + str(x1id_) + str(x2id_) + str(y1id_))
                    m.addConstr(pi.sum(x1id_, x2id_, y1id_, '*') * x1prob_ <= p_upper * pi.sum(x1id_, x2id_, '*', '*'),
                                name='McCormick2' + '_' + str(x1id_) + str(x2id_) + str(y1id_))
                    m.addConstr(q_upper * pi.sum(x1id_, '*', y1id_, '*') + p_upper * pi.sum(x1id_, x2id_, '*', '*') -
                                p_upper * q_upper <= pi.sum(x1id_, x2id_, y1id_, '*') * x1prob_,
                                name='McCormick3' + '_' + str(x1id_) + str(x2id_) + str(y1id_))


    if McCor_Anticausal:

        for x1id_ in range(nx1):
            for y1id_ in range(ny1):
                for y2id_ in range(ny2):
                    x1prob_ = px_1[x1id_]
                    y1prob_ = py_1[y1id_]
                    y2prob_ = py_2[y2id_]
                    q_upper = min(y1prob_, y2prob_)
                    p_upper = min(x1prob_, y1prob_)
                    m.addConstr(pi.sum(x1id_, '*', y1id_, y2id_)*y1prob_ <= q_upper*pi.sum(x1id_, '*', y1id_, '*'),
                                name='McCormickA1' + '_' + str(x1id_) + str(y1id_) + str(y2id_))
                    m.addConstr(pi.sum(x1id_, '*', y1id_, y2id_)*y1prob_ <= p_upper*pi.sum('*', '*', y1id_, y2id_),
                                name='McCormickA2' + '_' + str(x1id_) + str(y1id_) + str(y2id_))
                    m.addConstr(q_upper*pi.sum(x1id_, '*', y1id_, '*') + p_upper*pi.sum('*', '*', y1id_, y2id_) -
                                p_upper*q_upper <= pi.sum(x1id_, '*', y1id_, y2id_) * y1prob_,
                                name='McCormickA3' + '_' + str(x1id_) + str(y1id_) + str(y2id_))


    # # Causal constraints
    # if CAUSAL:
    #     for x1_idx in range(nx1):
    #         for x2_idx in range(nx2):
    #             for y1_idx in range(ny1):
    #                 m.addConstr(px_1[x1_idx] * pi.sum(x1_idx, x2_idx, y1_idx, '*') ==
    #                             pi.sum(x1_idx, '*', y1_idx, '*') * pix[x1_idx, x2_idx],
    #                             name='anticausal' + '_' + str(x1_idx) + '_' + str(x2_idx) + '_' + str(y1_idx))
    #
    #
    # if ANTICAUSAL:
    #     for y1_idx in range(ny1):
    #         for y2_idx in range(ny2):
    #             for x1_idx in range(nx1):
    #                 m.addConstr(py_1[y1_idx] * pi.sum(x1_idx, '*', y1_idx, y2_idx) ==
    #                             pi.sum(x1_idx, '*', y1_idx, '*') * piy[y1_idx, y2_idx],
    #                             name='causal' + '_' + str(y1_idx) + '_' + str(y2_idx) + '_' + str(x1_idx))




    obj = quicksum(
        [cost[i, j, k, l] * pi[i, j, k, l] for i in range(nx1) for j in range(nx2) for k in range(ny1) for l in
         range(ny2)])

    if MAX:
        m.setObjective(obj, GRB.MAXIMIZE)
    else:
        m.setObjective(obj, GRB.MINIMIZE)


    # if CAUSAL or ANTICAUSAL:
    #     m.params.NonConvex = 2
    #     m.setParam('MIPGap', 1e-3)
    # print('Maximizing the exotic option...')

    m.optimize()
    m.printQuality()

    names_to_retrieve = (f"transport[{i},{j},{k},{l}]" for i in range(nx1) for j in range(nx2) for k in range(ny1) for l
                         in range(ny2))
    coupling = np.array([m.getVarByName(name).X for name in names_to_retrieve])
    # for ans in m.getVars():
    #   print('%s %g' % (ans.VarName, ans.X))

    coupling = coupling.reshape((nx1, nx2, ny1, ny2))

    # print('Obj: %g' % obj.getValue())

    ############ Find where causal condition is violated ###################
    # anti_voli_list = []
    # for y1_idx in range(ny1):
    #     for y2_idx in range(ny2):
    #         for x1_idx in range(nx1):
    #             LHS = py_1[y1_idx] * np.sum(coupling[x1_idx, :, y1_idx, y2_idx])
    #             RHS = np.sum(coupling[x1_idx, :, y1_idx, :]) * np.sum(coupling[:, :, y1_idx, y2_idx])
    #             if np.abs(LHS - RHS) > 1e-3:
    #                 # print(bcolors.RED + 'Noncausal coupling found!!!' + bcolors.ENDC)
    #                 # print('LHS', LHS, 'RHS', RHS)
    #                 # print('y1', 'y2', 'x1', y1_idx, y2_idx, x1_idx)
    #                 anti_voli_list.append([x1_idx, y1_idx, y2_idx])


    # cau_voli_list = []
    # for x1_idx in range(nx1):
    #     for x2_idx in range(nx2):
    #         for y1_idx in range(ny1):
    #             LHS = px_1[x1_idx]*np.sum(coupling[x1_idx, x2_idx, y1_idx, :])
    #             RHS = np.sum(coupling[x1_idx, :, y1_idx, :]) * np.sum(coupling[x1_idx, x2_idx, :, :])
    #             if np.abs(LHS - RHS) > 1e-3:
    #                 # print('Noncausal coupling found!!!')
    #                 # print('LHS', LHS, 'RHS', RHS)
    #                 # print('y1', 'x1', 'x2', y1_idx, x1_idx, x2_idx)
    #                 # cau_voli_list.append(np.abs(LHS-RHS))
    #                 cau_voli_list.append([y1_idx, x1_idx, x2_idx])

    # return cau_voli_list, anti_voli_list, obj.getValue()
    return obj.getValue()



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


ticker1 = 'JPM'
ticker2 = 'MS'

maturity_idx1 = 0
maturity_idx2 = 4

option_data = pd.read_csv('D:/2022_{}Call.csv'.format(ticker1))
cur_date = option_data['date'].unique()
scenes = len(cur_date)

for d_idx in range(scenes): #joint_baMOT

    init_date = cur_date[d_idx]
    matur_date = option_data[option_data['date'] == init_date]['exdate'].unique()

    dates = [datetime.datetime.strptime(ts, "%Y-%m-%d") for ts in matur_date]
    dates.sort()
    matur_date = [datetime.datetime.strftime(ts, "%Y-%m-%d") for ts in dates]
    matur_date = np.array(matur_date)

    t2_date = np.min([maturity_idx2, len(matur_date)-1])
    expdate = matur_date[[maturity_idx1, t2_date]]
    print('Date of data', init_date, expdate)

    px_1, X1, px_2, X2, _ = calibdensity.get_dist_ba(f_path='D:/2022_{}Call.csv'.format(ticker1),
                                                    init_date=init_date, expiry_date=expdate,
                                                    strikes_low=0.5, strikes_upper=1.5,
                                                    OI_thrs1=1, OI_thrs2=1, verbose=False)

    try:
        py_1, Y1, py_2, Y2, _ = calibdensity.get_dist_ba(f_path='D:/2022_{}Call.csv'.format(ticker2),
                                                         init_date=init_date, expiry_date=expdate,
                                                         strikes_low=0.5, strikes_upper=1.5,
                                                         OI_thrs1=1, OI_thrs2=1, verbose=False)
    except ValueError:
        print("Not on the same maturity, skipped")
        continue


    nx1 = X1.shape[0]
    nx2 = X2.shape[0]
    ny1 = Y1.shape[0]
    ny2 = Y2.shape[0]
    print('Problem size', nx1, nx2, ny1, ny2)

    cost = np.zeros((nx1, nx2, ny1, ny2))
    print('X1', X1)
    print('X2', X2)
    print('Y1', Y1)
    print('Y2', Y2)
    strike = int(((px_1*X1).sum() + (px_2*X2).sum() + (py_1*Y1).sum() + (py_2*Y2).sum())/4) # 50 # # 60 #
    print('Strike:', strike)
    for x1_idx in range(nx1):
        for x2_idx in range(nx2):
            for y1_idx in range(ny1):
                for y2_idx in range(ny2):
                    ## basket option
                    # cost[x1_idx, x2_idx, y1_idx, y2_idx] = (X1[x1_idx] + Y1[y1_idx] + X2[x2_idx] + Y2[y2_idx])
                    cost[x1_idx, x2_idx, y1_idx, y2_idx] = np.max([(X1[x1_idx]+Y1[y1_idx]+X2[x2_idx]+Y2[y2_idx])/4 - strike, 0])
                    ## spread option
                    # cost[x1_idx, x2_idx, y1_idx, y2_idx] = np.abs(X2[x2_idx] - Y2[y2_idx])

                    # if X1[x1_idx] + X2[x2_idx] + Y1[y1_idx] + Y2[y2_idx] < strike * 4*0.7:
                    #     cost[x1_idx, x2_idx, y1_idx, y2_idx] = 0.0
                    # elif X1[x1_idx] + X2[x2_idx] + Y1[y1_idx] + Y2[y2_idx] < strike * 4*0.9:
                    #     cost[x1_idx, x2_idx, y1_idx, y2_idx] = 10.0
                    # elif X1[x1_idx] + X2[x2_idx] + Y1[y1_idx] + Y2[y2_idx] < strike * 4*1.0:
                    #     cost[x1_idx, x2_idx, y1_idx, y2_idx] = 20.0
                    # elif X1[x1_idx] + X2[x2_idx] + Y1[y1_idx] + Y2[y2_idx] < strike * 4*1.1:
                    #     cost[x1_idx, x2_idx, y1_idx, y2_idx] = 25.0
                    # else:
                    #     cost[x1_idx, x2_idx, y1_idx, y2_idx] = 30.0

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
        objval_0 = causal_test(MAX_FLAG, cost, X1, X2, Y1, Y2, px_1, px_2, py_1, py_2,
                               McCor_Causal=False, McCor_Anticausal=False)


        objval_2 = causal_test(MAX_FLAG, cost, X1, X2, Y1, Y2, px_1, px_2, py_1, py_2,
                               McCor_Causal=True, McCor_Anticausal=True)

        print('MOT:', objval_0)
        print('McCormick:', objval_2)

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
print('MOT Max', MOT_max)

print('MOT Min', MOT_min)

ratios = np.divide(Mc_max - Mc_min, MOT_max - MOT_min)
print('Ratios:', ratios, ratios.mean())
print(len(ratios))
print(ratios.min())





log_dir = './{}_{}_logs'.format(ticker1, ticker2)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

res_df = pd.DataFrame({'time0': time0, 'time1': time1, 'time2': time2, 'strike': strike_hist,
                       'MOT_max': MOT_max, 'MOT_min': MOT_min,
                       'McCormick_max': Mc_max, 'McCormick_min': Mc_min,
                       'Ratios': ratios})

res_df.to_csv('{}/{}_{}.csv'.format(log_dir, ticker1, ticker2), index=False)
plt.figure()
plt.hist(ratios)
# plt.legend(loc='best', fontsize=15)
plt.savefig('{}/{}_{}.pdf'.format(log_dir, ticker1, ticker2), format='pdf',
            dpi=1000, bbox_inches='tight', pad_inches=0.1)




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

