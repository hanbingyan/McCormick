import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gurobipy import *

def optimize_ba(strikes1, best_bid1, best_ask1, rf1, expire1, fwd_price1,
                strikes2, best_bid2, best_ask2, rf2, expire2, fwd_price2, verbose=False):
    m = Model('Primal')
    if verbose:
        m.setParam('OutputFlag', 1)
        m.setParam('LogToConsole', 1)
    else:
        m.setParam('OutputFlag', 0)
        m.setParam('LogToConsole', 0)
    # m.setParam('Method', 0)
    # m.setParam('MIPGap', 1e-9)
    m.setParam('FeasibilityTol', 5e-9)
    # m.setParam('NumericFocus', 3)

    scaled_strikes1 = strikes1/fwd_price1
    scaled_strikes2 = strikes2/fwd_price2

    nx1 = scaled_strikes1.shape[0]
    nx2 = scaled_strikes2.shape[0]
    if verbose:
        print('Problem size', nx1, nx2)
    pix1 = m.addVars(nx1, lb=0.0, ub=1.0, name='transportx1')
    pix2 = m.addVars(nx2, lb=0.0, ub=1.0, name='transportx2')
    pi = m.addVars(nx1, nx2, lb=0.0, ub=1.0, name='transport')
    # scaled price = price/discounter/forward_price
    scaled_price1 = m.addVars(nx1, lb=0.0, ub=1.0, name='scaled_price1')
    scaled_price2 = m.addVars(nx2, lb=0.0, ub=1.0, name='scaled_price2')

    bid_gap1 = m.addVars(nx1, lb=0.0, ub =100.0, name='bid_gap1')
    ask_gap1 = m.addVars(nx1, lb=0.0, ub=100.0, name='ask_gap1')
    bid_gap2 = m.addVars(nx2, lb=0.0, ub=100.0, name='bid_gap2')
    ask_gap2 = m.addVars(nx2, lb=0.0, ub=100.0, name='ask_gap2')

    for k1_idx in range(nx1):
        # pricing formula, stock prices supported at strikes
        m.addConstr(quicksum((scaled_strikes1[j] - scaled_strikes1[k1_idx])*pix1[j] for j in
                             range(k1_idx, nx1)) == scaled_price1[k1_idx],
                    name='pricing_x1_'+str(k1_idx))

    # marginal 1 sums to one
    m.addConstr(pix1.sum('*') == 1.0, name='marginal1')
    # mean value equals to forward
    m.addConstr(quicksum([pix1[i]*strikes1[i] for i in range(nx1)]) == fwd_price1, name='mean')

    for k2_idx in range(nx2):
        # pricing formula at x2
        m.addConstr(quicksum((scaled_strikes2[j] - scaled_strikes2[k2_idx]) * pix2[j] for j in
                             range(k2_idx, nx2)) == scaled_price2[k2_idx], name='pricing_x2_' + str(k2_idx))


    # marginal 2 sums to one
    m.addConstr(pix2.sum('*') == 1.0, name='marginal2')
    # mean value equals to forward
    m.addConstr(quicksum([pix2[i]*strikes2[i] for i in range(nx2)]) == fwd_price2, name='mean')

    ## marginal constraints
    m.addConstrs((pi.sum(i, '*') == pix1[i] for i in range(nx1)), name='x1_marginal')
    m.addConstrs((pi.sum('*', i) == pix2[i] for i in range(nx2)), name='x2_marginal')

    # joint martingale constraints
    m.addConstrs(((quicksum(pi[i, j]*scaled_strikes2[j] for j in range(nx2)) == scaled_strikes1[i]*pi.sum(i, '*'))
                  for i in range(nx1)), name='martingale_at_x1')


    # Reformulate the absolute value as a linear constraint
    m.addConstrs(((scaled_price1[i]*fwd_price1 - best_ask1[i]*np.exp(rf1*expire1) <= ask_gap1[i]) for i in range(nx1)),
                 name='ask_gap11')
    m.addConstrs(((scaled_price1[i]*fwd_price1 - best_ask1[i]*np.exp(rf1*expire1) >= -ask_gap1[i]) for i in range(nx1)),
                 name='ask_gap12')

    m.addConstrs(((best_bid1[i]*np.exp(rf1*expire1) - scaled_price1[i]*fwd_price1 >= -bid_gap1[i]) for i in range(nx1)),
                 name='bid_gap11')
    m.addConstrs(((best_bid1[i]*np.exp(rf1*expire1) - scaled_price1[i]*fwd_price1 <= bid_gap1[i]) for i in range(nx1)),
                 name='bid_gap11')

    m.addConstrs(((scaled_price2[i]*fwd_price2 - best_ask2[i]*np.exp(rf2*expire2) >= -ask_gap2[i]) for i in range(nx2)),
                 name='ask_gap21')
    m.addConstrs(((scaled_price2[i]*fwd_price2 - best_ask2[i]*np.exp(rf2*expire2) <= ask_gap2[i]) for i in range(nx2)),
                 name='ask_gap22')

    m.addConstrs(((best_bid2[i]*np.exp(rf2*expire2) - scaled_price2[i]*fwd_price2 >= -bid_gap2[i]) for i in range(nx2)),
                 name='bid_gap21')
    m.addConstrs(((best_bid2[i]*np.exp(rf2*expire2) - scaled_price2[i]*fwd_price2 <= bid_gap2[i]) for i in range(nx2)),
                 name='bid_gap22')

    obj = quicksum([ask_gap1[i] for i in range(nx1)])
    obj += quicksum([bid_gap1[i] for i in range(nx1)])
    obj += quicksum([ask_gap2[i] for i in range(nx2)])
    obj += quicksum([bid_gap2[i] for i in range(nx2)])

    m.setObjective(obj, GRB.MINIMIZE)

    m.optimize()
    # m.printQuality()

    # m.computeIIS()
    # m.write('infeasible.ilp')

    names_to_retrieve = (f"transportx1[{i}]" for i in range(nx1))
    px1 = np.array([m.getVarByName(name).X for name in names_to_retrieve])
    px1 = px1.reshape((nx1))

    names_to_retrieve = (f"transportx2[{i}]" for i in range(nx2))
    px2 = np.array([m.getVarByName(name).X for name in names_to_retrieve])
    px2 = px2.reshape((nx2))

    names_to_retrieve = (f"transport[{i},{j}]" for i in range(nx1) for j in range(nx2))
    px = np.array([m.getVarByName(name).X for name in names_to_retrieve])
    px = px.reshape((nx1, nx2))

    names_to_retrieve = (f"scaled_price1[{i}]" for i in range(nx1))
    s_price1 = np.array([m.getVarByName(name).X for name in names_to_retrieve])
    s_price1 = s_price1.reshape((nx1))

    names_to_retrieve = (f"scaled_price2[{i}]" for i in range(nx2))
    s_price2 = np.array([m.getVarByName(name).X for name in names_to_retrieve])
    s_price2 = s_price2.reshape((nx2))


    price1 = s_price1/np.exp(rf1*expire1)*fwd_price1
    price2 = s_price2/np.exp(rf2*expire2)*fwd_price2



    if verbose:
        print('Marginal 1 price violation:',
              np.maximum(best_bid1 - price1, 0).sum() + np.maximum(price1 - best_ask1, 0).sum())

        print('Marginal 2 price violation:',
              np.maximum(best_bid2 - price2, 0).sum() + np.maximum(price2 - best_ask2, 0).sum())

        plt.figure()
        plt.plot(strikes1, price1-best_bid1, label='Bid 1')
        plt.plot(strikes1, price1-best_ask1, label='Ask 1')
        # plt.plot(strikes1, best_ask1, label='Ask 1')
        plt.legend(loc='best')
        plt.show()

        plt.figure()
        plt.plot(strikes2, price2-best_bid2, label='Bid 2')
        # plt.plot(strikes2, price2, label='Smoothed price 2')
        plt.plot(strikes2, price2-best_ask2, label='Ask 2')
        plt.legend(loc='best')
        plt.show()

        plt.figure()
        plt.plot(strikes1, price1, label='Price 1')
        plt.legend(loc='best')
        plt.show()

        plt.figure()
        plt.plot(strikes2, price2, label='Price 2')
        plt.legend(loc='best')
        plt.show()

        print(price1)
        print(price2)
    return px1, px2, px





##### data loader #####
def get_dist_ba(f_path, init_date, expiry_date, strikes_low=0.5, strikes_upper=1.5,
                OI_thrs1=1, OI_thrs2=1, verbose=False):

    option_data = pd.read_csv(f_path)

    option_data = option_data[(option_data['date'] == init_date) & option_data['exdate'].isin(expiry_date)]
    option_data = option_data[option_data['impl_volatility']>0.01]
    option_data = option_data.reset_index(drop=True)
    option_data['rf_rate'] /= 100
    option_data['maturity'] /= 365.0

    avg_fwd = option_data['forward'].mean()
    option_data = option_data[(option_data['strike_price'] >= avg_fwd*strikes_low) &
                              (option_data['strike_price'] <= avg_fwd*strikes_upper)]

    option_data = option_data.sort_values(by=['strike_price'], ascending=[True])
    option_data = option_data.reset_index(drop=True)

    marginal1 = option_data[option_data['exdate'] == expiry_date[0]]
    marginal1 = marginal1[marginal1['open_interest'] >= OI_thrs1]
    marginal1 = marginal1.reset_index(drop=True)

    marginal2 = option_data[option_data['exdate'] == expiry_date[1]]
    marginal2 = marginal2[marginal2['open_interest'] >= OI_thrs2]
    marginal2 = marginal2.reset_index(drop=True)

    if len(marginal1) == 0 or len(marginal2) == 0:
        raise ValueError('No data on these dates.')

    strikes1 = marginal1['strike_price'].values
    best_bid1 = marginal1['best_bid'].values
    best_ask1 = marginal1['best_offer'].values
    rf1 = marginal1['rf_rate'].values[0]
    expire1 = marginal1['maturity'].values[0]
    fwd_price1 = marginal1['forward'].values[0]

    strikes2 = marginal2['strike_price'].values
    best_bid2 = marginal2['best_bid'].values
    best_ask2 = marginal2['best_offer'].values
    rf2 = marginal2['rf_rate'].values[0]
    expire2 = marginal2['maturity'].values[0]
    fwd_price2 = marginal2['forward'].values[0]

    ### add prices with strikes = 0 or very high number
    strikes1 = np.concatenate((np.array([0.0]), strikes1, np.array([strikes1.max()+50.0])))
    best_bid1 = np.concatenate((np.array([np.exp(-rf1*expire1)*fwd_price1]), best_bid1, np.array([0.0])))
    best_ask1 = np.concatenate((np.array([np.exp(-rf1*expire1)*fwd_price1]), best_ask1, np.array([best_ask1.min()])))

    strikes2 = np.concatenate((np.array([0.0]), strikes2, np.array([strikes2.max()+50.0])))
    best_bid2 = np.concatenate((np.array([np.exp(-rf2*expire2)*fwd_price2]), best_bid2, np.array([0.0])))
    best_ask2 = np.concatenate((np.array([np.exp(-rf2*expire2)*fwd_price2]), best_ask2, np.array([best_ask2.min()])))



    px1, px2, px = optimize_ba(strikes1, best_bid1, best_ask1, rf1, expire1, fwd_price1,
                               strikes2, best_bid2, best_ask2, rf2, expire2, fwd_price2, verbose)

    ## px is the joint density
    nonzero1 = np.where(px1>0.0)
    zeros1 = np.where(px1 == 0.0)
    px = np.delete(px, obj=zeros1, axis=0)

    px1 = px1[nonzero1]
    strikes1 = strikes1[nonzero1]

    nonzero2 = np.where(px2>0.0)
    zeros2 = np.where(px2 == 0.0)
    px = np.delete(px, obj=zeros2, axis=1)

    px2 = px2[nonzero2]
    strikes2 = strikes2[nonzero2]


    if verbose:
        plt.figure()
        plt.plot(strikes1, px1, label='Probability Marginal 1')
        plt.legend(loc='best')
        plt.show()

        plt.figure()
        plt.plot(strikes2, px2, label='Probability Marginal 2')
        plt.legend(loc='best')
        plt.show()

    return px1, strikes1, px2, strikes2, px

if __name__ == "__main__":

    option_data = pd.read_csv('./data/2022_GSKCall.csv')
    cur_date = option_data['date'].unique()
    init_date = cur_date[2]
    matur_date = option_data[option_data['date'] == init_date]['exdate'].unique()
    expdate = matur_date[[0, 1]]
    print('Date of data', init_date, expdate)
    px_1, X1, px_2, X2, px = get_dist_ba(f_path='./data/2022_GSKCall.csv',
                                         init_date=init_date, expiry_date=expdate,
                                         strikes_low=0.1, strikes_upper=2.0,
                                         OI_thrs1=1, OI_thrs2=1, verbose=True)


    print(X1)
    print(px_1, px_1.shape, px_1.sum())
    print(X2)
    print(px_2, px_2.shape, px_2.sum())

