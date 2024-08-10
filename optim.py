from gurobipy import *

##### The main optimizer ######
def mcoptim(MAX, cost, X1, X2, Y1, Y2, px_1, px_2, py_1, py_2,
            McCor_Causal, McCor_Anticausal, Causal=False, Anticausal=False, cons=False, thres=0.01, verbose=False):
    m = Model('Primal')
    if verbose:
        m.setParam('OutputFlag', 1)
        m.setParam('LogToConsole', 1)
    else:
        m.setParam('OutputFlag', 0)
        m.setParam('LogToConsole', 0)

    nx1 = X1.shape[0]
    nx2 = X2.shape[0]
    ny1 = Y1.shape[0]
    ny2 = Y2.shape[0]

    ### Forward price ###
    fwdx1 = (X1*px_1).sum()
    fwdx2 = (X2*px_2).sum()
    fwdy1 = (Y1*py_1).sum()
    fwdy2 = (Y2*py_2).sum()

    # joint density within X or Y
    pix = m.addVars(nx1, nx2, lb=0.0, ub=1.0, name='transportx')
    piy = m.addVars(ny1, ny2, lb=0.0, ub=1.0, name='transporty')
    # marginal constraints
    m.addConstrs((pix.sum(i, '*') == px_1[i] for i in range(nx1)), name='x1_marginal')
    m.addConstrs((pix.sum('*', i) == px_2[i] for i in range(nx2)), name='x2_marginal')
    m.addConstrs((piy.sum(i, '*') == py_1[i] for i in range(ny1)), name='y1_marginal')
    m.addConstrs((piy.sum('*', i) == py_2[i] for i in range(ny2)), name='y2_marginal')

    ### joint density of X and Y
    if cons:
        pi = m.addVars(nx1, nx2, ny1, ny2, lb=0.0, ub=0.01, name='transport')
        m.addConstrs((pi[i, j, k, l] >= thres * min([px_1[i], px_2[j], py_1[k], py_2[l]])
                      for i in range(0, nx1, 3)
                      for j in range(0, nx2, 3)
                      for k in range(0, ny1, 3)
                      for l in range(0, ny2, 3)), name='marginal')
    else:
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
    if Causal:
        for x1_idx in range(nx1):
            for x2_idx in range(nx2):
                for y1_idx in range(ny1):
                    m.addConstr(px_1[x1_idx] * pi.sum(x1_idx, x2_idx, y1_idx, '*') ==
                                pi.sum(x1_idx, '*', y1_idx, '*') * pix[x1_idx, x2_idx],
                                name='anticausal' + '_' + str(x1_idx) + '_' + str(x2_idx) + '_' + str(y1_idx))


    if Anticausal:
        for y1_idx in range(ny1):
            for y2_idx in range(ny2):
                for x1_idx in range(nx1):
                    m.addConstr(py_1[y1_idx] * pi.sum(x1_idx, '*', y1_idx, y2_idx) ==
                                pi.sum(x1_idx, '*', y1_idx, '*') * piy[y1_idx, y2_idx],
                                name='causal' + '_' + str(y1_idx) + '_' + str(y2_idx) + '_' + str(x1_idx))




    obj = quicksum([cost[i,j,k,l] * pi[i,j,k,l] for i in range(nx1) for j in range(nx2) for k in range(ny1) for l in
                    range(ny2)])

    if MAX:
        m.setObjective(obj, GRB.MAXIMIZE)
    else:
        m.setObjective(obj, GRB.MINIMIZE)


    if Causal or Anticausal:
        m.params.NonConvex = 2
        m.setParam('MIPGap', 1e-3)
    # print('Maximizing the exotic option...')

    m.optimize()
    m.printQuality()

    # names_to_retrieve = (f"transport[{i},{j},{k},{l}]" for i in range(nx1) for j in range(nx2) for k in range(ny1) for l
    #                      in range(ny2))
    # coupling = np.array([m.getVarByName(name).X for name in names_to_retrieve])
    # for ans in m.getVars():
    #   print('%s %g' % (ans.VarName, ans.X))

    # coupling = coupling.reshape((nx1, nx2, ny1, ny2))

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