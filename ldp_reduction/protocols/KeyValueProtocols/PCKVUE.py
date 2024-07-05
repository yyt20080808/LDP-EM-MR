import numpy as np
import math
import random


def perturb_ue(v,p1,p2):
    if v > 0:
        p = random.random()
        if p < p1:
            return 0
        elif p <= 0.5:
            return 1
        return 2
    elif v < 0:
        p = random.random()
        if p < p1:
            return 1
        elif p <= 0.5:
            return 0
        return 2
def perturb_ue2(p3,p4):
    p = random.random()
    if p < p4:
        return 0
    elif p < 2*p4:
        return 1
    return 2

def pckv_UE(data, epsilonls):
    # print("this is pckv UE,real_f is",len(data[0])/(len(data[0])+len(data[1])))
    reslist_f = []
    reslist_v = []
    real_f = len(data[0]) / (len(data[0]) + len(data[1]))
    for epsilon in epsilonls:
        GRR_epsilon = math.e ** epsilon
        mse_f,mse_v = 0,0
        for times in range(100):
            sum_average_all = [sum(i) / len(i) for i in data]
            n = 0
            for users in data:
                n += len(users)
            # for dataIndex in range(data_types):
            res_all_x = [0, 0, 0]
            p1 = GRR_epsilon / (1+GRR_epsilon)/2
            p2 = 1 / (1+GRR_epsilon)/2
            p3 = (GRR_epsilon+1) / (GRR_epsilon+3)
            p4 = 1 / (GRR_epsilon + 3)
            for dataIndex in range(1):
                for v in data[0]:
                    round_v = rounding(v)
                    index = perturb_ue(round_v,p1,p2)
                    res_all_x[index] += 1
                for v in data[1]:
                    index = perturb_ue2(p3,p4)
                    res_all_x[index] += 1
            estimate_f,estimate_v = revise_ue(n,res_all_x[0], res_all_x[1],GRR_epsilon)
            # estimate_v = 0
            # index_revise = 0
            if estimate_f < 0.001:
                estimate_f = 0.001
            if estimate_v > 1:
                estimate_v = 1
            elif estimate_v < -1:
                estimate_v = -1
            # print((n1+n2)/n)
            # estimate_v[index_revise] = (n1 - n2) / n / (1/data_types)
            # print(estimate_v,sum_average_all[0])
            mse_f += (estimate_f - real_f)**2
            mse_v += (estimate_v - sum_average_all[0]) ** 2
            # index_revise+=1
        reslist_f.append(mse_f / 100)
        reslist_v.append(mse_v / 100)
    # print(reslist)
    return reslist_f,reslist_v

def revise_ue(n,n1,n2,Grr_ep):
    a = 1/ 2
    b = 1/ ((Grr_ep+1)/2+1)
    p = Grr_ep / (Grr_ep+1)
    return ((n1+n2)/n-b)/(a-b),(n1-n2)*(a-b)/a/(2*p-1)/(n1+n2-n*b)

def rounding(v):
    pro = (1 + v) / 2
    if random.random() < pro:
        return 1
    else:
        return -1