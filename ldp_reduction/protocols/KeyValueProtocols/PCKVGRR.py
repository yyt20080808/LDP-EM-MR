import math
import numpy as np
import random
from .PCKVUE import rounding
# 这种方案不需要任何的分组，假设有两个key,那么用duchi就将其分成6组，pckv——GRR
def pckv_GRR(data, epsilonls,mul_d):
    # print("this is pckv_Grr")
    reslist_f = []
    reslist_v = []
    real_f = len(data[0])/(len(data[0])+len(data[1]))
    for epsilon in epsilonls:
        GRR_epsilon = math.e ** (epsilon)
        mse_f, mse_v = 0, 0
        for times in range(100):
            sum_average_all = [sum(i) / len(i) for i in data]
            n = 0
            for users in data:
                n += len(users)
            # for dataIndex in range(data_types):
            res_all_x = [0, 0, 0]
            for dataIndex in range(1):
                for v in data[0]:
                    round_v = rounding(v)
                    if round_v > 0:
                        round_v = dataIndex * 2
                    else:
                        round_v = dataIndex * 2 + 1
                    index = perturb(round_v, mul_d * 2, GRR_epsilon)
                    if index >= 2:
                        res_all_x[2] += 1
                    else:
                        res_all_x[index] += 1
                for v in data[1]:
                    round_v = rounding(0)
                    if round_v > 0:
                        round_v = 1 * 2
                    else:
                        round_v = 1 * 2 + 1
                    index = perturb(round_v, mul_d * 2, GRR_epsilon)
                    if index >= 2:
                        res_all_x[2] += 1
                    else:
                        res_all_x[index] += 1
            estimate_f,estimate_v = revise(res_all_x, GRR_epsilon, mul_d * 2, n)
            # n1 = res_revise_x[0]
            # n2 = res_revise_x[1]
            # estimate_v = (n1 - n2) / (n1 + n2)
            # estimate_f = n1+n2
            if estimate_v > 1:
                estimate_v = 1
            elif estimate_v < -1:
                estimate_v = -1
            # print((n1+n2)/n)
            # estimate_v[index_revise] = (n1 - n2) / n / (1/data_types)
            # print(estimate_v, sum_average_all[0])
            mse_v += (estimate_v - sum_average_all[0]) ** 2
            mse_f += (estimate_f - real_f)**2
        reslist_f.append(mse_f/100)
        reslist_v.append(mse_v/100)
    return reslist_f,reslist_v

def perturb(v, all_length, e_epsilon):
    p_true = e_epsilon / (e_epsilon + all_length - 1)
    if random.random() < p_true:
        return v
    pro = random.random()
    s = 1 / (all_length - 1)
    b = int(pro / s)
    if b >= v:
        return b + 1
    return b


def revise(input_value, e_epsilon, D, n):
    p = e_epsilon / (e_epsilon + D - 1)
    q = 1 / (e_epsilon + D - 1)
    n1 = (input_value[0] - q * n) / (p - q)
    n2 = (input_value[1] - q * n) / (p - q)
    return (n1+n2)/n, (n1 - n2) / (n1 + n2)