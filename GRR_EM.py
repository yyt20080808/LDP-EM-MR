import random
import math
import numpy as np
from GRR_EM_reduction import EM_GR_reduct,gen_matrix
from OLH_EM import calMAE
def GRR(e_epsilon, input_v, total):
    p_true = e_epsilon / (e_epsilon + total - 1)
    pro = random.random()
    if pro < p_true:
        return input_v
    else:
        v = np.random.randint(0, total-1)
        if v>=input_v:
            v += 1
        return v

def GRR_revision_no(noised_reports, number_GRR, epsilon):
    e_epsilon = math.e**epsilon
    p_true = e_epsilon / (e_epsilon + number_GRR - 1)
    q = 1 / (e_epsilon + number_GRR - 1)
    res =[]
    n = len(noised_reports)
    ns_hist,_ = np.histogram(noised_reports, bins=number_GRR, range=(0, number_GRR-1))
    for i in range(number_GRR):
        xv = (ns_hist[i] / n - q) / (p_true - q)
        res.append(xv)
    return res

def GRR_revision(noised_reports,number_GRR,epsilon):
    e_epsilon = math.e**epsilon
    p_true = e_epsilon / (e_epsilon + number_GRR - 1)
    q = 1 / (e_epsilon + number_GRR - 1)
    res =[]
    n = len(noised_reports)
    ns_hist,_ = np.histogram(noised_reports, bins=number_GRR, range=(0, number_GRR-1))
    for i in range(number_GRR):
        xv = (ns_hist[i] / n - q) / (p_true - q)
        if xv <0.01:
            xv = 0
        res.append(xv)
    return res

def norm_sub(est_value_list, user_num , tolerance = 1):
    np_est_value_list = np.array(est_value_list)
    estimates = np.copy(np_est_value_list)

    while (np.fabs(sum(estimates) - user_num) > tolerance) or (estimates < 0).any():
        if (estimates <= 0).all():
            estimates[:] = user_num / estimates.size
            break
        estimates[estimates < 0] = 0
        total = sum(estimates)
        mask = estimates > 0
        diff = (user_num - total) / sum(mask)
        estimates[mask] += diff

    return estimates


res = []
def EM_GR(ori,noised_report,num_GRR, e_epsilon,ifcut = 0):
    matrix = gen_matrix(e_epsilon,num_GRR)
    weights = np.ones(num_GRR-ifcut) / (num_GRR-ifcut)
    # Run EM algorithm
    tolerance = 1e-2
    max_iter = 10000
    ll_old = -np.inf
    for iter in range(max_iter):
        # E-step: compute expected values of latent variable z
        probs = weights * matrix[noised_report, :num_GRR - ifcut]
        # probs /= probs.sum(axis=1, keepdims=True)
        # for j in range((num_GRR-ifcut)):
        #     # for index in range(len(noised_report)):
        #     probs[:, j] = weights[j] * matrix[noised_report,j]
        probs /= probs.sum(axis=1, keepdims=True)

        # M-step: update mixture weights
        weights = probs.mean(axis=0)

        # Compute log-likelihood and check for convergence
        ll = np.sum(np.log(probs @ weights))
        if ll - ll_old < tolerance:
            break
        ll_old = ll
        if iter % 5 == 0:
            weights1 = np.concatenate((weights,np.array([0 for i in range(ifcut)])),axis=0)
            res.append(calMAE(weights1, ori))
            print(iter,ll,calMAE(weights1,ori))
    print("loglikelihood", ll)
    print("BIC", (num_GRR-ifcut)*np.log(len(noised_report))-2*ll)
    print(res)
    weights1 = np.concatenate((weights,np.array([0 for i in range(ifcut)])),axis=0)
    return weights1

def generate_noised_data(epsilon):
    # SFC dataset
    # ori = np.array([2418, 2706, 731, 3998, 836, 3595, 979, 1526, 461, 1255, 1136, 988, 1881, 1225, 1189, 1664, 247, 2436, 274, 1363, 1021, 1104, 85, 1052, 371, 410, 200, 810,
    #                 442, 784, 265, 633, 18, 367, 497, 11, 110, 254, 194, 362, 180, 237, 116, 147, 50, 7, 15, 84,
    #                 19, 5, 16, 3, 11, 19, 10, 16, 29,3,5,7])
    ori = np.array([5000,2000,3000,2000,1200,500,10,3,0,80,
                    50,0,6,0,12,0,9,0,0,0,
                    0,0,0,5,20,75,4,12,3,5])
    # uniform dataset
    # ori = np.array([1000 for i in range(50)])
    # Income dataset
    # ori = np.array([20334, 13023, 23003, 20312, 15800, 18972, 15900, 18901, 20100,
    #           24221, 20321, 17892, 21004, 13501, 9084, 21031, 14023, 22213,
    #           2344, 0, 0, 10, 0, 0, 4, 0, 0, 0, 2, 0, 4, 80, 3, 0, 74, 0, 0, 0, 0,
    #           0, 0, 0, 0, 33, 0, 0, 323, 0, 0, 0, 21, 0, 0, 3, 0, 5, 213, 2, 3, 4, 23, 0, 0, 0, 0, 23, 23, 123, 11, 234,
    #           9, 2, 1, 4, 5, 2, 3, 2,
    #           2, 34, 1,2,5,6,3,2,0,0,231,0,3,4,5,6,7,43,2])

    number = len(ori)
    # print(len(ori),np.sum(ori))
    # print(ori[0:15]/np.sum(ori))
    noised_data = []
    for i in range(len(ori)):
        for j in range(ori[i]):
            noised_data.append(GRR(math.e**epsilon,i,number))
    noised_data = np.array(noised_data)
    # print(len(noised_data))
    return noised_data,number,ori/np.sum(ori)


if __name__ == "__main__":
    epsilon = 1
    acc1 = []
    acc2 = []
    acc3 = []
    acc4 = []
    acc5 = []
    exper_time = 1

    for i in range(exper_time):
        noised_data,number,ori = generate_noised_data(epsilon)
        weights = EM_GR(ori,noised_data,number,math.e**epsilon)
        res_EM = calMAE(weights,ori)
        acc1.append(res_EM)
        print(weights)
        weights2 = EM_GR_reduct(ori,noised_data, number, math.e ** epsilon)
        res_EMR = calMAE(weights2, ori)
        acc2.append(res_EMR)
        print(weights2)
        weights3 = GRR_revision(noised_data,number,epsilon)
        res_ttt = calMAE(weights3, ori)
        acc3.append(res_ttt)
        print(weights3)
        weights4 = GRR_revision_no(noised_data, number, epsilon)
        res_normal = calMAE(weights4, ori)
        acc4.append(res_normal)
        print(weights4)
        weights5 = norm_sub(np.array(weights4)*300000,300000)
        acc5.append(calMAE(weights5/300000, ori))
        print(weights5)


    # print(acc1)
    # print(acc2)
    # print(acc3)
    # print(acc4)
    # print(acc5)
    print("MAE")
    print("GRR-EM:",sum(acc1) / exper_time)
    print("GRR-MR:",sum(acc2)/exper_time)
    print("GRR-Basecut:",sum(acc3) / exper_time)
    print("GRR-Nonprocess:",sum(acc4) / exper_time)
    print("GRR-NormSub:",sum(acc5) / exper_time)