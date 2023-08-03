import random
import math
import numpy as np
from OLH_EM import calMAE
def gen_matrix(e_epsilon,number):
    p = e_epsilon / (e_epsilon + number - 1)
    q = 1 / (e_epsilon + number - 1)
    # Initialize the matrix with all elements set to q
    matrix = np.full((number, number), q)

    # Set diagonal elements to 1/2
    np.fill_diagonal(matrix, p)
    return matrix

res = []
def EM_GR_reduct(ori,noised_report,num_GRR, e_epsilon):
    matrix = gen_matrix(e_epsilon,num_GRR)
    weights = np.ones(num_GRR) / (num_GRR)
    # Run EM algorithm
    tolerancelist = [0,10e-1,10e-1,10e-1,10e-2,10e-2]
    max_iter = 3000
    ll_old = -np.inf
    throd = 1 * np.sqrt((num_GRR-2+e_epsilon)/len(noised_report))/e_epsilon  # super-param
    # print("throd is", throd)
    times = 0
    weights_old = []
    BIC_old = 50000000
    count_para = num_GRR*2  # remove one component means delete two parameters, in GRR
    while True:
        times += 1
        for iter in range(max_iter):
            # E-step: compute expected values of latent variable z
            probs = weights * matrix[noised_report, :]
            probs /= probs.sum(axis=1, keepdims=True)

            # M-step: update mixture weights
            weights = probs.mean(axis=0)
            # Compute log-likelihood and check for convergence
            ll = np.sum(np.log(probs @ weights))
            if iter %5==0:
                print(iter, ll, calMAE(weights, ori))
                res.append(calMAE(weights,ori))
            if ll - ll_old < tolerancelist[times] and ll - ll_old >0:
                break
            ll_old = ll
            if times >= 2 and iter == 0:
                BIC_new = (count_para)*np.log(len(noised_report))-2*ll
                if BIC_new > BIC_old:
                    return weights_old # stop and return

        BIC_old = (count_para)*np.log(len(noised_report))-2*ll
        count = int(num_GRR/(2**times))
        templist = []
        for k in range(num_GRR):
            if weights[num_GRR-1-k]!=0 and weights[num_GRR-1-k] < throd:
                templist.append(weights[num_GRR-1-k])
        templist = sorted(templist)
        minV = min(len(templist),count)
        count_para -= 2*minV
        weights_old = np.copy(weights)
        for k in range(minV):
            for v in range(num_GRR):
                if templist[k]==weights[v]:
                    weights[v] = 0
        if times > 4:
            break
    # print("loglikelihood", ll)
    # print("BIC", (num_GRR)*np.log(len(noised_report))-2*ll)
    # print(res)
    return weights


if __name__ == "__main__":
    epsilon = 1
    n=10000
