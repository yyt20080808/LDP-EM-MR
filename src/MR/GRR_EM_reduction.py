import numpy as np
from utils import calMAE, calMSE
import random

def gen_matrix_merge(e_epsilon, number):
    p = e_epsilon / (e_epsilon + number - 1)
    q = 1 / (e_epsilon + number - 1)
    # Initialize the matrix with all elements set to q
    matrix = np.full((number, number), q)

    # Set diagonal elements to 1/2
    np.fill_diagonal(matrix, p)
    return matrix


res = []


def noisedP2hist(noised_report, num_GRR):
    a = [0 for i in range(num_GRR)]
    for v in noised_report:
        a[v] += 1
    return np.array(a)


def gen_new_matrix(e_epsilon, reflected_indexlist, number, count_reducted, old_ns_hist,oldindexlist):
    p = e_epsilon / (e_epsilon + number - 1)
    q = 1 / (e_epsilon + number - 1)
    new_matrix = np.full((number - count_reducted + 1, number - count_reducted + 1), q)

    # Set diagonal elements to 1/2
    np.fill_diagonal(new_matrix, p)
    new_ns_hist = np.ones(number - count_reducted + 1)

    new_matrix[:, -1] = count_reducted * q
    new_matrix[-1, -1] = p + (count_reducted - 1) * q

    new_indexlist = np.ones(number - count_reducted,dtype=int)
    i, j = 0, 0
    sums = 0
    # for ns_hist, 1. find the first no reducted index in the indexlist
    while i < number:  # find the first -1
        if reflected_indexlist[i] != -1:
            # 2. get the old_ns_hist value in
            new_ns_hist[j] = old_ns_hist[i]
            # reflected_indexlist[i] = j
            new_indexlist[j] = i # the newlist's j-th pos stores the original index i
            j += 1
        else:
            sums += old_ns_hist[i]
        i += 1
    new_ns_hist[number - count_reducted] = sums
    print("check:",reflected_indexlist)
    return new_ns_hist, new_matrix.T, new_indexlist


def EM_GRR_reduct_merge(ori, noised_report, num_GRR, e_epsilon,init_weight, topk=10):
    matrix = gen_matrix_merge(e_epsilon, num_GRR)
    real, maxtimes = 0, 1

    weights = np.ones(num_GRR) / (num_GRR)
    for i in range(num_GRR):
        if init_weight[i] > 0:
            weights[i] = init_weight[i]
        else:
            weights[i] = 0.00001
    ns_hist = noisedP2hist(noised_report, num_GRR)
    old_ns_hist = np.copy(ns_hist)
    tolerancelist = [1e-3, 1e-3, 1e-3, 1e-6 * e_epsilon]
    if len(noised_report) > 100000:
        tolerancelist = [i / 2 for i in tolerancelist]
        maxtimes = 1
    max_iter = [5000, 5000, 1000, 1000, 2000]
    ll_old = -np.inf
    throd = 4/ 3 * np.sqrt((num_GRR - 2 + e_epsilon) / len(noised_report)) / (e_epsilon - 1)
    if e_epsilon < 5:
        throd = 15/ 3 * np.sqrt((num_GRR - 2 + e_epsilon) / len(noised_report)) / (e_epsilon - 1)

    print("throd is:",throd)
    times = 0
    BIC_old = 50000000
    count_para = num_GRR * 2  # remove one component means delete two parameters, in GRR
    weights2 = np.copy(weights)
    reflect_indexlist = [i for i in range(num_GRR)]
    indexlist =[i for i in range(num_GRR)]
    count_merged = 0  # accumulate the reducted inputs
    # Run EM algorithm
    while True:
        times += 1
        for iter in range(max_iter[times - 1]):
            weights_old = np.copy(weights)
            X_condition = np.matmul(matrix, weights_old)
            TMP = matrix.T / X_condition
            P = np.copy(np.matmul(TMP, ns_hist))
            P = P * weights_old
            weights = np.copy(P / sum(P))


            ll = np.inner(ns_hist, np.log(np.matmul(matrix, weights)))
            # if iter % 20 == 0:
            #     # res.append(calMSE(weights,ori,topk))
            #     print(iter, ll)
            if ll - ll_old < tolerancelist[times - 1] and ll - ll_old > 0:
                break
            ll_old = ll
            # if times >= 1 and iter == 0:
            #     BIC_new = (count_para)*np.log(len(noised_report))-2*ll
            #     if BIC_new > BIC_old:
            #         return weights # stop and return
        print(times,"\t", iter, ll, weights[-1])
        BIC_old = (count_para) * np.log(len(noised_report)) - 2 * ll
        count = int(num_GRR / (2 ** (times)))
        templist = []
        ttt = len(weights)
        if times > 1:
            ttt -= 1
        for k in range(ttt):
            if weights[k] == 0 or weights[k] < throd:
                templist.append(weights[k])
        templist = sorted(templist)
        random.shuffle(templist)
        minV = min(len(templist), count)
        count_para -= 2 * minV
        for k in range(minV):
            for v in range(topk+5,ttt):
                if templist[k] == weights[v]:
                    weights[v] = 0
                    reflect_indexlist[indexlist[v]] = -1
                    count_merged += 1

        if times > maxtimes or count_merged == 0:
            break
        ns_hist, matrix, indexlist = gen_new_matrix(e_epsilon, reflect_indexlist, num_GRR, count_merged, old_ns_hist,indexlist)
        weights2 = np.copy(weights)
        weights = np.ones(num_GRR - count_merged + 1)
        v = 0
        for k in range(len(weights2)):
            if weights2[k] > 0:
                weights[v] = weights2[k]
                v += 1
        weights[-1] = 2-sum(weights)
    weights2 = np.zeros(num_GRR)
    # print("\n weights:", weights)
    i = 0
    while i < len(weights)-1:
        weights2[indexlist[i]] =weights[i]
        i+=1
    i = 0

    while i < num_GRR:
        if weights2[i]==0 and  count_merged>0:
            weights2[i] = weights[-1]/count_merged
        i+=1
    print("\t uniform:", weights[-1],count_merged,calMSE(weights2,ori,topk))
    return weights2
