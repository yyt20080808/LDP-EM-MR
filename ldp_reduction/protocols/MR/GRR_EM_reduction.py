import numpy as np
# from ..categoricalProtocols.utils import calMAE, calMSE
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


def gen_new_matrix(e_epsilon, reflected_indexlist, number, count_reducted_list, old_ns_hist,
                   lastime_ns_hist, times):
    count_reducted = sum(count_reducted_list)
    p = e_epsilon / (e_epsilon + number - 1)
    q = 1 / (e_epsilon + number - 1)
    new_matrix = np.full((number - count_reducted + times, number - count_reducted + times), q)

    # Set diagonal elements to p
    np.fill_diagonal(new_matrix, p)
    new_ns_hist = np.ones(number - count_reducted + times)
    for i in range(times):
        new_matrix[:, -(times - i)] = count_reducted_list[i] * q
        new_matrix[-(times - i), -(times - i)] = p + (count_reducted_list[i] - 1) * q

    new_indexlist = np.ones(number - count_reducted, dtype=int)
    i, j = 0, 0
    sums = 0
    # for ns_hist, 1. find the first no reducted index in the indexlist
    while i < number:  # find the first -1
        if reflected_indexlist[i] >= 0:
            # 2. get the old_ns_hist value in
            new_ns_hist[j] = old_ns_hist[i]
            # reflected_indexlist[i] = j
            new_indexlist[j] = i  # the newlist's j-th pos stores the original index i
            j += 1
        elif reflected_indexlist[i] == 0 - times:
            sums += old_ns_hist[i]
        i += 1
    for i in range(times, 1, -1):
        new_ns_hist[-i] = lastime_ns_hist[-i + 1]
    new_ns_hist[number - count_reducted+times-1] = sums
    # print("check:", times,reflected_indexlist)

    # print("sum:", sum(new_ns_hist), new_ns_hist[-3:])
    return new_ns_hist, new_matrix.T, new_indexlist


def EM_GRR_reduct_merge( noised_report, num_GRR, e_epsilon, topk=10):
    matrix = gen_matrix_merge(e_epsilon, num_GRR)
    maxtimes = 2

    weights = np.ones(num_GRR) / (num_GRR)

    ns_hist = noisedP2hist(noised_report, num_GRR)
    old_ns_hist = np.copy(ns_hist)
    tolerancelist = [1e-3 * e_epsilon, 1e-3 * e_epsilon, 1e-3 * e_epsilon]
    reductNum = [int(num_GRR/4),int(num_GRR/4),int(num_GRR/8),int(num_GRR/8)]
    max_iter = [int(1000+num_GRR*10), int(1000+num_GRR*10), 1000, 1000, 1000]
    ll_old = -np.inf
    throd = 2 * np.sqrt((num_GRR - 2 + e_epsilon) / len(noised_report)) / (e_epsilon - 1)
    if e_epsilon < 5:
        # maxtimes=1
        throd = 1 * np.sqrt((num_GRR - 2 + e_epsilon) / len(noised_report)) / (e_epsilon - 1)

    times = 0
    BIC_old = 50000000
    count_para = num_GRR * 2  # remove one component means delete two parameters, in GRR
    weights2 = np.copy(weights)
    reflect_indexlist = [i for i in range(num_GRR)]
    indexlist = [i for i in range(num_GRR)]
    count_merged = [0 for i in range(maxtimes)]  # count the merged components in each iteration
    # Run EM algorithm
    while True:
        times += 1
        for _ in range(max_iter[times - 1]):
            weights_old = np.copy(weights)
            X_condition = np.matmul(matrix, weights_old)
            TMP = matrix.T / X_condition
            P = np.copy(np.matmul(TMP, ns_hist))
            P = P * weights_old
            weights = np.copy(P / sum(P))

            ll = np.inner(ns_hist, np.log(np.matmul(matrix, weights)))
            if ll - ll_old < tolerancelist[times - 1] and ll - ll_old > 0:
                break
            ll_old = ll
            if times >= 1 and iter == 0:
                BIC_new = (count_para)*np.log(len(noised_report))-2*ll
                if BIC_new > BIC_old:
                    return weights # stop and return
        # print(times, "\t", iter, ll, weights[-3:])
        BIC_old = (count_para) * np.log(len(noised_report)) - 2 * ll
        count = reductNum[times-1]
        templist = []
        ttt = len(weights)
        if times > 1:
            ttt -= (times-1)
        for k in range(ttt):
            if weights[k] == 0 or weights[k] < throd:
                templist.append(weights[k])
        templist = sorted(templist)
        # random.shuffle(templist)
        minV = min(len(templist), count)
        count_para -= 4 * minV
        if times > maxtimes:
            break
        for k in range(minV):
            for v in range(topk, ttt):
                if templist[k] == weights[v]:
                    weights[v] = 0
                    reflect_indexlist[indexlist[v]] = 0 - times
                    count_merged[times - 1] += 1

        if count_merged[times - 1] == 0:
            break
        ns_hist, matrix, indexlist = gen_new_matrix(e_epsilon, reflect_indexlist, num_GRR, count_merged,
                                                    old_ns_hist, ns_hist, times)
        weights2 = np.copy(weights)
        weights = np.ones(num_GRR - sum(count_merged) + times)
        v = 0
        for k in range(len(weights2)):
            if weights2[k] > 0:
                weights[v] = weights2[k]
                v += 1
        while v < len(weights):
            weights[v] =1/num_GRR
            v+=1
    weights2 = np.zeros(num_GRR)
    # print("\n weights:", weights)
    i = 0
    times -= 1
    while i < len(weights) - times:
        weights2[indexlist[i]] = weights[i]
        i += 1
    i = 0
    # print(weights[-3:],ns_hist[-3:])
    while i < num_GRR:
        if weights2[i] == 0:
            # print(reflect_indexlist[i])
            for ttt in range(times):
                if reflect_indexlist[i] == -2: #-2
                    weights2[i] = (weights[-1]) / (count_merged[1])
                if reflect_indexlist[i] == -1: #-2
                    weights2[i] = (weights[-2]) / (count_merged[0])
            # weights2[i] = 0
        i += 1
    # print("\t uniform:", count_merged,sum(weights2))
    return weights2
