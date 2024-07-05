import math
import random

import xxhash

import numpy as np
from ..categoricalProtocols.OLH_EM import gen_matrix
# import time

def gen_new_matrix(recent_matrix, reflected_indexlist, num_domain, count_reducted_list,
                   times, originMatrix):
    user_num = originMatrix.shape[0]
    count_reducted = sum(count_reducted_list)
    new_matrix = np.full((user_num, num_domain - count_reducted + times), 0.1)
    new_indexlist = np.ones(num_domain - count_reducted, dtype=int)
    i, j = 0, 0
    temp = np.zeros(user_num)
    while i < num_domain:
        if reflected_indexlist[i] >= 0:
            new_indexlist[j] = i  # the newlist's j-th pos stores the original index i
            new_matrix[:, j] = originMatrix[:, i]
            j += 1
        elif reflected_indexlist[i] == 0 - times:
            temp += originMatrix[:, i]
        i += 1
    for i in range(times, 1, -1):
        new_matrix[:, -i] = recent_matrix[:, -i + 1]
    # here we need normalize it
    new_matrix[:, num_domain - count_reducted + times - 1] = temp / count_reducted_list[times - 1]

    return new_matrix, new_indexlist


def olh_EM_reduct(noisy_samples, num_domain, eps, init_weight, topk=30):
    g = max(3, int(np.exp(eps)) + 1)
    weights = np.ones(num_domain) / num_domain
    for i in range(num_domain):
        if init_weight[i] > 0:
            weights[i] = init_weight[i]
        else:
            weights[i] = 0.0001

    maxtimes = 2
    tolerance = [1e-1 * np.exp(eps), 5e-2 * np.exp(eps), 1e-2 * np.exp(eps), 1e-2 * np.exp(eps), 1e-2 * np.exp(eps)]
    if len(noisy_samples) > 100000:
        tolerance = [i / 10 for i in tolerance]
    max_iter = [max(300,int(700/np.exp(eps))) for _ in range(5)]
    transferM = gen_matrix(noisy_samples, num_domain, g, eps)
    ori_matrix = np.copy(transferM)
    redu_num_domain = num_domain
    throd = 4*np.sqrt(np.exp(eps)) / (np.exp(eps) - 1) / np.sqrt(len(noisy_samples))
    if eps<1:
        maxtimes=1
    # print("throd is", throd)
    times = 0
    BIC_old = 50000
    count_para = num_domain * 20
    ll_old = -10000000
    reflect_indexlist = [i for i in range(num_domain)]
    indexlist = [i for i in range(num_domain)]
    count_merged = [0 for _ in range(num_domain)]  # count the merged components in each iteration
    while True:
        times += 1
        ll_old -= 2000
        for iter in range(max_iter[times - 1]):
            # E-step: compute expected values of latent variable z

            temp = transferM @ weights
            probs = (transferM * weights) / temp[:, None]
            probs /= probs.sum(axis=1, keepdims=True)
            # M-step: update mixture weights
            weights = probs.mean(axis=0)
            ll = np.sum(np.log(probs @ weights))
            if (tolerance[times - 1] > ll - ll_old >=0) or (iter > 63 and ll < ll_old):
                break
            # if iter % 20 == 0:
            #     print(iter, ll - ll_old,ll,weights[-3:])
            if times >= 2 and iter == 48:
                BIC_new = count_para * np.log(len(noisy_samples)) - 2 * ll
                if BIC_new > BIC_old:
                    print("")
                    # return weights_old
            ll_old = ll
        BIC_old = count_para * np.log(len(noisy_samples)) - 2 * ll
        weights_old = np.copy(weights)
        count = int(num_domain / (2 ** (times+0.2)))
        templist = []
        ttt = len(weights)
        if times > 1:
            ttt -= (times - 1)
        for k in range(topk, ttt):
            if weights[k] != 0 and weights[k] < throd:
                templist.append(weights[k])
        templist = sorted(templist)
        # if eps < 1.1:
        #     random.shuffle(templist)
        minV = min(len(templist), count)
        # print("sums:", weights[-3:])
        if times > maxtimes:
            break
        # reduction step: remove the low frequent candidates
        for k in range(minV):
            for v in range(len(weights)):
                if templist[k] == weights[v]:
                    weights[v] = 0
                    redu_num_domain -= 1
                    reflect_indexlist[indexlist[v]] = 0 - times
                    count_merged[times - 1] += 1
        count_para = redu_num_domain * 20
        if count_merged[times - 1] == 0:
            break
        # merge steps
        transferM, indexlist = gen_new_matrix(transferM, reflect_indexlist, num_domain, count_merged,
                                              times, ori_matrix)
        weights2 = np.copy(weights)
        weights = np.ones(num_domain - sum(count_merged) + times)/len(weights)
        v = 0
        for k in range(len(weights2)):
            if weights2[k] > 0:
                weights[v] = weights2[k]
                v += 1
        restw = max(1/num_domain,1- sum(weights))
        if len(weights)-v >0:
            restw = restw / (len(weights) -v)
        while v < len(weights):
            weights[v] = restw
            v += 1
    i = 0
    times -= 1
    weights2 = np.zeros(num_domain)
    while i < len(weights) - times:
        weights2[indexlist[i]] = weights[i]
        i += 1
    i = 0
    while i < num_domain:
        if weights2[i] == 0:
            for ttt in range(times): # 0~1 times =2
                # weight_index = -1-ttt
                # merge_num_index = times-1-ttt
                # reflect_index = -times+ttt
                if reflect_indexlist[i] == -times+ttt:
                    weights2[i] = (weights[-1 - ttt]) / (count_merged[times-1-ttt])
        i += 1
    return weights2
