import math
import random

import xxhash

import numpy as np
from OLH_EM import lh_perturb
# from compareIndifferentDataset_OLH import calMAE
import time
from utils import calMAE,calMSE
# def lh_r(real_dist, eps):
#     p = 0.5
#     g = int(np.exp(eps)) + 1
#     num_domain = len(real_dist)
#     noisy_samples = lh_perturb(real_dist, g, p)
#     est_dist = olh_EM_reduct(noisy_samples, num_domain,eps, g, real_dist/sum(real_dist))
#     return est_dist
res = []

def gen_matrix22(noisy_samples, num_domain, g, eps,re):
    q = 1/ (np.exp(eps)+g-1)
    transferM = np.zeros((len(noisy_samples), num_domain-re+1))
    for i in range(len(noisy_samples)):
        for v in range(num_domain):
            x = xxhash.xxh32(str(v), seed=noisy_samples[i][1]).intdigest() % g
            if noisy_samples[i][0] == x:
                transferM[i,v] = np.exp(eps) * q
        for v in range(num_domain):
            if transferM[i,v]==0:
                transferM[i, v] = q
        # transferM[i,num_domain]= 1/(g)
    # Divide each row by its sum to normalize the rows
    for i in range(len(noisy_samples)):
        newV= 0
        for j in range(re+1):
            newV += transferM[i, num_domain-1-j]
        transferM[i, num_domain -1- re] = newV/(re+1)
        # print(transferM[:,0])
    return transferM[:,:num_domain-re]
# def olh_EM_reduct(noisy_samples, num_domain,eps,  real_dist,init_weight):
#     g = max(3, int(np.exp(eps)) + 1)
#     throd = 1.6*2 * math.sqrt(np.exp(eps)) / (np.exp(eps) - 1) / np.sqrt(len(noisy_samples))
#     re = 0
#     index_list = [i for i in range(len(init_weight))]
#     for i in range(len(init_weight)):
#         if init_weight[i] < throd:
#             re+=1
#             index_list[i] = -1
#
#     # if num_domain>200:
#     #     re = 125
#     # else:
#     #     re = 5
#     print(init_weight,re)
#     print("minimax:",max(init_weight[-25:]),min(init_weight[0:25]))
#     weights = np.ones(num_domain-re) / num_domain
#     for i in range(num_domain-re):
#         if init_weight[i] > 0:
#             weights[i] = init_weight[i]
#         else:
#             weights[i] = 0.00001
#     weights[-1] = min(0.05,max(0.02,1- sum(weights)))
#     tolerance = [1e-2*np.exp(eps),1e-1*np.exp(eps),5e-2*np.exp(eps),5e-2*np.exp(eps),1e-2*np.exp(eps)]
#     if len(noisy_samples)>100000:
#         tolerance = [i/10 for i in tolerance]
#     max_iter = [200, 100, 100, 100, 100]
#
#     transferM = gen_matrix22(noisy_samples, num_domain, g, eps,re)
#
#
#     if len(noisy_samples)>100000:
#         throd = throd*2
#     print("throd is",throd)
#     times = 0
#     BIC_old = 50000
#     count_para = num_domain * 2
#     probs = np.zeros((len(noisy_samples), num_domain-re))
#     ll_old = -10000000
#     while True:
#         for iter in range(max_iter[times]):
#             # E-step: compute expected values of latent variable z
#             for i in range(len(noisy_samples)):
#                 temp = np.dot(transferM[i, :], weights)
#                 for k in range(num_domain-re):
#                     probs[i, k] = weights[k] * transferM[i,k] /temp
#             probs /= probs.sum(axis=1, keepdims=True)
#             # M-step: update mixture weights
#             weights = probs.mean(axis=0)
#             ll = np.sum(np.log(probs @ weights))
#             if (ll - ll_old < tolerance[times] and ll - ll_old >= 0) or (iter>130 and ll <ll_old):
#                 break
#             if iter % 10 == 0:
#                 # temp = calMAE(real_dist,weights,topk)
#                 temp2 = calMAE(weights,real_dist,topK=True)
#                 res.append(temp2)
#                 print(iter,temp2, ll-ll_old,weights[-1])
#             if times >= 2 and iter == 0:
#                 BIC_new = (count_para)*np.log(len(noisy_samples))-2*ll
#                 if BIC_new > BIC_old:
#                     break
#             ll_old = ll
#         times += 1
#         break
#     weights_new = np.zeros(num_domain)
#     temp = weights[-1]
#     for i in range(len(weights)-1):
#         weights_new[i] = weights[i]
#     for i in range(len(weights)-1,len(weights_new)):
#         weights_new[i] = temp / (len(weights_new)+1-len(weights))
#     print("???\t",weights_new)
#     return weights_new

def olh_EM_reduct(noisy_samples, num_domain,eps,  real_dist,init_weight,topk=10):
    g = max(3, int(np.exp(eps)) + 1)
    q = 1 / (np.exp(eps) + g - 1)
    p = np.exp(eps) / q

    weights = np.ones(num_domain) / num_domain
    for i in range(num_domain):
        if init_weight[i] > 0:
            weights[i] = init_weight[i]
        else:
            weights[i] = 0.00001
    for i in range(100,num_domain):
        if random.random()>0.5:
            weights[i] = 0

    tolerance = [np.exp(eps),1e-1*np.exp(eps),5e-2*np.exp(eps),5e-2*np.exp(eps),1e-2*np.exp(eps)]
    if len(noisy_samples)>100000:
        tolerance = [i/10 for i in tolerance]
    max_iter = [100, 100, 100, 1000, 500]
    transferM = gen_matrix22(noisy_samples, num_domain, g, eps,0)

    redu_num_domain = num_domain
    throd = 5 * math.sqrt(2.73)/ (np.exp(eps)+1)/np.sqrt(len(noisy_samples))
    if len(noisy_samples)>100000:
        throd = throd*2
    print("throd is",throd)
    times = 0

    BIC_old = 50000
    count_para = num_domain * 2
    probs = np.zeros((len(noisy_samples), num_domain))
    ll_old = -10000000
    while True:
        for iter in range(max_iter[times]):
            # E-step: compute expected values of latent variable z
            for i in range(len(noisy_samples)):
                temp = np.dot(transferM[i, :], weights)
                for k in range(num_domain):
                    probs[i, k] = weights[k] * transferM[i,k] /temp
            probs /= probs.sum(axis=1, keepdims=True)
            # M-step: update mixture weights
            weights = probs.mean(axis=0)
            ll = np.sum(np.log(probs @ weights))
            if (ll - ll_old < tolerance[times] and ll - ll_old >= 0) or (iter>26 and ll <ll_old):
                break
            if iter % 2 == 0:
                temp = calMAE(real_dist,weights,topk)
                temp2 = calMSE(real_dist,weights,topk)
                res.append(temp)
                print(iter,temp,temp2, ll-ll_old,weights[-1])
            if times >= 2 and iter == 0:
                BIC_new = (count_para)*np.log(len(noisy_samples))-2*ll
                if BIC_new > BIC_old:
                    return weights_old
            ll_old = ll
        BIC_old = (count_para)*np.log(len(noisy_samples))-2*ll
        weights_old = np.copy(weights)
        times += 1
        count = int(num_domain / (2**(times)))
        templist = []
        ll_old=ll
        # vmin = min(times,2)
        # new_throd = throd * (10.6-times*0.6)/10
        for k in range(num_domain):
            if weights[num_domain - 1 - k] != 0 and weights[num_domain - 1 - k] < throd:
                templist.append(weights[num_domain - 1 - k])
                # templist.append(weights[num_domain - 1 - k])
        templist = sorted(templist)
        minV = min(len(templist), count)
        # reduction step: remove the low frequent candidates
        for k in range(minV):
            for v in range(num_domain):
                if templist[k] == weights[v]:
                    weights[v] = 0
                    redu_num_domain -= 1
                    transferM[:,v] = 0
        count_para = redu_num_domain*2
        if times >1:
            break
    print(weights[0:])
    return weights