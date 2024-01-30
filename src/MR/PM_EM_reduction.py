# coding=utf-8
import numpy as np
from PM_EM import EMS,EM
# from scipy.stats import binom
from numpy import linalg as LA
import matplotlib.pyplot as plt
import random

import math

def PM_noise(samples, k, B, C, pr):
    res = []
    for value in samples:
        lt = k * value - B
        rt = lt + 2 * B
        if random.random() <= pr:
            perturbed_value = random.random() * 2 * B + lt
        else:
            temp = random.random()
            ppp = (lt + C) / (2 * k)
            if ppp > temp:
                perturbed_value = temp * (2 * k) - C
            else:
                perturbed_value = rt + (temp - ppp) * (2 * k)

        res.append(perturbed_value)

    return np.array(res)

def perturb_PM(ori_samples,eps):
    e_epsilon = math.e**(eps)
    e_epsilon_sqrt = math.sqrt(e_epsilon)
    B = 1 / (e_epsilon_sqrt - 1)
    k = B * e_epsilon / (B * e_epsilon - 1 - B)
    C = k + B
    q = 1 / (2 * B * e_epsilon + 2 * k)
    p = e_epsilon / (2 * B * e_epsilon + 2 * k)
    pr = 2 * B * e_epsilon * q
    noisy_samples = PM_noise(ori_samples, k, B, C, pr)
    return noisy_samples



def PM_E(noisy_samples, eps, randomized_bins=1024, domain_bins=1024,loglikelihood_threshold = 1e-3, ifReduct=False,remained=1):
    loglikelihood_threshold = 1e-4 * (math.e ** eps)
    if len(noisy_samples) < 100000:
        loglikelihood_threshold = 1e-3 * (math.e ** eps)
    e_epsilon = math.e ** eps
    e_epsilon_sqrt = math.sqrt(e_epsilon)
    B = 1/ (e_epsilon_sqrt - 1)
    k = B * e_epsilon / (B * e_epsilon - 1 - B)
    C = k + B
    q = 1 / (2 * B * e_epsilon + 2 * k)

    # report matrix
    m = randomized_bins
    n = domain_bins
    m_cell = (2 * C) / m

    transform = np.ones((m, n)) * q * m_cell
    q_value = transform[0, 0]
    p_value = q_value * e_epsilon
    pro_pass = 0
    left_index = 0
    right_index = 0
    #
    for i in range(0, n):
        if i== n-1:
            sss =1
        if i == 0:
            a = int(B / C * n)
            reseverytime = 1 - a * p_value - (n - a) * q_value
            sss = (p_value-q_value-reseverytime)
            pro_pass = ((n - a-1)  * (p_value - q_value) + (p_value-q_value-reseverytime)) / (n-1)
            transform[0:a, i ] = p_value
            transform[a, i ] = q_value + reseverytime
            right_index = a
            left_index = 0
        else:
            temp_right = transform[left_index,i-1] - pro_pass
            if temp_right >= q_value:
                transform[left_index, i] = transform[left_index,i-1] - pro_pass # 左边减去 1
                if transform[right_index, i-1] + pro_pass < p_value:
                    transform[right_index, i ] = transform[right_index, i-1] + pro_pass
                else:
                    overflow = transform[right_index, i-1] + pro_pass - p_value
                    transform[right_index, i] = p_value
                    if right_index >= n-1 and   overflow < 1e-5:
                        transform[left_index + 1:right_index, i] = p_value
                        break
                    right_index+=1
                    transform[right_index,i] = q_value + overflow
            else:
                overflow = pro_pass - (transform[left_index,i-1] - q_value)
                transform[left_index, i] = q_value
                left_index+= 1
                transform[left_index,i] = p_value - overflow
                if transform[right_index, i-1] + pro_pass < p_value:
                    transform[right_index, i ] = transform[right_index, i-1] + pro_pass
                else:
                    overflow = transform[right_index, i-1] + pro_pass - p_value
                    transform[right_index, i] = p_value
                    if right_index == n-1 and  overflow < 1e-4:
                        transform[left_index+1:right_index, i] = p_value
                        break
                    right_index+=1
                    transform[right_index,i] = q_value + overflow
            for jjj in range(left_index+1,right_index):
                transform[jjj, i] = p_value


    max_iteration = 5000
    ns_hist, _ = np.histogram(noisy_samples, bins=randomized_bins, range=(-1 * C, C))
    throd = 0.4 / e_epsilon_sqrt / (np.log(len(noisy_samples))-1)

    if not ifReduct:
        theta = EMS(int(n), ns_hist, transform, max_iteration, loglikelihood_threshold,m)* len(noisy_samples)
    else:
        if eps<=0.5 and len(noisy_samples)<100000:
            throd = throd /(math.e**eps -1)
        elif eps >=1 and len(noisy_samples)>100000:
            throd = throd/e_epsilon_sqrt
        if eps < 0.2 and len(noisy_samples)>100000:
            throd = throd*2
        print(throd)
        theta = EMS_reduct(throd,len(noisy_samples), ns_hist, transform, max_iteration, loglikelihood_threshold,remained,m) * len(noisy_samples)
    return theta






# Smoothing step


def EMS_reduct(throd,reports_number, ns_hist, transform, max_iteration, loglikelihood_threshold, remained,bin_number):
    n = bin_number
    theta = np.ones(n) / float(n)
    theta_old = np.zeros(int(n))

    smoothing_factor = 2
    binomial_tmp = [1, 2, 1]  # [1,6,15,20,15,6,1] #[1, 2, 1]
    smoothing_matrix = np.zeros((n, n))
    central_idx = int(len(binomial_tmp) / 2)
    for i in range(int(smoothing_factor / 2)):
        smoothing_matrix[i, : central_idx + i + 1] = binomial_tmp[central_idx - i:]
    for i in range(int(smoothing_factor / 2), n - int(smoothing_factor / 2)):
        smoothing_matrix[i, i - central_idx: i + central_idx + 1] = binomial_tmp
    for i in range(n - int(smoothing_factor / 2), n):
        remain = n - i - 1
        smoothing_matrix[i, i - central_idx + 1:] = binomial_tmp[: central_idx + remain]
    row_sum = np.sum(smoothing_matrix, axis=1)
    smoothing_matrix = (smoothing_matrix.T / row_sum).T

    sample_size = sum(ns_hist)
    old_logliklihood = 0
    ori_n = bin_number
    transform = transform[:, 0:int(n)]

    indexset = [0 for _ in range(ori_n)]
    count_remove = 0
    length = int(ori_n / 2)
    superF = 0.03
    iter =0
    if reports_number > 100000:
        superF = throd/2
    lis1,lis2 = [],[]
    while True:
        iter+=1
        r = 0
        old_logliklihood = 0
        while LA.norm(theta_old - theta, ord=1) > 1 / sample_size and r < max_iteration and ori_n * remained < n:

            # while r < max_iteration:
            theta_old = np.copy(theta)
            X_condition = np.matmul(transform, theta_old)

            TMP = transform.T / X_condition

            P = np.copy(np.matmul(TMP, ns_hist))
            P = P * theta_old

            theta = np.copy(P / sum(P))
            theta = np.matmul(smoothing_matrix, theta)
            theta = theta / sum(theta)
            logliklihood = np.inner(ns_hist, np.log(np.matmul(transform, theta)))
            imporve = logliklihood - old_logliklihood

            if r > 1 and abs(imporve) < loglikelihood_threshold:
                # print("stop when", imporve, loglikelihood_threshold,r)
                break

            old_logliklihood = logliklihood
            r += 1
        if iter >= int(math.sqrt(1 / remained)) + 2:
            break

        loglikelihood_threshold = loglikelihood_threshold * 2
        max_iteration = max_iteration / 2

        value = [1 for _ in range(ori_n - length)]
        largest_v = 0
        mini_v = 0
        first = 0
        for i in range(len(value)):
            ifinclude = False
            for j in range(length):
                if indexset[j + i] == 1:
                    ifinclude = True
                    break
            if not ifinclude and ((i < length + 150) or (i > ori_n - length - 150)):
                value[i] = sum(theta[i:i + length])
                largest_v = i
                if first == 0:
                    first = 1
                    mini_v = i

        minIndex = value.index(min(value))
        if abs(value[minIndex] - value[largest_v]) < superF / (2 ** (iter - 1)) and value[mini_v] > 2 * value[
            largest_v]:
            minIndex = largest_v
        elif abs(value[minIndex] - value[mini_v]) < superF / (2 ** (iter - 1)):
            minIndex = mini_v
        print(length,minIndex,value[minIndex],throd/(2**(count_remove)))
        if value[minIndex] < throd / (2 ** (count_remove)):
            count_remove += 1
            print("reduction invoke", minIndex, length)
            # print(value[minIndex], value[0], value[30])
            for i in range(length):
                theta[i + minIndex] = 0
                indexset[i + minIndex] = 1
            n = n - length
            if (reports_number>100000):
                para = 1.1
            else: para=3
            theta[minIndex + int(length / 2)+1] = value[minIndex]/para
            ttt = np.sum(transform[:,minIndex:minIndex+length], axis=1)/length
            # print(ttt,len(ttt))
            lis1.append(minIndex)
            lis2.append(length)
            length = int(length / 2)
            transform[:,minIndex + length] = ttt

        else:
            length = length - 64
            if sum(indexset)==0:
                iter-=1
            else:
                iter-=0.5
        if length < 64:
            break
        # theta = theta / np.sum(theta)
    for t in range(len(lis1)):
        start = lis1[t]
        length = lis2[t]
        v = sum(theta[start:start+length])/length
        for i in range(length):
            theta[start+i] = v
    return theta

# def genNewtransform(transform,stratIndex,length):
#     n

def EM_reduct(throd,reports_number, ns_hist, transform, max_iteration, loglikelihood_threshold,remained,bin_number):
    n = bin_number
    theta = np.ones(n) / float(n)
    theta_old = np.zeros(int(n))
    sample_size = sum(ns_hist)
    old_logliklihood = 0
    ori_n = bin_number
    # print("11111")
    transform = transform[:,0:int(n)]

    indexset = [0 for _ in range(ori_n)]
    count_remove = 0
    length = int(ori_n / 2 )
    superF=0.13
    iter_times = int(math.sqrt(1 / remained))+2
    count = -1
    if reports_number>100000:
        superF = 0.004
    while True:
        r = 0
        count += 1
        old_logliklihood = 0
        while LA.norm(theta_old - theta, ord=1) > 1 / sample_size and r < max_iteration and ori_n * remained < n:
        # while r < max_iteration:
            theta_old = np.copy(theta)
            X_condition = np.matmul(transform, theta_old)
            TMP = transform.T / X_condition
            P = np.copy(np.matmul(TMP, ns_hist))
            P = P * theta_old
            theta = np.copy(P / sum(P))
            logliklihood = np.inner(ns_hist, np.log(np.matmul(transform, theta)))
            imporve = logliklihood - old_logliklihood
            if r > 1 and abs(imporve) < loglikelihood_threshold:
                # print("stop when", imporve, loglikelihood_threshold,r)
                break
            old_logliklihood = logliklihood
            r += 1
        if count >= iter_times or length <67:
            break

        loglikelihood_threshold = loglikelihood_threshold * 2
        max_iteration = max_iteration/2

        value = [1 for _ in range(ori_n - length)]
        largest_v = 0
        mini_v = 0
        first = 0
        for i in range(len(value)):
            ifinclude = False
            for j in range(length):
                if indexset[j + i] == 1:
                    ifinclude = True
                    break
            if not ifinclude and ((i < length+150) or (i > ori_n-length-150)):
                value[i] = sum(theta[i:i + length])
                largest_v = i
                if first==0:
                    first=1
                    mini_v = i

        minIndex = value.index(min(value))
        if reports_number<100000 and abs(value[minIndex]-value[largest_v])< superF/ (2**(count_remove)) and value[mini_v] > 2* value[largest_v]:
            minIndex = largest_v
        elif abs(value[minIndex]-value[mini_v])<superF/(2**(count_remove)):
            minIndex = mini_v
        # print(length,minIndex,value[minIndex],throd/(2**(count_remove)))
        if value[minIndex] < throd/(2**(count_remove)):
            count_remove+=1
            print("reduction invoke",minIndex,length)
            # print(value[minIndex],value[0],value[30])
            for i in range(length):
                theta[i + minIndex] = 0
                indexset[i + minIndex] = 1
            cc=0
            for i in range(1024):
                if theta[i]!=0:
                    cc+=1
            for i in range(1024):
                if theta[i]!=0:
                    theta[i]=1/cc
            n = n - length
            length = int(length/2)
            theta[minIndex+int(length/2)] = 0.001
        else:
            length = length-64
            count -= 0.5
        theta = theta / np.sum(theta)
        if length < 64:
            break
    return theta

#  old
# def EM_reduct(throd,reports_number, ns_hist, transform, max_iteration, loglikelihood_threshold,remained,bin_number):
#     n = bin_number
#     theta = np.ones(n) / float(n)
#     theta_old = np.zeros(int(n))
#     sample_size = sum(ns_hist)
#     old_logliklihood = 0
#     ori_n = bin_number
#     # print("11111")
#     transform = transform[:,0:int(n)]
#
#     indexset = [0 for _ in range(ori_n)]
#     count_remove = 0
#     length = int(ori_n / 2 )
#     superF=0.13
#     iter_times = int(math.sqrt(1 / remained))+2
#     count = -1
#     if reports_number>100000:
#         superF = 0.004
#     while True:
#         r = 0
#         count += 1
#         old_logliklihood = 0
#         while LA.norm(theta_old - theta, ord=1) > 1 / sample_size and r < max_iteration and ori_n * remained < n:
#         # while r < max_iteration:
#             theta_old = np.copy(theta)
#             X_condition = np.matmul(transform, theta_old)
#
#             TMP = transform.T / X_condition
#
#             P = np.copy(np.matmul(TMP, ns_hist))
#             P = P * theta_old
#
#             theta = np.copy(P / sum(P))
#
#             logliklihood = np.inner(ns_hist, np.log(np.matmul(transform, theta)))
#             imporve = logliklihood - old_logliklihood
#
#             if r > 1 and abs(imporve) < loglikelihood_threshold:
#                 # print("stop when", imporve, loglikelihood_threshold,r)
#                 break
#
#             old_logliklihood = logliklihood
#             r += 1
#         if count >= iter_times or length <67:
#             break
#
#         loglikelihood_threshold = loglikelihood_threshold * 2
#         max_iteration = max_iteration/2
#
#         value = [1 for _ in range(ori_n - length)]
#         largest_v = 0
#         mini_v = 0
#         first = 0
#         for i in range(len(value)):
#             ifinclude = False
#             for j in range(length):
#                 if indexset[j + i] == 1:
#                     ifinclude = True
#                     break
#             if not ifinclude and ((i < length+150) or (i > ori_n-length-150)):
#                 value[i] = sum(theta[i:i + length])
#                 largest_v = i
#                 if first==0:
#                     first=1
#                     mini_v = i
#
#         minIndex = value.index(min(value))
#         if reports_number<100000 and abs(value[minIndex]-value[largest_v])< superF/ (2**(count_remove)) and value[mini_v] > 2* value[largest_v]:
#             minIndex = largest_v
#         elif abs(value[minIndex]-value[mini_v])<superF/(2**(count_remove)):
#             minIndex = mini_v
#         # print(length,minIndex,value[minIndex],throd/(2**(count_remove)))
#         if value[minIndex] < throd/(2**(count_remove)):
#             count_remove+=1
#             print("reduction invoke",minIndex,length)
#             # print(value[minIndex],value[0],value[30])
#             for i in range(length):
#                 theta[i + minIndex] = 0
#                 indexset[i + minIndex] = 1
#             cc=0
#             for i in range(1024):
#                 if theta[i]!=0:
#                     cc+=1
#             for i in range(1024):
#                 if theta[i]!=0:
#                     theta[i]=1/cc
#             n = n - length
#             length = int(length/2)
#         else:
#             length = length-64
#             count -= 0.5
#         theta = theta / np.sum(theta)
#         if length < 64:
#             break
#     return theta


def calmean(res,bin_number):
    # print(res)
    normalized_arr = res / np.sum(res)
    # print(normalized_arr)
    mean_est = 0
    for i in range(bin_number):
        mean_est += normalized_arr[i] * (i)
    mean_est = (mean_est / bin_number - 0.5) * 2  # 转换到 -1 到 1 的区间
    return mean_est


