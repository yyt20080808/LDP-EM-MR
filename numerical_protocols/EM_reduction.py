# coding=utf-8
import numpy as np
from PM_EM import EMS,EM
# from scipy.stats import binom
from numpy import linalg as LA
import matplotlib.pyplot as plt
import random

import math
bin_number = 1024

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


    max_iteration = 10000
    ns_hist, _ = np.histogram(noisy_samples, bins=randomized_bins, range=(-1 * C, C))
    throd = 0.05 / e_epsilon_sqrt / (np.log10(len(noisy_samples))-1)
    if not ifReduct:
        theta = EM(int(n), ns_hist, transform, max_iteration, loglikelihood_threshold)* len(noisy_samples)
    else:
        theta = EM_reduct(throd, ns_hist, transform, max_iteration, loglikelihood_threshold,remained) * len(noisy_samples)
    return theta



def EMS_reduct( ns_hist, transform, max_iteration, loglikelihood_threshold, remained):
    # smoothing matrix
    ori_n = bin_number
    n = ori_n
    smoothing_factor = 2
    # binomial_tmp = [binom(smoothing_factor, k) for k in range(smoothing_factor + 1)]
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

    # EMS
    theta = np.ones(n) / float(n)
    theta_old = np.zeros(n)
    sample_size = sum(ns_hist)
    transform = transform[:, 0:n]
    throd = 0.001
    indexset = [0 for i in range(ori_n)]
    for times in range(int(math.sqrt(1/remained))):
        r = 0
        old_logliklihood = -1000000
        while LA.norm(theta_old - theta, ord=1) > 1 / sample_size and r < max_iteration and ori_n * remained < n:
            theta_old = np.copy(theta)
            X_condition = np.matmul(transform, theta_old)

            TMP = transform.T / X_condition

            P = np.copy(np.matmul(TMP, ns_hist))
            P = P * theta_old

            theta = np.copy(P / sum(P))

            # Smoothing step
            theta = np.matmul(smoothing_matrix, theta)
            theta = theta / sum(theta)
            logliklihood = np.inner(ns_hist, np.log(np.matmul(transform, theta)/sum(ns_hist)))
            imporve = logliklihood - old_logliklihood
            if r > 1 and abs(imporve) < loglikelihood_threshold:
                # print("stop when", imporve / old_logliklihood, loglikelihood_threshold)
                break
            old_logliklihood = logliklihood
            print(r,logliklihood)
            r += 1
        length = int(ori_n/(2**(times+1)))
        value = [1 for _ in range(ori_n-length)]
        for i in range(len(value)):
            ifinclude = False
            for j in range(length):
                if indexset[j+i]==1:
                    ifinclude = True
                    break
            if not ifinclude:
                value[i] = sum(theta[i:i+length])
        minIndex = value.index(min(value))
        if value[minIndex] < throd:
            for i in range(length):
                theta[i+minIndex]=0
                indexset[i+minIndex]=1
            n = n-length



    # print("logliklihood",logliklihood,len(theta)*np.log(10000)- 2*logliklihood)
    return theta

def EM_reduct(throd, ns_hist, transform, max_iteration, loglikelihood_threshold,remained):
    n = bin_number
    theta = np.ones(n) / float(n)
    theta_old = np.zeros(int(n))
    sample_size = sum(ns_hist)
    old_logliklihood = 0
    ori_n = bin_number
    # print("11111")
    transform = transform[:,0:int(n)]

    indexset = [0 for i in range(ori_n)]
    count_remove = 0
    for iter in range(int(math.sqrt(1 / remained))+4):
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

            logliklihood = np.inner(ns_hist, np.log(np.matmul(transform, theta)))
            imporve = logliklihood - old_logliklihood

            if r > 1 and abs(imporve) < loglikelihood_threshold:
                # print("stop when", imporve, loglikelihood_threshold,r)
                break

            old_logliklihood = logliklihood

            r += 1
        length = int(ori_n / (2 ** (iter + 1)))
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
        if abs(value[minIndex]-value[largest_v])<0.03/(2**(iter-1)):
            minIndex = largest_v
        elif abs(value[minIndex]-value[mini_v])<0.03/(2**(iter-1)):
            minIndex = mini_v
        # print(length,minIndex,value[minIndex],throd/(2**(count_remove)))
        if value[minIndex] < throd/(2**(count_remove)):
            count_remove+=1
            # print("reduction invoke")
            for i in range(length):
                theta[i + minIndex] = 0
                indexset[i + minIndex] = 1
            n = n - length

    return theta


def calmean(res):
    # print(res)
    normalized_arr = res / np.sum(res)
    # print(normalized_arr)
    mean_est = 0
    for i in range(bin_number):
        mean_est += normalized_arr[i] * (i)
    mean_est = (mean_est / bin_number - 0.5) * 2  # 转换到 -1 到 1 的区间
    return mean_est




if __name__ == "__main__":
    eplist = [0.1]
    # ori = np.random.normal(-0.94, 0.0005, 5000)
    # ori2 = np.random.uniform(-0.1,0,5000)
    # b = np.array([0.9,0.8,0.5,0.9,0.5,0.3,0.5,0.4,0.2,0.6]*1 )
    # ori = np.concatenate((ori,ori2,b),axis=0)
    # realmean = np.sum(ori)/len(ori)
    #
    # epsilon = 1
    # new_ori = []
    # for i in ori:
    #     if i > 1 or i < -1:
    #         pass
    #     else:
    #         new_ori.append(i)
    # for i in ori2:
    #     new_ori.append(i)

    # acc_cut = []
    # acc_ori = []
    # acc_pm = []
    # var_cut = []
    # var_ori = []
    # var_pm = []
    # expri_times = 10
    # for i in range(expri_times):
    #     new_ori = np.array(new_ori)
    #     theta = PM(new_ori, epsilon, randomized_bins=512, domain_bins=512,ifcut=2/5)
    #     y = [0 for j in range(512-len(theta))]
    #     theta = np.concatenate((theta,np.array(y)),axis=0)
    #     res_cut = (calmean(theta)- realmean)**2
    #     acc_cut.append(res_cut)
    #     noisy_samples = PM_noiseonly(new_ori, epsilon)
    #     res_pm = (sum(noisy_samples)/len(noisy_samples)- realmean)**2
    #     acc_pm.append(res_pm)
    #     theta2 = PM(new_ori, epsilon, randomized_bins=512, domain_bins=512, ifcut=1)
    #     print(sum(theta2[128:])/sum(theta2),sum(theta2[384:])/sum(theta2),sum(theta2[256:384])/sum(theta2),sum(theta2[128:256])/sum(theta2))
    #     # y = [0 for j in range(len(theta))]
    #     # theta = np.concatenate((theta, np.array(y)), axis=0)
    #     res_ori = (calmean(theta2) - realmean)**2
    #     print("compare",res_cut,res_ori,res_pm)
    #     acc_ori.append( res_ori)
    # print("reduction:",sum(acc_cut)/expri_times)
    # print("ori",sum(acc_ori)/expri_times)
    # print("pmm", sum(acc_pm) / expri_times)
    # print(theta)
    #     x = [i for i in range(512)]
    # y = [0 for i in range(len(theta))]
    # theta = np.concatenate((theta,np.array(y)),axis=0)

    # print(calmean(theta))
    # plt.bar(x, theta)
    # plt.xlabel(r'$\epsilon$', fontsize=16)
    # plt.ylabel(u'2', fontsize=20)
    # plt.tick_params(labelsize=20)
    # plt.text(1.19, 3.5, "key point", fontdict={'size': '16', 'color': 'b'})
    # plt.title(u"Comparision and varying epsilon", fontsize=16)
    # plt.axvline(x=1.185, ls="-", c="red")  # 添加垂直直线
    # plt.axvline(x=1.29, ls="-", c="gray")  # 添加垂直直线
    # plt.legend()
    # plt.show()
