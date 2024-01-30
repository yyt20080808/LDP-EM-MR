import math
import numpy as np
from MR.GRR_EM_reduction import noisedP2hist
import random
from deconvvolution_ldp import *
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

def gen_matrix(e_epsilon,number):
    p = e_epsilon / (e_epsilon + number - 1)
    q = 1 / (e_epsilon + number - 1)
    # Initialize the matrix with all elements set to q
    matrix = np.full((number, number), q)

    # Set diagonal elements to 1/2
    np.fill_diagonal(matrix, p)
    return matrix

def GRR_revision_no(noised_reports, num_GRR, epsilon):
    matrix = gen_matrix(np.exp(epsilon), num_GRR)
    ns_hist = noisedP2hist(noised_reports, num_GRR)[0:num_GRR]
    ns_hist = ns_hist / len(noised_reports)

    c = np.matrix(matrix)
    d = np.matmul(c.I, ns_hist)
    d = np.asarray(d)
    return d[0]

def GRR_revision_basecut(noised_reports,number_GRR,epsilon):
    e_epsilon = math.e**epsilon
    p_true = e_epsilon / (e_epsilon + number_GRR - 1)
    q = 1 / (e_epsilon + number_GRR - 1)
    res =[]
    n = len(noised_reports)
    ns_hist,_ = np.histogram(noised_reports, bins=number_GRR, range=(0, number_GRR-1))
    for i in range(number_GRR):
        xv = (ns_hist[i] / n - q) / (p_true - q)
        if xv <0.005:
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

def IIW(noised_data, domain_size, epsilon):
    # epsilon = 3
    # noised_data,domain_size,ori_est = generate_noised_data(epsilon)
    noised_histogram, bin_edges = np.histogram(noised_data, bins=domain_size,range=(0,domain_size-1))
    # print(noised_histogram)
    pure_p = np.exp(epsilon)/(np.exp(epsilon)+domain_size-1)
    pure_q = 1 /(np.exp(epsilon)+domain_size-1)
    est_psd_n = get_est_noise_psd(pure_p, pure_q, domain_size, sum(noised_data))

    a = [pure_p]
    b = [pure_q for i in range(domain_size-1)]
    a = np.concatenate((np.array(a), np.array(b)), axis=0)
    est_f = improved_iterative_wiener(np.array(a),np.array(noised_histogram),est_psd_n,100)
    # est_f = average_multiple_random_permuted(np.array(a),np.array(noised_histogram),est_psd_n,10)
    # print(est_f/len(noised_data))
    return est_f/len(noised_data)
    # print(est_f/len(noised_data))
    #
    # print(calMAE(ori_est,est_f/len(noised_data)))