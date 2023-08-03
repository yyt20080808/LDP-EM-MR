import numpy as np

from deconvvolution_ldp import *

from GRR_EM import *
from OUE_EM import oue
def test_GRR():
    epsilon = 3
    noised_data,domain_size,ori_est = generate_noised_data(epsilon)
    noised_histogram, bin_edges = np.histogram(noised_data, bins=domain_size,range=(0,domain_size-1))
    print(noised_histogram)
    pure_p = np.exp(epsilon)/(np.exp(epsilon)+domain_size-1)
    pure_q = 1 /(np.exp(epsilon)+domain_size-1)
    est_psd_n = get_est_noise_psd(pure_p, pure_q, domain_size, sum(noised_data))

    a = [pure_p]
    b = [pure_q for i in range(domain_size-1)]
    a = np.concatenate((np.array(a), np.array(b)), axis=0)
    est_f = improved_iterative_wiener(np.array(a),np.array(noised_histogram),est_psd_n,10)
    # est_f = average_multiple_random_permuted(np.array(a),np.array(noised_histogram),est_psd_n,10)

    print(est_f/len(noised_data))

    print(calMAE(ori_est,est_f/len(noised_data)))


def test_OUE():
    # SFC dataset
    # ori = np.array(
    #     [2418, 2706, 731, 3998, 836, 3595, 979, 1526, 461, 1255, 1136, 988, 1881, 1225, 1189, 1664, 247, 2436, 274,
    #      1363, 1021, 1104, 85, 1052, 371, 410, 200, 810,
    #      442, 784, 265, 633, 187, 367, 497, 11, 180, 254, 194, 362, 180, 237, 146, 147, 50, 7, 15, 84,
    #      19, 5, 16, 3, 11, 19, 10, 16, 29])
    # # Income dataset
    # ori = np.array([20000, 20000, 23000, 21000, 19800, 18900, 15900, 18900, 20000,
    #                 24200, 20000, 14000, 23000, 21000, 20000, 22000, 14000, 22210,
    #                 234, 0, 0, 10, 0, 0, 4, 0, 0, 0, 2, 0, 0, 80, 0, 0, 78, 0, 0, 0, 0,
    #                 0, 0, 0, 0, 33, 0, 0, 323, 0, 0, 0, 21, 0, 0, 3, 0, 5, 213, 2, 3, 4, 23, 0, 0, 0, 0, 23, 23, 123,
    #                 11, 234,
    #                 9, 2, 1, 4, 5, 2, 3, 2,
    #                 2, 34, 1])
    ori = np.array([1000 for i in range(50)])
    # ori = np.array([20000, 20000, 23000, 21000, 19800, 18900, 15900, 18900, 20000,
    #           24200, 20000, 14000, 23000, 21000, 20000, 22000, 14000, 22210,
    #           234, 0, 0, 10, 0, 0, 4, 0, 0, 0, 2, 0, 0, 80, 0, 0, 78, 0, 0, 0, 0,
    #           0, 0, 0, 0, 33, 0, 0, 323, 0, 0, 0, 21, 0, 0, 3, 0, 5, 213, 2, 3, 4, 23, 0, 0, 0, 0, 23, 23, 123, 11, 234,
    #           9, 2, 1, 4, 5, 2, 3, 2,
    #           2, 34, 1])
    epsilon = 3
    noised_data = oue(ori,epsilon)
    agg_column_samples = np.sum(noised_data, axis=0)
    domain_size = len(agg_column_samples)
    pure_p = 1 / 2
    pure_q = 1 / (np.exp(epsilon) + 1)
    a = [pure_p]
    b = [pure_q for i in range(domain_size - 1)]
    a = np.concatenate((np.array(a), np.array(b)), axis=0)
    est_psd_n = get_est_noise_psd(pure_p, pure_q, domain_size, sum(noised_data))
    est_f = average_multiple_random_permuted(np.array(a), np.array(agg_column_samples), est_psd_n, 20)

    print(est_f / len(noised_data))
    ori_est = ori/sum(ori)
    print(calMAE(ori_est, est_f / len(noised_data)))
if __name__ == "__main__":
    test_OUE()
    # test_GRR()