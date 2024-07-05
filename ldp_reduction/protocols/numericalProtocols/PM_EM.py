import numpy as np
from numpy import linalg as LA
def EM(n, ns_hist, transform, max_iteration, loglikelihood_threshold,bin_number):
    theta = np.ones(n) / float(n)
    theta_old = np.zeros(int(n))
    r = 0
    sample_size = sum(ns_hist)
    old_logliklihood = 0
    # print("11111")
    transform = transform[:,0:int(n)]
    while LA.norm(theta_old - theta, ord=1) > 1 / sample_size and r < max_iteration:
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
    return theta
def EMS(n, ns_hist, transform, max_iteration, loglikelihood_threshold,bin_number):
    # smoothing matrix
    ori_n = n
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
    r = 0
    sample_size = sum(ns_hist)
    old_logliklihood = 0
    transform = transform[:, 0:n ]
    while LA.norm(theta_old - theta, ord=1) > 1 / sample_size and r < max_iteration:
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
        # print(logliklihood)

        old_logliklihood = logliklihood

        r += 1
    # print("logliklihood",logliklihood,len(theta)*np.log(10000)- 2*logliklihood)
    return theta
