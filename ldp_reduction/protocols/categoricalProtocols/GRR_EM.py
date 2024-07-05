
import numpy as np
from ..MR.GRR_EM_reduction import noisedP2hist
from .utils import calMAE,calMSE
from .GRR import gen_matrix

res = []
def EM_GR(ori,noised_report,num_GRR, e_epsilon,ifcut=0,topK=10):
    smoothing_factor = 2
    # binomial_tmp = [binom(smoothing_factor, k) for k in range(smoothing_factor + 1)]
    binomial_tmp = [1, 2, 1]  # [1,6,15,20,15,6,1] #[1, 2, 1]
    smoothing_matrix = np.zeros((num_GRR, num_GRR))
    central_idx = int(len(binomial_tmp) / 2)
    for i in range(int(smoothing_factor / 2)):
        smoothing_matrix[i, : central_idx + i + 1] = binomial_tmp[central_idx - i:]
    for i in range(int(smoothing_factor / 2), num_GRR - int(smoothing_factor / 2)):
        smoothing_matrix[i, i - central_idx: i + central_idx + 1] = binomial_tmp
    for i in range(num_GRR - int(smoothing_factor / 2), num_GRR):
        remain = num_GRR - i - 1
        smoothing_matrix[i, i - central_idx + 1:] = binomial_tmp[: central_idx + remain]
    row_sum = np.sum(smoothing_matrix, axis=1)
    smoothing_matrix = (smoothing_matrix.T / row_sum).T



    matrix = gen_matrix(e_epsilon,num_GRR)
    weights = np.ones(num_GRR-ifcut) / (num_GRR-ifcut)
    # Run EM algorithm
    ns_hist = noisedP2hist(noised_report, num_GRR)[0:num_GRR]
    tolerance = 1e-6
    max_iter = 1400
    ll_old = -np.inf
    for iter in range(max_iter):
        # E-step: compute expected values of latent variable z
        X_condition = np.matmul(matrix, weights)
        TMP = matrix.T / X_condition
        P = np.copy(np.matmul(TMP, ns_hist))
        P = P * weights

        weights = np.copy(P / sum(P))
        # weights = np.matmul(smoothing_matrix, weights)
        # weights = weights / sum(weights)
        ll = np.inner(ns_hist, np.log(np.matmul(matrix, weights)))
        if ll - ll_old < tolerance:
            break
        ll_old = ll
        if iter % 20 == 0:
            new_weights=np.concatenate((weights,np.array([0 for i in range(ifcut)])),axis=0)
            res.append(calMSE(new_weights, ori,topK))
    # print("EM:",iter, calMSE(new_weights, ori,topK),ll)
    return weights




