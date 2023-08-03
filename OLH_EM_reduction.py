
import numpy as np
from OLH_EM import gen_matrix,lh_perturb,calMAE
import time
def lh_r(real_dist, eps):
    p = 0.5
    g = int(np.exp(eps)) + 1
    q = 1 / g
    num_domain = len(real_dist)
    noisy_samples = lh_perturb(real_dist, g, p)
    est_dist = olh_EM(noisy_samples, num_domain,eps, g, p, q,real_dist/sum(real_dist))
    return est_dist

res = []
def olh_EM(noisy_samples, num_domain,eps, g, p, q,real_dist):
    weights = np.ones(num_domain) / num_domain
    tolerance = [1,1e-1,1e-2,1e-3]
    max_iter = [1000,1000,1000,1000]
    transferM = gen_matrix(noisy_samples, num_domain, g, p, q)

    # redu_num_domain
    redu_num_domain = num_domain
    throd = 4/ (np.exp(eps))/np.sqrt(len(noisy_samples)) # super-param
    print("throd is",throd)
    times = 0
    ll_old = -10000000
    weights_old = np.array([0])
    BIC_old = 50000000
    count_para = num_domain * 2
    probs = np.zeros((len(noisy_samples), num_domain))
    while True:

        ll_old = -10000000
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
            if ll - ll_old < tolerance[times] and ll - ll_old >= 0:
                print(ll,ll_old)
                # print(weights,"55555")
                break
            ll_old = ll
            print(iter,ll,calMAE(real_dist,weights))
            if times>=1 and iter==2:
                print(weights)
            if iter % 5 == 0:
                res.append(calMAE(real_dist, weights))
            if times >= 2 and iter == 0:
                BIC_new = (count_para)*np.log(len(noisy_samples))-2*ll
                if BIC_new > BIC_old:
                    return weights_old
        # print(transferM[0:20,:])
        # print("weights",weights)
        BIC_old = (count_para)*np.log(len(noisy_samples))-2*ll
        weights_old = np.copy(weights)
        # reduction step: remove the low frequent candidates
        for w in range(len(weights)):
            if weights[w] < throd:
                # transferM[:,w] = 0
                weights[w] = 0
                redu_num_domain -= 1
        # print("weights2", weights)
        times += 1
        count_para = redu_num_domain*2

        # print(weights)

        if times >3:
            break
    print(",".join(str(i) for i in res))
    # dummy step to get the final weights
    return weights


if __name__ == "__main__":
    # ori_dataDist = np.array([4000,3000,1000,2000,4000,0,4550,2000,500,0,9,8,22,31,14,15,6,8,5])
    ori_dataDist = np.array([100, 500, 100, 100, 100, 155, 100, 50,100, 100, 0, 0, 0, 0, 0,0, 0, 0, 0,0,0,0,0,0,00,0,0,0,0,0,0,0,00, 0, 0, 100])
        # np.array([4000,3000,1000,2000,4000,4550,2000,500])

    # ori_dataDist = np.array([20000,20000,23000,21000,19800,18900,15900,18900,20000,
    #                    24200,20000,14000,23000,21000,20000,22000,14000,22210,
    #                             234,0,0,10,0,0,4,0,0,0,2,0,0,80,0,0,78,0,0,0,0,
    #                           0,0,0,0,33,0,0,323,0,0,0,21,0,0,3,0,5,213,2,3,4,23,0,0,0,0,23,23,123,11,234,9,2,1,4,5,2,3,2,
    #                          2,34,1])
    start = time.time()
    b = lh_r(ori_dataDist, 3)
    end = time.time()
    print("time cost (s):",end-start)
    print(b[0:8])
    print(calMAE(ori_dataDist/sum(ori_dataDist),b))
    # 597.54s  MAE: 0.0028