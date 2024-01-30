
import numpy as np
import xxhash
import time
from utils import calMAE,calMSE
res = []
def lh(real_dist, eps, ifEM=False):
    p = 0.5
    g = int(np.exp(eps)) + 1
    q = 1 / g
    num_domain = len(real_dist)
    # n = sum(real_dist)
    noisy_samples = lh_perturb(real_dist, g, p)
    # print(noisy_samples)
    if ifEM:
        est_dist = olh_EM(noisy_samples, num_domain, g, eps, q,real_dist/sum(real_dist))
    else:
        est_dist = lh_aggregate(noisy_samples, num_domain, g, p, q)

    return est_dist

def revise(est, g, p, q):
    a = 1 / (p - q)
    b =  q / (p - q)
    est = a * est - b
    return est

def olh_EM(noisy_samples, num_domain, eps, real_dist, init_weight,topk=10):
    g = max(3,int(np.exp(eps)) + 1)
    weights = np.ones(num_domain) / num_domain
    for i in range(num_domain):
        if init_weight[i]>0:
            weights[i] = init_weight[i]
        else:
            weights[i]=0.00001
    tolerance = 1e-4*np.exp(eps)
    max_iter = 40
    ll_old = -np.inf
    transferM = gen_matrix(noisy_samples, num_domain, g, eps)
    probs = np.zeros((len(noisy_samples), num_domain))
    for iter in range(max_iter):
        # E-step: compute expected values of latent variable z
        for i in range(len(noisy_samples)):
            temp = np.dot(transferM[i, :], weights)
            for k in range(num_domain):
                probs[i, k] = weights[k] * transferM[i,k] /temp
        probs /= probs.sum(axis=1, keepdims=True)

        # M-step: update mixture weights
        weights = probs.mean(axis=0)
        ll = np.sum(np.log(probs @ weights))
        if ll - ll_old < tolerance:
            # print(iter,"55555\n")
            break
        ll_old = ll

        if iter%20==0:
            print(iter, ll, calMSE(real_dist, weights,topk))
            res.append(calMAE(real_dist, weights,topk))
    # print(",".join(str(i) for i in res))
    # print("EM:",weights)
    return weights
def gen_matrix(noisy_samples, num_domain, g, eps):
    q = 1/ (np.exp(eps)+g-1)
    transferM = np.zeros((len(noisy_samples), num_domain))
    for i in range(len(noisy_samples)):
        for v in range(num_domain):
            x = xxhash.xxh32(str(v), seed=noisy_samples[i][1]).intdigest() % g
            if noisy_samples[i][0] == x:
                transferM[i,v] = np.exp(eps) * q
        for v in range(num_domain):
            if transferM[i,v]==0:
                transferM[i, v] = q
    # normalization
    # row_sums = transferM.sum(axis=1)
    return transferM





def lh_perturb(real_dist, g, p):
    n = sum(real_dist)
    noisy_samples = np.zeros(n, dtype=object)
    samples_one = np.random.random_sample(n)
    seeds = np.random.randint(0, n, n)

    counter = 0
    for k, v in enumerate(real_dist):
        for _ in range(v):
            y = x = xxhash.xxh32(str(int(k)), seed=seeds[counter]).intdigest() % g
            if samples_one[counter] > p:
                y = np.random.randint(0, g - 1)
                if y >= x:
                    y += 1
            noisy_samples[counter] = tuple([y, seeds[counter]])
            counter += 1
    return noisy_samples


def lh_aggregate(noisy_samples, domain, g, p, q):
    n = len(noisy_samples)
    est = np.zeros(domain, dtype=np.int32)
    for i in range(n):
        for v in range(domain):
            x = xxhash.xxh32(str(v), seed=noisy_samples[i][1]).intdigest() % g
            if noisy_samples[i][0] == x:
                est[v] += 1

    a = 1.0 / (p - q)
    b = n * q / (p - q)
    est = a * est - b

    return est/n
