import numpy as np
import xxhash
import time

res = []


def revise(est, g, p, q):
    a = 1 / (p - q)
    b = q / (p - q)
    est = a * est - b
    return est


def olh_EM(noisy_samples, num_domain, eps, init_weight, ori):
    g = max(3, int(np.exp(eps)) + 1)
    weights = np.ones(num_domain) / num_domain
    for i in range(num_domain):
        if init_weight[i] > 0:
            weights[i] = init_weight[i]
        else:
            weights[i] = 0.001
    tolerance = 1e-4 * np.exp(eps)
    weights = weights / sum(weights)
    max_iter = int(2000/np.exp(eps))
    ll_old = -np.inf
    transferM = gen_matrix(noisy_samples, num_domain, g, eps)
    for iter in range(max_iter):
        # E-step: compute expected values of latent variable z
        temp = transferM @ weights
        probs = (transferM * weights) / temp[:, None]
        probs /= probs.sum(axis=1, keepdims=True)

        # M-step: update mixture weights
        weights = probs.mean(axis=0)
        ll = np.sum(np.log(probs @ weights))
        if 0<ll - ll_old < tolerance or (iter > 200 and ll - ll_old < 0 ):
            break
        # if iter % 20 == 0:
        #     print("OLH-EM", iter, ll - ll_old, "MSE:", calMSE(weights, ori))
        ll_old = ll
    return weights


def gen_matrix(noisy_samples, num_domain, g, eps):
    q = 1 / (np.exp(eps) + g - 1)
    p = np.exp(eps) * q
    transferM = np.full((len(noisy_samples), num_domain), q)
    for i, (noisy_value, seed) in enumerate(noisy_samples):
        hashes = [xxhash.xxh32(str(v), seed=seed).intdigest() % g for v in range(num_domain)]
        transferM[i, np.array(hashes) == noisy_value] = p
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

    return est / n
