
import numpy as np
import xxhash
from utils import calMAE
# from GRR_EM import norm_sub
# def lh(real_dist, eps):
#     p = 0.5
#     g = int(np.exp(eps)) + 1
#     q = 1 / g
#     domain = len(real_dist)
#
#     noisy_samples = lh_perturb(real_dist, g, p)
#     est_dist = lh_aggregate(noisy_samples, domain, g, p, q)
#
#     return est_dist


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


def lh_aggregate(noisy_samples, domain, eps):
    p = 0.5
    g = max(3,int(np.exp(eps)) + 1)
    q = 1 / g
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


