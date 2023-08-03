
import numpy as np
import xxhash

def lh(real_dist, eps, ifEM=False):
    p = 0.5
    g = int(np.exp(eps)) + 1
    q = 1 / g
    num_domain = len(real_dist)
    # n = sum(real_dist)
    noisy_samples = lh_perturb(real_dist, g, p)
    # print(noisy_samples)
    if ifEM:
        est_dist = olh_EM(noisy_samples, num_domain, g, p, q)
    else:
        est_dist = lh_aggregate(noisy_samples, num_domain, g, p, q)

    return est_dist


def olh_EM(noisy_samples, num_domain, g, p, q):
    weights = np.ones(num_domain) / num_domain
    tolerance = 1e-5
    max_iter = 5000
    ll_old = -np.inf
    transferM = gen_matrix(noisy_samples, num_domain, g, p, q)
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
            print(iter,"55555")
            break
        ll_old = ll

    return weights
def gen_matrix(noisy_samples, num_domain, g, p, q):

    transferM = np.zeros((len(noisy_samples), num_domain))
    for i in range(len(noisy_samples)):
        for v in range(num_domain):
            x = xxhash.xxh32(str(v), seed=noisy_samples[i][1]).intdigest() % g
            if noisy_samples[i][0] == x:
                transferM[i,v] = 1/2
        for v in range(num_domain):
            if transferM[i,v]==0:
                transferM[i, v] = q
    print(transferM[:,0])
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
def calMAE(a,b):
    length = len(a)
    # length = 12
    res = 0
    for i in range(length):
        res += abs(a[i]-b[i])
    return res/length
if __name__ == "__main__":
    ori_dataDist = np.array([1000 for i in range(50)])
    b = lh(np.array(ori_dataDist), 1, ifEM = True)
    print(b)
    print(calMAE(ori_dataDist / sum(ori_dataDist), b / sum(b)))