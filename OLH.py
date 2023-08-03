
import numpy as np
import xxhash
from OLH_EM import calMAE
from GRR_EM import norm_sub
def lh(real_dist, eps):
    p = 0.5
    g = int(np.exp(eps)) + 1
    q = 1 / g
    domain = len(real_dist)

    noisy_samples = lh_perturb(real_dist, g, p)
    est_dist = lh_aggregate(noisy_samples, domain, g, p, q)

    return est_dist


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

    return est



if __name__ == "__main__":
    ori_dataDist = np.array([100, 500, 100, 100, 100, 155, 100, 50,100, 100, 0, 0, 0, 0, 0,0, 0, 0, 0,0,0,0,0,0,00,0,0,0,0,0,0,0,00, 0, 0, 100])
    # ori_dataDist= np.array([4000,3000,1000,2000,4000,0,4550,2000,500,0,9,8,22,31,14,15,6,8,5])
    # ori_dataDist = np.array([2000,3000,2000,3000,3000,2000,
                       # 20,0,90,5,0,0,50,0,100,0,100,0,0,4,0,0,0,0])
    # ori_dataDist = np.array([20000,20000,23000,21000,19800,18900,15900,18900,20000,
    #                    24200,20000,14000,23000,21000,20000,22000,14000,22210,
    #                             234,0,0,10,0,0,4,0,0,0,2,0,0,80,0,0,78,0,0,0,0,
    #                           0,0,0,0,33,0,0,323,0,0,0,21,0,0,3,0,5,213,2,3,4,23,0,0,0,0,23,23,123,11,234,9,2,1,4,5,2,3,2,
    #                          2,34,1])
    # ori_dataDist = np.array([1000 for i in range(50)])
    # ori_dataDist = np.array([1000, 1000, 1300, 1100, 980, 890, 590, 890, 1000,
    #                          1420, 2000, 400, 1300, 1100, 1000, 1200, 400, 1221,
    #                          0, 0, 0, 10, 0, 0, 4, 0, 0, 0, 2, 0, 0, 80, 0, 0, 0, 0, 0, 0, 0,
    #                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 21, 0, 0, 0, 0, 0])
    b = lh(np.array(ori_dataDist), 3)

    # b = norm_sub(np.array(b) , sum(ori_dataDist))
    print(b / sum(b))
    print(ori_dataDist / sum(ori_dataDist))
    print(calMAE(ori_dataDist/sum(ori_dataDist),b/sum(b)))