def calMAE(a,b,topK=False):

    res = 0
    count = 0
    if topK:
        for i in range(len(a)):
            if b[i]>0.005:
                res += (abs(a[i]-b[i]))
                count +=1
    else:
        for i in range(len(a)):
            res += (abs(a[i]-b[i]))
            count += 1
    return res/count

def calMSE(a,b,topK=False):

    res = 0
    count = 0
    if topK:
        for i in range(len(a)):
            if b[i]>0.005:
                res += (abs(a[i]-b[i]))**2
                count +=1
    else:
        for i in range(len(a)):
            res += (abs(a[i]-b[i]))**2
            count += 1
    return res/count

import numpy as np
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
