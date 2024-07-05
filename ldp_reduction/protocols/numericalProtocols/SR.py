import numpy as np
import math
import random
def duchi(ori_data,epsilon,hybird):

    e_epsilon = np.exp(epsilon)
    para_ep = (e_epsilon - 1) / (2 * e_epsilon + 2)
    t = (e_epsilon + 1) / (e_epsilon - 1)
    MSE = 0
    # three-outputs when epsilon > 0.69
    if epsilon > 0.69 and hybird == 1:
        delta_0 = e_epsilon**4 + 14 * (e_epsilon**3) + 50 * (e_epsilon**2) - 2*e_epsilon+25
        delta_1 = -2 * (e_epsilon**6) -42 * (e_epsilon**5) - 270* (e_epsilon**4) -404*(e_epsilon**3) -918*(e_epsilon**2) + 30 * e_epsilon -250
        if epsilon < math.log(2):
            p00 = 0
        elif epsilon < math.log(5.53):
            p00 = -1/6*(-(e_epsilon**2)-4*e_epsilon-5+2*math.sqrt(delta_0)*math.cos(3.14159/3+1/3*math.acos(-delta_1/(2*(delta_0**1.5)))))
        else:
            p00 = e_epsilon/(e_epsilon+2)

        newa = []
        for j in ori_data:
            if j <= 1 and j >= -1:
                newa.append(j)
        res = 0
        all = sum(newa)
        for j in newa:
            if j >= 0:
                pr_xc = (1-p00)/2 + ((1-p00)/2 - (e_epsilon-p00)/e_epsilon/(e_epsilon+1)) * j
                pr_x_pos_c = (1-p00)/2 + ((e_epsilon-p00)/(e_epsilon+1) - (1-p00)/2 ) * j
            else:
                pr_xc = (1-p00)/2 - ((e_epsilon - p00) / (e_epsilon + 1) - (1 - p00) / 2) * j
                pr_x_pos_c = (1 - p00) / 2 + ((1 - p00) / 2 - (e_epsilon - p00) /e_epsilon / (e_epsilon + 1)) * j
            random_value = random.random()
            px0 = p00 + (p00/e_epsilon-p00)*j
            # if random_value < 0.02 or random_value>0.98:
            #     print(pr_x_pos_c + pr_xc + px0)
            if random_value < pr_xc:
                res += (-t)
            elif random_value <= (pr_xc+px0):
                res += 0
            else:
                res += t
            length = len(newa)
            res = res / length
            # print("three??",res)
            if res > 1:
                res = 1
            elif res < -1:
                res = -1
            MSE = (all / length - res) ** 2
    else:
        newa = []
        for j in ori_data:
            if j <= 1 and j >= -1:
                newa.append(j)
        res = 0
        all = sum(newa)
        for j in newa:
            pr = para_ep * j + 0.5
            if random.random() <= pr:
                res += t
            else:
                res -= t
        length = len(newa)
        res = res / length
        if res > 1:
            res = 1
        elif res < -1:
            res = -1
        MSE = ( all/length- res )**2
    return MSE

