import matplotlib.pyplot as plt
import numpy as np
import math
import random
from numpy import linalg as LA

pre_bins = 1024


def SVEPM(gamma, data, epsilonls, mul_d):
    numberExperi = 5
    real_f = len(data[0]) / (len(data[0]) + len(data[1]))
    # print("this is SVE",)
    res_f = []
    res_v_1 = []
    res_v_2 = []
    for epsilon in epsilonls:
        mse_f = []
        mse_v_1 = []
        mse_v_2 = []

        e_epsilon = math.e ** (epsilon)
        e_epsilon_sqrt = math.e ** (epsilon / 2)
        B = (e_epsilon_sqrt + 9) / (10 * e_epsilon_sqrt - 10)
        k = B * e_epsilon / (B * e_epsilon - 1 - B)
        C = k + B
        p = 1 / (2 * B * e_epsilon + 2 * k)
        pr = 2 * B * e_epsilon * p
        # 生成 矩阵
        Matrix = generateMatrix(epsilon, pre_bins)
        result_append = []

        # print("真实和假：",len(data[0]),len(data[1]))
        for times in range(numberExperi):
            observed_value = []
            sum_all = [sum(i) / len(i) for i in data]
            numbersofall = 0
            # data_types = 2
            res_all_x = 0  # 使用 OPM 返回的观察值
            res_all_l = 0  # 使用RR的返回的观察值
            count_l = 0  # 使用 RR的用户总数
            # for dataIndex in range(data_types):
            for i in range(1):
                target_data = data[0]
                otherdata = data[1]
                for j in target_data:
                    numbersofall += 1
                    res_x, res_l = pcmm(e_epsilon, True, gamma, j, k, B, C, pr, mul_d)
                    if (res_l == -1):
                        res_all_x += res_x
                        observed_value.append(res_x)
                    else:
                        # GRR 方法
                        res_all_l += res_l
                        count_l += 1
                for j in otherdata:
                    numbersofall += 1
                    if random.random() < gamma:
                        if mul_d < 3 * e_epsilon + 2:
                            res_l = GRR(e_epsilon, False, mul_d)
                        else:
                            res_l = OUE(e_epsilon, False)
                        res_all_l += res_l
                        count_l += 1
                    else:
                        # 随机选一个数字
                        res_x = random.random() * 2 * C - C
                        res_all_x += res_x
                        observed_value.append(res_x)

            noisedData, _ = np.histogram(observed_value, bins=pre_bins, range=(-C, C))

            theta_OPM_NoS = myEM_one(pre_bins, noisedData, Matrix, 10000, 0.0001 * e_epsilon)
            estimate_v_2 = 0
            estimate_f = 1 - theta_OPM_NoS[-1]
            theta_OPM_NoS = theta_OPM_NoS[0:-1]
            theta_OPM_NoS = theta_OPM_NoS / (sum(theta_OPM_NoS))
            # plt.plot([i for i in range(pre_bins)], theta_OPM_NoS)
            # plt.show()
            # print("sum",sum(theta_OPM_NoS))
            for i in range(pre_bins):
                estimate_v_2 += theta_OPM_NoS[i] * (i)
            estimate_v_2 = (estimate_v_2 / pre_bins - 0.5) * 2
            # print("EM:", sum_all[0], estimate_v_2, estimate_f)
            mse_f.append(((estimate_f - real_f) ** 2))
            mse_v_2.append((sum_all[0] - estimate_v_2) ** 2)

        # writefile("Income15", mul_d,acc1)
        # reslist.append(accuracy / numberExperi)
        newRes = result_append.copy()
        result_append.sort()
        offset = sum(result_append[0 + 1:len(result_append) - 1]) / (len(result_append) - 2) * 0.8
        acc3 = 0
        for i in range(len(newRes)):
            acc3 += (newRes[i] - offset) ** 2

        res_f.append(sum(mse_f) / numberExperi)
        res_v_1.append(sum(mse_v_1) / numberExperi)
        res_v_2.append(sum(mse_v_2) / numberExperi)
        # res_var1.append(np.var(var1))
        # res_var2.append(np.var(var2))

    # print(reslist)
    # print(res1)
    # print(res2)
    return res_f, res_v_1, res_v_2

def myEM_one(n, ns_hist, transform, max_iteration, loglikelihood_threshold):
    smoothing_factor = 2
    binomial_tmp = [1, 2, 1]
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

    theta = (np.ones(n + 1)) / (float(n)) / 8
    theta[-1:-1] = 0.00001
    theta[0:250] = 0.0000001
    theta[n] = 7 / 8
    theta_old = np.zeros(n + 1)
    transform2 = np.ones((n, n + 1))
    for i in range(n):
        for j in range(n):
            transform2[i, j] = transform[i, j]
        transform2[i, n] = 1 / n
    r = 0
    sample_size = sum(ns_hist)
    old_logliklihood = 0
    imporve = 100
    while LA.norm(theta_old - theta, ord=1) > 1 / sample_size / 100 and r < max_iteration:
        theta_old = np.copy(theta)
        X_condition = np.matmul(transform2, theta_old)
        TMP = transform2.T / X_condition
        P = np.copy(np.matmul(TMP, ns_hist))
        P = P * theta_old

        theta = np.copy(P / sum(P))
        if imporve > 0.0003:
            theta[0:n] = np.matmul(smoothing_matrix, theta[0:n])
            # print(sum(theta))
            if r > 5000:
                theta[n] = theta[n] * (1 + loglikelihood_threshold * 3)
        theta = theta / sum(theta)

        logliklihood = np.inner(ns_hist, np.log(np.matmul(transform2, theta)))
        imporve = logliklihood - old_logliklihood

        if r > 2000 and abs(imporve) < loglikelihood_threshold:
            break

        old_logliklihood = logliklihood

        r += 1
    # print("r=", r, "noise components:", theta[-1])
    return theta


def pcmm(e_epsilon, l, gamma, x, k, B, C, pr, mul_d):
    rand_value = random.random()
    res_l = -1
    res_x = 0
    if rand_value < gamma:
        if mul_d < 3 * e_epsilon + 2:
            res_l = GRR(e_epsilon, l, mul_d)
        else:
            res_l = OUE(e_epsilon, l)
        return res_x, res_l
    else:
        res_x = OPMnoise(x, k, B, pr, C)
    return res_x, res_l


def RR(e_epsilon, l):
    k = random.random()
    if k <= 1 / (1 + e_epsilon):
        return 1 - l
    else:
        return l


# l 为 0，1，2，3，4 从0 开始index
def GRR(e_epsilon, l, total):
    p_true = e_epsilon / (e_epsilon + total - 1)
    q = 1 / (e_epsilon + total - 1)
    if l == True:
        if random.random() < p_true:
            return 1
    else:
        if random.random() < q:
            return 1
    return 0


def generateMatrix(eps, binNumber):
    e_epsilon = math.e ** eps
    e_epsilon_sqrt = math.sqrt(e_epsilon)
    B = (e_epsilon_sqrt + 9) / (10 * e_epsilon_sqrt - 10)
    k = B * e_epsilon / (B * e_epsilon - 1 - B)
    C = k + B
    q = 1 / (2 * B * e_epsilon + 2 * k)
    avg_q = 1 / binNumber
    m = binNumber
    n = binNumber
    m_cell = (2 * C) / m
    transform = np.ones((binNumber, binNumber)) * q * m_cell
    q_value = transform[0, 0]
    p_value = q_value * e_epsilon
    pro_pass = 0
    left_index = 0
    right_index = 0
    # 计算 一列中多少个
    for i in range(0, n):

        if i == 0:
            a = int(B / C * n)
            reseverytime = 1 - a * p_value - (n - a) * q_value
            pro_pass = ((n - a - 1) * (p_value - q_value) + (p_value - q_value - reseverytime)) / (n - 1)
            transform[0:a, i] = p_value
            transform[a, i] = q_value + reseverytime
            right_index = a
            left_index = 0
        else:
            temp_right = transform[left_index, i - 1] - pro_pass
            if temp_right >= q_value:
                transform[left_index, i] = transform[left_index, i - 1] - pro_pass  # 左边减去 1
                if transform[right_index, i - 1] + pro_pass < p_value:
                    transform[right_index, i] = transform[right_index, i - 1] + pro_pass
                else:  # 说明要 right_index+1
                    overflow = transform[right_index, i - 1] + pro_pass - p_value
                    transform[right_index, i] = p_value
                    if right_index >= n - 1 and overflow < 1e-5:
                        transform[left_index + 1:right_index, i] = p_value
                        break
                    right_index += 1
                    transform[right_index, i] = q_value + overflow
            else:
                overflow = pro_pass - (transform[left_index, i - 1] - q_value)
                transform[left_index, i] = q_value
                left_index += 1
                transform[left_index, i] = p_value - overflow
                if transform[right_index, i - 1] + pro_pass < p_value:
                    transform[right_index, i] = transform[right_index, i - 1] + pro_pass
                else:  # 说明要 right_index+1
                    overflow = transform[right_index, i - 1] + pro_pass - p_value
                    transform[right_index, i] = p_value
                    if right_index == n - 1 and overflow < 1e-4:
                        transform[left_index + 1:right_index, i] = p_value
                        break
                    right_index += 1
                    transform[right_index, i] = q_value + overflow
            for jjj in range(left_index + 1, right_index):
                transform[jjj, i] = p_value
    return transform


def GRR_revision(e_epsilon, true_numbers, n, mul_d):
    p = e_epsilon / (e_epsilon + mul_d - 1)
    q = 1 / (e_epsilon + mul_d - 1)
    xv = (true_numbers / n - q) / (p - q)
    if xv > 1:
        xv = 1
    if xv < 0.001:
        xv = 0.001
    return xv


def OUE(e_epsilon, l):
    if l == 1:
        p_true = 1 / 2
        if random.random() < p_true:
            return 1
        return 0
    else:
        q = 1 / (e_epsilon + 1)
        if random.random() < q:
            return 1
        return 0


def OUE_revision(e_epsilon, obsever_numbers, n):
    p = 1 / 2
    q = 1 / (e_epsilon + 1)
    xv = (obsever_numbers / n - q) / (p - q)
    if xv > 1:
        xv = 1
    if xv < 0:
        xv = 0
    return xv


def OPMnoise(x, k, B, pr, C):
    lt = k * x - B
    rt = lt + 2 * B
    # 回答很好的条件
    if random.random() <= pr:
        res = random.random() * 2 * B + lt
    else:
        temp = random.random()
        ppp = (lt + C) / (2 * k)
        if ppp > temp:
            res = temp * (2 * k) - C
        else:
            res = rt + (temp - ppp) * (2 * k)
    return res


def sample(data, epsilon_list):
    print("this is bisample")
    reslist = []
    # epsilon_list = [0.5,0.75,1,1.25,2,2.5,3]
    temp = []
    if len(data) == 2:
        temp.extend(data[0])
        temp.extend(data[1])
    data = temp
    for epsilon in epsilon_list:
        GRR_epsilon = math.e ** (epsilon)
        temp_formular = (1 - GRR_epsilon) / (1 + GRR_epsilon)
        accuracy = 0

        # print(GRR_epsilon)
        for times in range(50):
            sum_average_all = 0
            counts = 0
            for value in data:
                if value != -1:
                    sum_average_all += value
                    counts += 1
            sum_average_all /= counts
            f_pos = 0
            f_neg = 0
            countf_pos = 0
            countf_neg = 0
            for value in data:
                s, b = sample_noise(value, temp_formular, 1 / (1 + GRR_epsilon))
                if s == 0:
                    f_neg += b
                    countf_neg += 1
                else:
                    f_pos += b
                    countf_pos += 1

            f_pos = f_pos / countf_pos
            f_neg = f_neg / countf_neg
            p = GRR_epsilon / (1 + GRR_epsilon)
            f222 = (1 - f_neg - f_pos) / (2 * p - 1)
            # print(f222)
            mean = (f_pos - f_neg) / (f_pos + f_neg + 2 * p - 2)
            # print(mean)
            accuracy += abs(mean - sum_average_all)
        accuracy = accuracy / 50
        reslist.append(accuracy)
    # print(reslist)
    return reslist


def sample_noise(j, temp, p2):
    pro = random.random()
    if pro > 0.5:
        pro = random.random()
        if j == -1:
            if pro > p2:
                return 0, 0
            else:
                return 0, 1
        if pro < j / 2 * temp + 0.5:
            return 0, 1
        else:
            return 0, 0
    else:
        pro = random.random()
        if j == -1:
            if pro > p2:
                return 1, 0
            else:
                return 1, 1
        if pro < j / 2 * temp * (-1) + 0.5:
            return 1, 1
        return 1, 0