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

def calMSE(a,b,topK):
    # length = 20
    # length = int(len(a)-1)
    # length = int(len(a)/2)
    res = 0
    for i in range(topK):
        res += (abs(a[i] - b[i]))
    return res / topK

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

import numpy as np
from scipy.stats import norm, uniform
import matplotlib.pyplot as plt


def generate_mixture_data(size, pi, mu1, sigma1, mu2, sigma2):
    uniform_data = np.random.uniform(0, 1, int(size * pi[0]))
    gaussian1_data = np.random.normal(mu1, sigma1, int(size * pi[1]))
    gaussian2_data = np.random.normal(mu2, sigma2, int(size * pi[2]))

    mixed_data = np.concatenate([uniform_data, gaussian1_data, gaussian2_data])
    np.random.shuffle(mixed_data)

    return mixed_data


def fit_mixture_model(data, num_iterations=100):
    pi = [1 / 3, 1 / 3, 1 / 3]  # 初始化混合系数
    mu = [0.2, 0.7, 0.9]  # 已知的高斯分布均值
    sigma = [0.1, 0.05, 0.1]  # 已知的高斯分布标准差
    a, b = 0.0, 1.0  # 均匀分布的区间

    for iteration in range(num_iterations):
        # E步
        posterior = expectation_step(data, pi, mu, sigma, a, b)

        # M步
        pi, mu, sigma, a, b = maximization_step(data, posterior)

    return pi, mu, sigma, a, b


def expectation_step(data, pi, mu, sigma, a, b):
    # E步：计算后验概率
    num_components = len(pi)
    posterior = np.zeros((len(data), num_components))

    for k in range(num_components - 1):
        posterior[:, k] = pi[k] * norm.pdf(data, mu[k], sigma[k])

    # 对于均匀分布部分
    posterior[:, -1] = pi[-1] * uniform.pdf(data, a, b)

    # 归一化
    posterior = posterior / np.sum(posterior, axis=1, keepdims=True)

    return posterior


def maximization_step(data, posterior):
    # M步：更新参数
    N = len(data)
    num_components = posterior.shape[1]

    # 更新混合系数
    pi = np.sum(posterior, axis=0) / N

    # 更新高斯分布的参数
    mu = np.sum(data[:, np.newaxis] * posterior[:, :-1], axis=0) / np.sum(posterior[:, :-1], axis=0)
    sigma = np.sqrt(
        np.sum(posterior[:, :-1] * (data[:, np.newaxis] - mu) ** 2, axis=0) / np.sum(posterior[:, :-1], axis=0))

    # 更新均匀分布的区间边界
    a = np.min(data[posterior[:, -1] > 0])
    b = np.max(data[posterior[:, -1] > 0])

    return pi, mu, sigma, a, b

#
# # 生成混合模型的数据
# np.random.seed(0)
# size = 10000
# true_pi = [1 / 3, 1 / 3, 1 / 3]
# true_mu = [0.2, 0.7, 0.9]
# true_sigma = [0.1, 0.05, 0.1]
# mixed_data = generate_mixture_data(size, true_pi, true_mu[0], true_sigma[0], true_mu[1], true_sigma[1])
#
# # 拟合混合模型
# pi_hat, mu_hat, sigma_hat, a_hat, b_hat = fit_mixture_model(mixed_data)
#
# print("True pi:", true_pi)
# print("True mu:", true_mu)
# print("True sigma:", true_sigma)
# print("Estimated pi:", pi_hat)
# print("Estimated mu:", mu_hat)
# print("Estimated sigma:", sigma_hat)
# print("Estimated a:", a_hat)
# print("Estimated b:", b_hat)
#
# # 绘制混合模型的拟合结果
# x = np.linspace(0, 1, 1000)
# pdf_true = true_pi[0] * uniform.pdf(x, 0, 1) + true_pi[1] * norm.pdf(x, true_mu[0], true_sigma[0]) + true_pi[
#     2] * norm.pdf(x, true_mu[1], true_sigma[1])
# pdf_estimated = pi_hat[0] * uniform.pdf(x, a_hat, b_hat) + pi_hat[1] * norm.pdf(x, mu_hat[0], sigma_hat[0]) + pi_hat[
#     2] * norm.pdf(x, mu_hat[1], sigma_hat[1])
#
# plt.hist(mixed_data, bins=30, density=True, alpha=0.5, label='True Mixture')
# plt.plot(x, pdf_true, 'r--', label='True PDF')
# plt.plot(x, pdf_estimated, 'g-', label='Estimated PDF')
# plt.legend()
# plt.show()
