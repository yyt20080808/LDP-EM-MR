from MR.PM_EM_reduction import PM_E,calmean
import numpy as np
import matplotlib.pyplot as plt


def PM_res(epsilon, realdata, noisy_samples):
    new_ori = np.array(realdata)
    ori_theta, _ = np.histogram(new_ori, bins=1024, range=(-1, 1))
    realmean = np.sum(new_ori) / len(new_ori)

    acc_pm = abs(sum(noisy_samples) / len(noisy_samples)-realmean)

    theta2 = PM_E(noisy_samples, epsilon, randomized_bins=1024, domain_bins=1024, ifReduct=False)
    acc_ori = abs(realmean-calmean(theta2,1024))

    # theta = PM_E(noisy_samples, epsilon, randomized_bins=512, domain_bins=512, ifReduct=False)
    theta = PM_E(noisy_samples, epsilon, randomized_bins=1024, domain_bins=1024, ifReduct=True, remained=1 / 4)
    acc_cut = abs(calmean(theta,1024) - realmean)
    WD_EM, WD_MR = compareWasserstein_distances(ori_theta, theta2, theta)
    print("calmean:",realmean, sum(noisy_samples) / len(noisy_samples),calmean(theta2,1024),calmean(theta,1024),WD_EM,WD_MR)

    plt.plot([i for i in range(len(theta))], ori_theta, color='green')
    plt.plot([i for i in range(len(theta))], theta, color='red')
    plt.plot([i for i in range(len(theta))], theta2, color='darkgrey')
    plt.show()

    return acc_pm, acc_ori, acc_cut,WD_EM,WD_MR

from scipy.stats import wasserstein_distance


def compareWasserstein_distances(ori, a, b):
    wda = wasserstein_distance(ori,a)
    wdb = wasserstein_distance(ori,b)
    return wda,wdb