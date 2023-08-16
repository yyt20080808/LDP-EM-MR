from EM_reduction import PM_E,calmean
import numpy as np
def PM_res(epsilon, realdata, noisy_samples):
    new_ori = np.array(realdata)

    realmean = np.sum(new_ori) / len(new_ori)

    acc_pm = abs(sum(noisy_samples) / len(noisy_samples)-realmean)

    theta2 = PM_E(noisy_samples, epsilon, randomized_bins=1024, domain_bins=1024, ifReduct=False)
    acc_ori = abs(realmean-calmean(theta2))

    theta = PM_E(noisy_samples, epsilon, randomized_bins=1024, domain_bins=1024, ifReduct=True, remained=1 / 2)
    acc_cut = abs(calmean(theta) - realmean)
    return acc_pm, acc_ori, acc_cut