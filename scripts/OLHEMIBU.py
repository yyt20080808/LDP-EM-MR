import numpy as np
from utils import calMAE, calMSE, norm_sub
from OLH import lh_perturb, lh_aggregate
from OLH_EM import olh_EM
from MR.OLH_EM_reduction import olh_EM_reduct
from OLH_IBU import IBU_est
# choose dataset:
# np.random.seed(22)
used_datasets = "Poisson"#"Gaussuan" # or "Poisson"
def generate_noised_data(epsilon):
    if used_datasets == "Gaussuan":
        samples = np.random.normal(0, 1, 10000)
        K=20
        ori,_ = np.histogram(samples,bins=K)
    else:
        samples =  np.random.poisson(3, 10000)
        K = 20
        ori, _ = np.histogram(samples, bins=K)
    number = len(ori)
    g =  max(2,int(np.exp(epsilon)) + 1)
    noisy_samples = lh_perturb(ori, g, 0.5)
    return noisy_samples, number, ori / np.sum(ori)


if __name__ == "__main__":
    for epsilon in [3]:
        acc1 = []
        acc11 = []
        acc2 = []
        acc21 = []
        acc3 = []
        acc31 = []
        acc4 = []
        acc41 = []
        acc5 = []
        acc51 = []
        acc6 = []
        acc61 = []
        exper_time = 1
        for i in range(exper_time):
            noised_data, number, ori = generate_noised_data(epsilon)
            # unbiased estimations from olh
            olh_est = lh_aggregate(noised_data, number, epsilon)
            acc1.append(calMAE(olh_est, ori))
            acc11.append(calMSE(olh_est, ori))

            # norm-sub on olh_est
            NS_est = norm_sub(np.array(olh_est) * len(noised_data), len(noised_data)) / len(noised_data)

            # IBU's implementation
            IBUOLH_est = IBU_est(noised_data,epsilon,number)
            acc4.append(calMAE(IBUOLH_est, ori))
            acc41.append(calMSE(IBUOLH_est, ori))
            # print(IBUOLH_est)
            # # # MLE by EM
            weights1 = olh_EM(noised_data, number, epsilon, olh_est)
            acc2.append(calMAE(weights1, ori))
            acc21.append(calMSE(weights1, ori))
            #
            # # Ours: MLE by EM-MR
            weights2 = olh_EM_reduct(noised_data, number, epsilon, NS_est)
            acc3.append(calMAE(weights2, ori))
            acc31.append(calMSE(weights2, ori))


        print("full domain","MSE of epsilon = ", epsilon,"\t\tMAE\t")
        print("OLH-unbiased:",  sum(acc11) / exper_time,"\t", sum(acc1) / exper_time)
        print("OLH-IBU:", sum(acc41) / exper_time, "\t",sum(acc4) / exper_time)
        print("OLH-EM:",  sum(acc21) / exper_time, "\t",sum(acc2) / exper_time)
        print("OLH-MR:",  sum(acc31) / exper_time, "\t",sum(acc3) / exper_time)
        import matplotlib.pyplot as plt

