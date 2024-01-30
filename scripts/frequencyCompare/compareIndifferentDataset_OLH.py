import numpy as np
from utils import calMAE, calMSE, norm_sub
from OLH import lh_perturb, lh_aggregate
from OLH_EM import olh_EM
from MR.OLH_EM_reduction import olh_EM_reduct

# choose dataset: SFC or Income
# used_datasets = "SFC"  # or "Income"
used_datasets = "Income"
length = 60
if used_datasets == "Income":
    length = 220

factor = 10


def generate_noised_data(epsilon):
    if used_datasets == "SFC":
        # SFC dataset
        ori = np.array(
            [2828, 1981, 1807, 1745, 1730, 1718, 1718, 1670, 1630, 1619, 1608, 1607, 1541, 1541, 1532, 1530, 1509, 1453,
             1404, 1308, 686, 667, 553, 540, 534, 334, 163, 253, 250, 219, 194, 180, 150, 145, 140, 137, 137, 134, 134,
             131, 153, 113, 90, 45, 32, 25, 17, 14, 12, 13, 3, 10, 23, 20, 13, 13, 13, 12, 12, 10])

        # ori = np.array([500 for i in range(100)])
        # ori = np.array(
        #     [1368, 1316, 1296, 1278,1444,628,931,115,146,115,122,3,4]
        # )

    else:
        # Income dataset
        ori = np.array(
            [13856, 13284, 12857, 11463, 9225, 9179, 8883, 8762, 8718, 8000, 7968, 7367, 7011, 6845, 6523, 6410, 6246,
             6020, 5904, 5829, 5726, 5702, 5661, 5552, 5329, 5215, 4952, 4462, 4294, 4184, 3910, 3884, 3767, 3661, 3498,
             3298, 3282, 3270, 3240, 3198, 3184, 3182, 3049, 2728, 2667, 2612, 2471, 2421, 2410, 2242, 2156, 2147, 2109,
             2068, 2064, 2063, 1908, 1672, 1653, 1539, 1528, 1514, 1488, 1485, 1469, 1451, 1176, 1176, 1124, 1111, 1080,
             1035, 1021, 887, 873, 714, 693, 693, 657, 642, 639, 516, 473, 434, 423, 400, 381, 379, 351, 341, 320, 320,
             316, 270, 256, 227, 200, 186, 179, 174, 169, 167, 152, 148, 124, 94, 81, 48, 38, 36, 33, 32, 31, 30, 26,
             25, 19, 19, 19, 18, 17, 16, 15, 14, 13, 12, 12, 12, 12, 12, 11, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
             10, 10, 10, 10, 10, 10, 10, 10, 10, 9, 9, 9, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
             7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2,
             2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1]

        )
    ori = np.floor_divide(ori, factor)
    number = len(ori)
    g =  max(3,int(np.exp(epsilon)) + 1)
    noisy_samples = lh_perturb(ori, g, 0.5)
    return noisy_samples, number, ori / np.sum(ori)


if __name__ == "__main__":
    for epsilon in [4]:
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
        sum_all = np.array([0.0 for i in range(length)])
        sum_all2 = np.array([0.0 for i in range(length)])
        sum_all4 = np.array([0.0 for i in range(length)])
        for i in range(exper_time):
            noised_data, number, ori = generate_noised_data(epsilon)
            # unbiased estimations from olh
            olh_est = lh_aggregate(noised_data, number, epsilon)
            acc1.append(calMAE(olh_est, ori, topK=True))
            acc11.append(calMAE(olh_est, ori))

            # norm-sub on olh_est
            NS_est = norm_sub(np.array(olh_est) * len(noised_data), len(noised_data)) / len(noised_data)
            acc4.append(calMAE(NS_est, ori, topK=True))
            acc41.append(calMAE(NS_est, ori))
            # print(acc1,acc11,acc4,acc41)
            # # # MLE by EM
            weights1 = olh_EM(noised_data, number, epsilon, ori, olh_est)
            acc2.append(calMAE(weights1, ori,topK=True))
            acc21.append(calMAE(weights1, ori))

            # Ours: MLE by EM-MR
            weights2 = olh_EM_reduct(noised_data, number, epsilon, ori, olh_est)
            acc3.append(calMAE(weights2, ori, topK=True))
            acc31.append(calMAE(weights2, ori))
            sum_all += weights2[0:length] * len(noised_data)*10
            sum_all2 += weights1[0:length] * len(noised_data)*10
            sum_all4 += np.array(olh_est[0:length]) * len(noised_data)*10

        print("MAE of epsilon = ", epsilon,"full domain","topK\t")
        print("OLH-unbiased:",  sum(acc11) / exper_time,sum(acc1) / exper_time, np.std(acc1), np.std(acc11))
        print("OLH-Normsub:", sum(acc41) / exper_time, sum(acc4) / exper_time, np.std(acc4), np.std(acc41))
        print("OLH-EM:",  sum(acc21) / exper_time,sum(acc2) / exper_time)
        print("OLH-MR:",  sum(acc31) / exper_time,sum(acc3) / exper_time, np.std(acc3), np.std(acc31))
        import matplotlib.pyplot as plt

        if used_datasets == "SFC":
            ori = np.array(
                [2828, 1981, 1807, 1745, 1730, 1718, 1718, 1670, 1630, 1619, 1608, 1607, 1541, 1541, 1532, 1530, 1509,
                 1453,
                 1404, 1308, 686, 667, 553, 540, 534, 334, 163, 253, 250, 219, 194, 180, 150, 145, 140, 137, 137, 134,
                 134,
                 131, 153, 113, 90, 45, 32, 25, 17, 14, 12, 13, 3, 10, 23, 20, 13, 13, 13, 12, 12, 10]
            )

        # ori = np.rint(ori / 10).astype(int)

        else:
            ori = np.array(
                [13856, 13284, 12857, 11463, 9225, 9179, 8883, 8762, 8718, 8000, 7968, 7367, 7011, 6845, 6523, 6410,
                 6246, 6020,
                 5904, 5829, 5726, 5702, 5661, 5552, 5329, 5215, 4952, 4462, 4294, 4184, 3910, 3884, 3767, 3661, 3498,
                 3298,
                 3282, 3270, 3240, 3198, 3184, 3182, 3049, 2728, 2667, 2612, 2471, 2421, 2410, 2242, 2156, 2147, 2109,
                 2068,
                 2064, 2063, 1908, 1672, 1653, 1539, 1528, 1514, 1488, 1485, 1469, 1451, 1176, 1176, 1124, 1111, 1080,
                 1035,
                 1021, 887, 873, 714, 693, 693, 657, 642, 639, 516, 473, 434, 423, 400, 381, 379, 351, 341, 320, 320,
                 316, 270,
                 256, 227, 200, 186, 179, 174, 169, 167, 152, 148, 124, 94, 81, 48, 38, 36, 33, 32, 31, 30, 26, 25, 19,
                 19, 19,
                 18, 17, 16, 15, 14, 13, 12, 12, 12, 12, 12, 11, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
                 10, 10,
                 10, 10, 10, 10, 9, 9, 9, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 7, 7, 7, 7, 7,
                 7, 6, 6,
                 6, 6, 6, 6, 6, 6, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                 2, 2, 2,
                 2, 1]
            )
        # ori = np.rint(ori / 10).astype(int)
        # ori = np.array(
        #     [1368, 1316, 1296, 1278,1444,628,931,31,46,115,122,3,4]
        # )
        # ori = np.floor_divide(ori, factor)

        plt.figure(figsize=(10, 6))
        plt.plot(range(len(ori)), ori, color="black", label="true count")
        plt.plot(range(len(sum_all)), sum_all / exper_time, color="red",label="ours")
        plt.plot(range(len(sum_all2)), sum_all2 / exper_time, color="green",linestyle='--',label="EM estimated count")
        plt.plot(range(len(sum_all4)), sum_all4 / exper_time, color="blue",label="unbiased estimated count")

        # Your plotting code here (unchanged)

        plt.xlabel('Index', fontsize=18)
        plt.ylabel('Value', fontsize=18)
        plt.grid(True)
        plt.xticks(fontsize=17)
        plt.yticks(fontsize=17)
        # Set font size for legend
        plt.legend(fontsize=17)
        plt.tight_layout()
        plt.savefig("frebias-d.pdf", bbox_inches='tight')
        plt.show()
