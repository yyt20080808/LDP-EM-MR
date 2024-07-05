import numpy as np
import sys
sys.path.append('../ldp_reduction')
from ldp_reduction.protocols.categoricalProtocols.utils import calMAE, calMSE, norm_sub
from ldp_reduction.protocols.categoricalProtocols.OLH import lh_perturb, lh_aggregate
from ldp_reduction.protocols.categoricalProtocols.OLH_EM import olh_EM
from ldp_reduction.protocols.MR.OLH_EM_reduction import olh_EM_reduct

# from OLH_IBU import IBU_est
# choose dataset: SFC or Income
# used_datasets = "SFC"  # or "Income"
used_datasets = "SFC"
length = 60
if used_datasets == "Income":
    length = 220

factor = 1


def generate_noised_data(epsilon):
    if used_datasets == "SFC":
        # SFC dataset
        ori = np.array(
            [2828, 1981, 1807, 1745, 1730, 1718, 1718, 1670, 1630, 1619, 1608, 1607, 1541, 1541, 1532, 1530, 1509, 1453,
             1404, 1308, 686, 667, 553, 540, 534, 334, 163, 153, 150, 119, 194, 110, 110, 145, 140, 97, 137, 34, 87,
             101, 113, 103, 90, 45, 32, 25, 17, 14, 12, 13, 3, 10, 23, 20, 13, 13, 13, 12, 12, 10])

        # ori = np.array([500 for i in range(100)])
        # ori = np.array(
        #     [1368, 1316, 1296, 1278,1444,1628,1931,215,246,]
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
    g = max(3, int(np.exp(epsilon)) + 1)
    noisy_samples = lh_perturb(ori, g, 0.5)
    # print("1. noised reports generated.")
    return noisy_samples, number, ori / np.sum(ori)


if __name__ == "__main__":
    for epsilon in [1]:
        acc1, acc11, acc2, acc21, acc3, acc31, acc4, acc41, acc5, acc51, acc6, acc61 = [[] for _ in range(12)]
        exper_time = 10
        sum_all = np.array([0.0 for i in range(length)])
        sum_all2 = np.array([0.0 for i in range(length)])
        sum_all4 = np.array([0.0 for i in range(length)])
        for i in range(exper_time):
            noised_data, number, ori = generate_noised_data(epsilon)
            # unbiased estimations from olh
            olh_est = lh_aggregate(noised_data, number, epsilon)
            acc1.append(calMSE(olh_est, ori))
            acc11.append(calMSE(olh_est, ori))

            # norm-sub on olh_est
            NS_est = norm_sub(np.array(olh_est) * len(noised_data), len(noised_data)) / len(noised_data)
            acc5.append(calMSE(NS_est, ori))
            acc51.append(calMSE(NS_est, ori))
            # print("Unbiased estimation finished.")
            # IBU's implementation
            # IBUOLH_est = IBU_est(noised_data,epsilon,number)
            # acc4.append(calMSE(IBUOLH_est, ori))
            # acc41.append(calMSE(IBUOLH_est, ori))

            # # # MLE by EM
            weights1 = olh_EM(noised_data, number, epsilon, olh_est,ori)
            acc2.append(calMSE(weights1, ori))
            acc21.append(calMSE(weights1, ori))
            sum_all2 += weights1[0:length] * len(noised_data)
            # #
            # # Ours: MLE by EM-MR
            weights2 = olh_EM_reduct(noised_data, number, epsilon, NS_est)
            acc3.append(calMSE(weights2, ori))
            acc31.append(calMSE(weights2, ori))
            sum_all += weights2[0:length] * len(noised_data)
            sum_all4 += np.array(olh_est[0:length]) * len(noised_data)

        print("Results.", "MSE of epsilon = ", epsilon)
        # print("OLH-unbiased:",  sum(acc11) / exper_time,sum(acc1) / exper_time)
        # print("OLH-Normsub:", sum(acc51) / exper_time, sum(acc5) / exper_time)
        # print("OLH-IBU:", sum(acc41) / exper_time, sum(acc4) / exper_time, acc41)
        # print("OLH-EM:",  sum(acc21) / exper_time,sum(acc2) / exper_time)
        # print("OLH-MR:",  sum(acc31) / exper_time,sum(acc3) / exper_time)
        print("OLH-unbiased:", sum(acc11) / exper_time)
        print("OLH-Normsub:", sum(acc51) / exper_time)
        print("OLH-IBU:", sum(acc41) / exper_time)
        print("OLH-EM:", sum(acc21) / exper_time)
        print("OLH-MR:", sum(acc31) / exper_time)
        import matplotlib.pyplot as plt

        if used_datasets == "SFC":
            ori = np.array(
                [2828, 1981, 1807, 1745, 1730, 1718, 1718, 1670, 1630, 1619, 1608, 1607, 1541, 1541, 1532, 1530, 1509,
                 1453,
                 1404, 1308, 686, 667, 553, 540, 534, 334, 163, 153, 150, 119, 194, 110, 110, 145, 140, 97, 137, 34, 87,
                 101, 113, 103, 90, 45, 32, 25, 17, 14, 12, 13, 3, 10, 23, 20, 13, 13, 13, 12, 12, 10])

            # ori = np.rint(ori / factor).astype(int)

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
        ori = np.floor_divide(ori, factor)

        plt.figure(figsize=(10, 6))
        plt.plot(range(len(ori)), ori, color="black", label="true count")
        plt.plot(range(len(sum_all)), sum_all / exper_time, color="red", label="ours")
        plt.plot(range(len(sum_all2)), sum_all2 / exper_time, color="green", linestyle='--', label="EM estimated count")
        plt.plot(range(len(sum_all4)), sum_all4 / exper_time, color="blue", label="unbiased estimated count")

        plt.xlabel('Index', fontsize=18)
        plt.ylabel('Value', fontsize=18)
        plt.grid(True)
        plt.xticks(fontsize=17)
        plt.yticks(fontsize=17)
        # Set font size for legend
        plt.legend(fontsize=17)
        plt.tight_layout()
        # plt.savefig("frebias-d.pdf", bbox_inches='tight')
        plt.show()
