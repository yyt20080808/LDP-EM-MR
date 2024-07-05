import math
import numpy as np
import sys
sys.path.append('../ldp_reduction')
from ldp_reduction.protocols.categoricalProtocols.utils import calMAE, calMSE
from ldp_reduction.protocols.categoricalProtocols.GRR import GRR, GRR_revision_no, norm_sub, GRR_revision_basecut, IIW
from ldp_reduction.protocols.categoricalProtocols.GRR_EM import EM_GR
from ldp_reduction.protocols.MR.GRR_EM_reduction import EM_GRR_reduct_merge
import matplotlib.pyplot as plt

dataset = "Income"
if dataset == "SFC":
    length = 60
elif dataset == "Income":
    length = 250
else:
    length = 12

def generate_noised_data(epsilon,dataset):
    # SFC dataset
    if dataset == "SFC":
        ori = np.array(
            [2828, 1981, 1807, 1745, 1730, 1718, 1718, 1670, 1630, 1619, 1608, 1607, 1541, 1541, 1532, 1530, 1509, 1453,
             1404, 1308, 686, 667, 553, 540, 534, 334, 163, 153, 150, 119, 194, 110, 110, 145, 140, 97, 137, 34, 87,
             101, 113, 103, 90, 45, 32, 25, 17, 14, 12, 13, 3, 10, 23, 20, 13, 13, 13, 12, 12, 10]
        )
    elif dataset == "Income":  # Income dataset
        ori = np.array(
            [13856, 13284, 12857, 11463, 9225, 9179, 8883, 8762, 8718, 8000, 7968, 7367, 7011, 6845, 6523, 6410, 6246,
             6020,
             5904, 5829, 5726, 5702, 5661, 5552, 5329, 5215, 4952, 4462, 4294, 4184, 3910, 3884, 3767, 3661, 3498, 3298,
             3282, 3270, 3240, 3198, 3184, 3182, 3049, 2728, 2667, 2612, 2471, 2421, 2410, 2242, 2156, 2147, 2109, 2068,
             2064, 2063, 1908, 1672, 1653, 1539, 1528, 1514, 1488, 1485, 1469, 1451, 1176, 1176, 1124, 1111, 1080, 1035,
             1021, 887, 873, 714, 693, 693, 657, 642, 639, 516, 473, 434, 423, 400, 381, 379, 351, 341, 320, 320, 316,
             270,
             256, 227, 200, 186, 179, 174, 169, 167, 152, 148, 124, 94, 81, 48, 38, 186, 179, 174, 169, 167, 152, 148,
             124, 94, 81, 48,
             186, 179, 174, 169, 167, 152, 148, 124, 94, 81, 48, 186, 179, 174, 169, 167, 152, 148, 124, 94, 81, 48,
             186, 179, 174, 169, 167, 152, 148, 124, 94, 81, 48, 186, 179, 174, 169, 167, 152, 148, 124, 94, 81, 48,
             186, 179, 174, 169, 167, 152, 148, 124, 94, 81, 48,
             2, 1, 10, 10, 10, 10, 9, 9, 9, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 7, 7, 7, 7,
             7, 7, 6, 6,
             6, 6, 6, 6, 6, 6, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
             2,
             2, 1]
        )
    else:
        ori = np.array([10000,2005,123,134,113,62,14,55,302,111,230,221,334,221])

    number = len(ori)
    ori = np.rint(ori).astype(int)
    noised_data = []
    for i in range(len(ori)):
        for j in range(ori[i]):
            noised_data.append(GRR(math.e ** epsilon, i, number))
    noised_data = np.array(noised_data)
    return noised_data, number, ori / np.sum(ori)


if __name__ == "__main__":
    epsilon = 1.5
    acc1, acc11, acc2, acc21, acc3, acc31, acc4, acc41, acc5, acc51, acc6, acc61 = [[] for _ in range(12)]

    exper_time = 10
    sum_all = np.array([0.0 for i in range(length)])
    sum_all2 = np.array([0.0 for i in range(length)])
    sum_all4 = np.array([0.0 for i in range(length)])
    sum_all3 = np.array([0.0 for i in range(length)])
    for i in range(exper_time):
        noised_data, number, ori = generate_noised_data(epsilon)

        weights4 = GRR_revision_no(noised_data, number, epsilon)
        acc4.append(calMSE(weights4, ori, length))
        acc41.append(calMSE(weights4, ori, topK=True))
        sum_all4 += np.array(weights4[0:length]) * len(noised_data)

        weights3 = GRR_revision_basecut(noised_data, number, epsilon)
        acc3.append(calMSE(weights3, ori))
        acc31.append(calMSE(weights3, ori, topK=True))

        weights5 = norm_sub(np.array(weights4) * len(noised_data), len(noised_data))
        acc5.append(calMSE(weights5 / len(noised_data), ori))
        acc51.append(calMSE(weights5 / len(noised_data), ori, topK=True))
        sum_all3 += np.array(weights5[0:length])
        #
        weights = EM_GR(ori, noised_data, number, math.e ** epsilon)
        acc1.append(calMSE(weights, ori))
        acc11.append(calMSE(weights, ori, topK=True))

        weights2 = EM_GRR_reduct_merge( noised_data, number, math.e ** epsilon)
        sum_all += weights2[0:length] * len(noised_data)
        sum_all2 += weights[0:length] * len(noised_data)
        acc2.append(calMSE(weights2, ori))
        acc21.append(calMSE(weights2, ori, topK=True))

        weights6 = IIW(noised_data, number, epsilon)
        acc6.append(calMSE(weights6, ori ))
        acc61.append(calMSE(weights6, ori,topK=True))


    print("MSE: \t full domain ")
    # print("GRR-Nonprocess:", sum(acc4) / exper_time, sum(acc41) / exper_time, np.std(acc4), np.std(acc41))
    # print("GRR-BaseCut:", sum(acc3) / exper_time, sum(acc31) / exper_time, np.std(acc3), np.std(acc31))
    # print("GRR-NormSub:", sum(acc5) / exper_time, sum(acc51) / exper_time, np.std(acc5), np.std(acc51))
    # print("GRR-IIW:", sum(acc6) / exper_time, sum(acc61) / exper_time, np.std(acc6), np.std(acc61))
    # print("GRR-EM:", sum(acc1) / exper_time, sum(acc11) / exper_time, np.std(acc1), np.std(acc11))
    # print("Ours,GRR-MR:", sum(acc2) / exper_time, sum(acc21) / exper_time, np.std(acc2), np.std(acc21))
    print("GRR-Nonprocess:", sum(acc4) / exper_time)
    print("GRR-BaseCut:", sum(acc3) / exper_time)
    print("GRR-NormSub:", sum(acc5) / exper_time)
    print("GRR-IIW:", sum(acc6) / exper_time)
    print("GRR-EM:", sum(acc1) / exper_time)
    print("Ours,GRR-MR:", sum(acc2) / exper_time)

    if dataset == "SFC":

        ori = np.array(
            [2828, 1981, 1807, 1745, 1730, 1718, 1718, 1670, 1630, 1619, 1608, 1607, 1541, 1541, 1532, 1530, 1509, 1453,
             1404, 1308, 686, 667, 553, 540, 534, 334, 163, 253, 250, 219, 194, 180, 150, 145, 140, 137, 137, 134, 134,
             131, 153, 113, 90, 45, 32, 25, 17, 14, 12, 13, 3, 10, 23, 20, 13, 13, 13, 12, 12, 10])

    else:
        ori = np.array(
            [13856, 13284, 12857, 11463, 9225, 9179, 8883, 8762, 8718, 8000, 7968, 7367, 7011, 6845, 6523, 6410, 6246,
             6020,
             5904, 5829, 5726, 5702, 5661, 5552, 5329, 5215, 4952, 4462, 4294, 4184, 3910, 3884, 3767, 3661, 3498, 3298,
             3282, 3270, 3240, 3198, 3184, 3182, 3049, 2728, 2667, 2612, 2471, 2421, 2410, 2242, 2156, 2147, 2109, 2068,
             2064, 2063, 1908, 1672, 1653, 1539, 1528, 1514, 1488, 1485, 1469, 1451, 1176, 1176, 1124, 1111, 1080, 1035,
             1021, 887, 873, 883, 714, 693, 693, 657, 642, 639, 516, 473, 434, 423, 400, 381, 379, 351, 341, 320, 320,
             316,
             270,
             256, 227, 200, 186, 179, 174, 169, 167, 152, 148, 124, 94, 81, 48, 38, 186, 179, 174, 169, 167, 152, 148,
             124, 94, 81, 48,
             186, 179, 174, 169, 167, 152, 148, 124, 94, 81, 48, 186, 179, 174, 169, 167, 152, 148, 124, 94, 81, 48,
             186, 179, 174, 169, 167, 152, 148, 124, 94, 81, 48, 186, 179, 174, 169, 167, 152, 148, 124, 94, 81, 48,
             186, 179, 174, 169, 167, 152, 148, 124, 94, 81, 48,
             2, 1, 10, 10, 10, 10, 9, 9, 9, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 7, 7, 7, 7,
             7, 7, 6, 6,
             6, 6, 6, 6, 6, 6, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
             2,
             2, 1]
        )

    ori = np.rint(ori).astype(int)
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(ori)), ori, color="black", label="true count")
    plt.plot(range(len(sum_all)), sum_all / exper_time, color="red", label="ours")
    plt.plot(range(len(sum_all2)), sum_all2 / exper_time, color="green", linestyle='--', label="EM estimated count")
    plt.plot(range(len(sum_all4)), sum_all4 / exper_time, color="blue", label="unbiased estimated count")
    # plt.plot(range(len(sum_all4)), sum_all3 / exper_time, color="orange", label="Normsub count")
    # Your plotting code here (unchanged)

    plt.xlabel('Index', fontsize=18)
    plt.ylabel('Value', fontsize=18)
    # plt.grid(True)
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)
    # # Set font size for legend
    plt.legend(fontsize=17)
    plt.tight_layout()
    # plt.savefig("frebias-b.pdf", bbox_inches='tight')
    plt.show()
