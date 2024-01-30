import math
import numpy as np
from utils import calMAE, calMSE
from GRR import GRR, GRR_revision_no, norm_sub, GRR_revision_basecut, IIW
from GRR_EM import EM_GR
from MR.GRR_EM_reduction import EM_GRR_reduct_merge
import random

dataset = "Income"
Topk = 20
length = 60
if dataset == "Income":
    Topk = 100
    length = 250


def generate_noised_data(epsilon):
    # SFC dataset
    if dataset == "SFC":
        ori = np.array(
            [2828, 1981, 1807, 1745, 1730, 1718, 1718, 1670, 1630, 1619, 1608, 1607, 1541, 1541, 1532, 1530, 1509, 1453,
             1404, 1308, 686, 667, 553, 540, 534, 334, 163, 253, 250, 219, 194, 180, 150, 145, 140, 137, 137, 134, 134,
             131, 153, 113, 90, 45, 32, 25, 17, 14, 12, 13, 3, 10, 23, 20, 13, 13, 13, 12, 12, 10]
        )
        # random.shuffle(ori)
        # ori = np.array(
        # ori = np.array([500 for i in range(100)])
    else:  # Income dataset
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
        # ori = np.array([169, 5726, 148, 3298, 186, 6845, 4, 270, 3270,
        #                 6, 8, 434, 1035, 7367, 4, 9, 873, 48,
        #                 341, 4, 8883, 179, 2068, 174, 4, 5, 4,
        #                 3184, 81, 3910, 1176, 169, 179, 179, 7011, 8,
        #                 148, 2147, 8, 48, 5552, 94, 8, 4, 6246,
        #                 8, 167, 227, 2, 169, 9225, 320, 9, 2471,
        #                 1, 3240, 148, 3661, 2, 48, 256, 1469, 2,
        #                 6020, 81, 10, 1653, 8, 8, 9179, 2, 167,
        #                 1672, 48, 48, 1176, 1528, 2612, 693, 179, 2242,
        #                 6, 2, 1485, 642, 94, 1514, 4, 5829, 186,
        #                 4, 174, 316, 3767, 2063, 48, 6, 179, 714,
        #                 167, 7, 152, 2109, 124, 124, 6, 2, 5904,
        #                 13856, 81, 381, 38, 48, 1451, 12857, 10, 1488,
        #                 7968, 186, 94, 169, 2410, 8, 6410, 351, 5,
        #                 8, 169, 473, 4294, 1, 81, 4462, 8, 3884,
        #                 1111, 1021, 167, 2, 2667, 4, 8, 8, 8000,
        #                 5329, 1124, 4, 167, 8, 4, 2, 8, 4,
        #                 3049, 174, 167, 169, 94, 2156, 7, 124, 7,
        #                 2421, 124, 8718, 174, 152, 10, 94, 2064, 2,
        #                 3198, 3282, 4, 2, 8, 7, 124, 423, 8762,
        #                 94, 148, 152, 5215, 11463, 516, 152, 81, 179,
        #                 4184, 2728, 8, 4, 148, 2, 6, 379, 6,
        #                 6, 6, 2, 8, 124, 94, 152, 883, 5,
        #                 9, 887, 167, 5702, 81, 4952, 7, 3182, 693,
        #                 124, 186, 4, 179, 8, 8, 152, 13284, 186,
        #                 7, 1908, 2, 1080, 6523, 174, 152, 174, 174,
        #                 639, 8, 400, 10, 5661, 148, 186, 1539, 320,
        #                 2, 657, 81, 8, 2, 169, 200, 186, 3498,
        #                 148])
    number = len(ori)
    ori = np.rint(ori).astype(int)
    # print(number)
    print(len(ori), np.sum(ori))
    # print(sum(ori[0:16])/np.sum(ori))
    noised_data = []
    for i in range(len(ori)):
        for j in range(ori[i]):
            noised_data.append(GRR(math.e ** epsilon, i, number))
    noised_data = np.array(noised_data)
    # print(ori/np.sum(ori))
    return noised_data, number, ori / np.sum(ori)


if __name__ == "__main__":
    epsilon = 2.2
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
    exper_time = 3
    sum_all = np.array([0.0 for i in range(length)])
    sum_all2 = np.array([0.0 for i in range(length)])
    sum_all4 = np.array([0.0 for i in range(length)])
    sum_all3 = np.array([0.0 for i in range(length)])
    for i in range(exper_time):
        noised_data, number, ori = generate_noised_data(epsilon)

        weights4 = GRR_revision_no(noised_data, number, epsilon)
        acc4.append(calMAE(weights4, ori, length))
        acc41.append(calMAE(weights4, ori, topK=True))
        sum_all4 += np.array(weights4[0:length]) * len(noised_data)

        weights3 = GRR_revision_basecut(noised_data, number, epsilon)
        acc3.append(calMAE(weights3, ori))
        acc31.append(calMAE(weights3, ori, topK=True))

        weights5 = norm_sub(np.array(weights4) * len(noised_data), len(noised_data))
        acc5.append(calMAE(weights5 / len(noised_data), ori))
        acc51.append(calMAE(weights5 / len(noised_data), ori, topK=True))
        sum_all3 += np.array(weights5[0:length])

        weights = EM_GR(ori, noised_data, number, math.e ** epsilon, topK=Topk)
        acc1.append(calMAE(weights, ori))
        acc11.append(calMAE(weights, ori, topK=True))

        weights2 = EM_GRR_reduct_merge(ori, noised_data, number, math.e ** epsilon, weights, topk=Topk)
        sum_all += weights2[0:length] * len(noised_data)
        sum_all2 += weights[0:length] * len(noised_data)
        acc2.append(calMAE(weights2, ori))
        acc21.append(calMAE(weights2, ori, topK=True))

        # print(weights2)
        # weights3 = EM_GR_reduct(ori,noised_data, number, math.e ** epsilon,ifcut=True)

        weights6 = IIW(noised_data, number, epsilon)
        acc6.append(calMAE(weights6, ori ))
        acc61.append(calMAE(weights6, ori,topK=True))

        # print(weights5/sum(weights5))

    print("MAE: \t full domain \t TopK \t\t sdr ")
    print("GRR-Nonprocess:", sum(acc4) / exper_time, sum(acc41) / exper_time, np.std(acc4), np.std(acc41))
    print("GRR-BaseCut:", sum(acc3) / exper_time, sum(acc31) / exper_time, np.std(acc3), np.std(acc31))
    print("GRR-NormSub:", sum(acc5) / exper_time, sum(acc51) / exper_time, np.std(acc5), np.std(acc51))
    print("GRR-IIW:", sum(acc6) / exper_time, sum(acc61) / exper_time, np.std(acc6), np.std(acc61))
    print("GRR-EM:", sum(acc1) / exper_time, sum(acc11) / exper_time, np.std(acc1), np.std(acc11))
    print("Ours,GRR-MR:", sum(acc2) / exper_time, sum(acc21) / exper_time, np.std(acc2), np.std(acc21))
    import matplotlib.pyplot as plt

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
        # ori = np.array([169, 5726, 148, 3298, 186, 6845, 4, 270, 3270,
        #                 6, 8, 434, 1035, 7367, 4, 9, 873, 48,
        #                 341, 4, 8883, 179, 2068, 174, 4, 5, 4,
        #                 3184, 81, 3910, 1176, 169, 179, 179, 7011, 8,
        #                 148, 2147, 8, 48, 5552, 94, 8, 4, 6246,
        #                 8, 167, 227, 2, 169, 9225, 320, 9, 2471,
        #                 1, 3240, 148, 3661, 2, 48, 256, 1469, 2,
        #                 6020, 81, 10, 1653, 8, 8, 9179, 2, 167,
        #                 1672, 48, 48, 1176, 1528, 2612, 693, 179, 2242,
        #                 6, 2, 1485, 642, 94, 1514, 4, 5829, 186,
        #                 4, 174, 316, 3767, 2063, 48, 6, 179, 714,
        #                 167, 7, 152, 2109, 124, 124, 6, 2, 5904,
        #                 13856, 81, 381, 38, 48, 1451, 12857, 10, 1488,
        #                 7968, 186, 94, 169, 2410, 8, 6410, 351, 5,
        #                 8, 169, 473, 4294, 1, 81, 4462, 8, 3884,
        #                 1111, 1021, 167, 2, 2667, 4, 8, 8, 8000,
        #                 5329, 1124, 4, 167, 8, 4, 2, 8, 4,
        #                 3049, 174, 167, 169, 94, 2156, 7, 124, 7,
        #                 2421, 124, 8718, 174, 152, 10, 94, 2064, 2,
        #                 3198, 3282, 4, 2, 8, 7, 124, 423, 8762,
        #                 94, 148, 152, 5215, 11463, 516, 152, 81, 179,
        #                 4184, 2728, 8, 4, 148, 2, 6, 379, 6,
        #                 6, 6, 2, 8, 124, 94, 152, 883, 5,
        #                 9, 887, 167, 5702, 81, 4952, 7, 3182, 693,
        #                 124, 186, 4, 179, 8, 8, 152, 13284, 186,
        #                 7, 1908, 2, 1080, 6523, 174, 152, 174, 174,
        #                 639, 8, 400, 10, 5661, 148, 186, 1539, 320,
        #                 2, 657, 81, 8, 2, 169, 200, 186, 3498,
        #                 148])

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
    # Set font size for legend
    plt.legend(fontsize=17)
    plt.tight_layout()
    plt.savefig("frebias-b.pdf", bbox_inches='tight')
    plt.show()
