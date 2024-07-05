# coding=utf-8
import pandas as pd
import numpy as np
import random
import sys
sys.path.append('../ldp_reduction')
from ldp_reduction.protocols.MR.PM_EM_reduction import perturb_PM
from ldp_reduction.protocols.numericalProtocols.PM import PM_distribution
from ldp_reduction.protocols.numericalProtocols.SW import sw
from ldp_reduction.protocols.MR.SW_EM_reduction import sw_reduct


def generate_Incomedata():
    Max_oo = 200000
    data1 = pd.read_csv("../ldp_reduction/protocols/opendata/incomeData/2013.csv", usecols=["A02650", "agi_stub"])
    datalist = []
    count = 0
    count2 = 0
    for index, row in data1.iterrows():
        expendure = row["A02650"]
        # print(expendure)
        if expendure > 0:
            if expendure > Max_oo:
                datalist.append(Max_oo * 1 / 2 * random.random())
                count += 1
            else:
                datalist.append(expendure)
    # print("2013: maxmin-Value:",max(datalist),min(datalist),len(datalist))

    data1 = pd.read_csv("../ldp_reduction/protocols/opendata/incomeData/2014.csv", usecols=["a02650"])
    datalist2 = []
    for index, row in data1.iterrows():
        expendure = row["a02650"]
        # print(expendure)
        if expendure > 0:
            if expendure > Max_oo:
                datalist2.append(Max_oo * 1 / 2 * random.random())
                count2 += 1
            else:
                datalist2.append(expendure)
    # print("counts: ",count,count2)
    # print("2014: maxmin-Value:",max(datalist2), min(datalist2))
    res = []
    datalist.extend(datalist2)
    Maxvalue, minValue = max(datalist), min(datalist)
    cha = Maxvalue - minValue
    for i in datalist:
        x = 2 * (i - minValue) / (cha) - 1
        res.append(round(x, 4))
    averages = sum(res) / len(res)
    # print(max(res),min(res),averages,(averages+1)/2 *cha,len(res))
    return res


def generate_SFCdata():
    data1 = pd.read_csv("../ldp_reduction/protocols/opendata/SFCDATA/sfc.csv", usecols=["Retirement"])
    datalist = []
    for index, row in data1.iterrows():
        expendure = row["Retirement"]
        # print(expendure)
        if expendure > 300000:
            datalist.append(300000)
        elif expendure > 1:
            datalist.append(expendure)
    # print(max(datalist),min(datalist),len(datalist))
    res = []
    Maxvalue, minValue = max(datalist), min(datalist)
    cha = Maxvalue - minValue
    for i in datalist:
        x = 2 * (i - minValue) / (cha) - 1
        res.append(round(x, 4))
    averages = sum(res) / len(res)
    # print(max(res),min(res),averages,(averages+1)/2 *cha)
    return res


sigma = 0.35
def generate_SMN1000(ifsmall=True):
    if ifsmall:
        n = 1000
    else:
        n = 5000
    res = []
    a = np.random.normal(0.7, sigma**2, int(n * 0.5))
    res.extend(a)
    b = np.random.normal(0.212, sigma**2, int(n * 0.5))
    res.extend(b)
    # res.extend([-0.5, -0.8, -0.3, -0.2, -0.1, -0.3, -0.1,-0.6])
    new_res = []
    for i in res:
        if i <= 1 and i >= -1:
            new_res.append(i)
    # print("mean:", sum(new_res) / len(new_res), "largest v", max(new_res))
    # ori_theta, _ = np.histogram(new_res, bins=124, range=(-1, 1))
    # plt.hist(ori_theta,bins=100,edgecolor='white')
    # plt.show()
    return new_res


exertime = 10
if __name__ == "__main__":

    ori_data = generate_SFCdata()
    # ori_data = generate_Incomedata()
    # ori_data = generate_SMN1000(ifsmall=False)

    for epsilon in [2]:  # 0.25,0.5,1,2
        var_EM, var_MR, wd_SW, wd_SW_MR = 0, 0, 0, 0
        wd_EM, wd_MR,wd_SW,wd_SW_MR = 0, 0, 0,0
        for _ in range(exertime):
            noised_samples = perturb_PM(ori_data, epsilon)
            var1,var2, wd1, wd2 = PM_distribution(epsilon, ori_data, noised_samples)
            wd_EM += wd1
            wd_MR += wd2
            var_EM += var1
            var_MR += var2
            acc_sw, wd3 = sw(np.array(ori_data), 0, 1, 1)
            acc_sw_MR, wd4 = sw_reduct(np.array(ori_data), 0, 1, 1)
            wd_SW += wd3
            wd_SW_MR += wd4
            print("\033[91m" + "wd of one estimate:", wd1, wd2, wd3,wd4)
            print("\033[0m")
            # print(res_duchi)
        print("The average WD ({0} times), when epsilon = {1}:".format(exertime,epsilon))
        print("WD of PM-EM:\t", wd_EM / exertime)
        print("WD of PM-MR:\t", wd_MR / exertime)
        print("WD of SW-EMS:\t", wd_SW / exertime)
        print("WD of SW-MR:\t", wd_SW_MR / exertime)
        print("The average MAE on distribution variance ({0} times)  of PM, when epsilon = {1}:".format(exertime, epsilon))
        print("VAR of PM-EM:\t", var_EM / exertime)
        print("VAR of PM-MR:\t", var_MR / exertime)
        # print("WD of SW-EMS:\t", wd_SW / exertime)
        # print("WD of SW-MR:\t", wd_SW_MR / exertime)