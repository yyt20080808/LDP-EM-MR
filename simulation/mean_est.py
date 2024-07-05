# coding=utf-8
import pandas as pd
import numpy as np
import random
import sys
sys.path.append('../ldp_reduction')
from ldp_reduction.protocols.MR.PM_EM_reduction import perturb_PM
from ldp_reduction.protocols.numericalProtocols.PM import PM_res
from ldp_reduction.protocols.numericalProtocols.SW import sw
from ldp_reduction.protocols.numericalProtocols.SR import duchi

def generate_Incomedata():
    Max_oo = 200000
    data1 = pd.read_csv("../ldp_reduction/protocols/opendata/incomeData/2013.csv", usecols=["A02650", "agi_stub"])
    datalist = []
    count = 0
    count2 = 0
    for index, row in data1.iterrows():
        expendure = row["A02650"]
        if expendure > 0:
            if expendure > Max_oo:
                datalist.append(Max_oo * 1 / 2 * random.random())
                count += 1
            else:
                datalist.append(expendure)

    data1 = pd.read_csv("../ldp_reduction/protocols/opendata/incomeData/2014.csv", usecols=["a02650"])
    datalist2 = []
    for index, row in data1.iterrows():
        expendure = row["a02650"]
        if expendure > 0:
            if expendure > Max_oo:
                datalist2.append(Max_oo * 1 / 2 * random.random())
                count2 += 1
            else:
                datalist2.append(expendure)
    res = []
    datalist.extend(datalist2)
    Maxvalue, minValue = max(datalist), min(datalist)
    cha = Maxvalue - minValue
    for i in datalist:
        x = 2 * (i - minValue) / (cha) - 1
        res.append(round(x, 4))
    return res


def generate_SFCdata():
    data1 = pd.read_csv("../ldp_reduction/protocols/opendata/SFCDATA/sfc.csv", usecols=["Retirement"])
    datalist = []
    for index, row in data1.iterrows():
        expendure = row["Retirement"]
        if expendure > 300000:
            datalist.append(300000)
        elif expendure > 1:
            datalist.append(expendure)
    res = []
    Maxvalue, minValue = max(datalist), min(datalist)
    cha = Maxvalue - minValue
    for i in datalist:
        x = 2 * (i - minValue) / (cha) - 1
        res.append(round(x, 4))
    return res


sigma = 0.35
def generate_SMN1000(ifsmall=True):
    if ifsmall:
        n = 1000
    else:
        n = 5000
    res = []
    a = np.random.normal(0.4, sigma**2, int(n * 0.4))
    res.extend(a)
    b = np.random.normal(0.712, sigma**2, int(n * 0.6))
    res.extend(b)
    new_res = []
    for i in res:
        if i <= 1 and i >= -1:
            new_res.append(i)
    return new_res


if __name__ == "__main__":
    # ori_data = generate_SFCdata()
    exetimes = 10
    # ori_data = generate_Incomedata()
    ori_data = generate_SMN1000(ifsmall=False)

    for epsilon in [1]:  # 0.25,0.5,1,2
        acc1, acc2, acc3, acc4,acc5 = 0, 0, 0, 0,0
        for _ in range(exetimes):
            noised_samples = perturb_PM(ori_data, epsilon)
            acc_pm, acc_EM, acc_reduct, wd1, wd2 = PM_res(epsilon, ori_data, noised_samples)
            acc1 += acc_pm
            acc2 += acc_EM
            acc3 += acc_reduct
            acc_sw, wd_3 = sw(np.array(ori_data), 0, 1, 1)
            acc4 += acc_sw
            acc5 += duchi(ori_data,epsilon,0)
            # print("\033[91m" + "MSE of one estimate:", acc_pm, acc_EM, acc_reduct, acc_sw)
            # print("\033[0m")
            # print(res_duchi)
        print("The averaged MSE ({0} times) of PM's when epsilon = {1}:".format(exetimes, epsilon))
        print("MSE of Directly adding:\t", acc1 / exetimes)
        print("MSE of EM:\t", acc2 / exetimes)
        print("MSE of MR:\t", acc3 / exetimes)
        print("Existing method, when epsilon = {0}:".format(epsilon))
        print("MSE of SW-EMS:\t", acc4 / exetimes)
        print("MSE of Duchi's:\t", acc5 / exetimes)
