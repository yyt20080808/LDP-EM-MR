# coding=utf-8
import pandas as pd
import numpy as np
import random
from MR.PM_EM_reduction import perturb_PM
from PM import PM_res
import argparse


def generate_Incomedata():
    Max_oo = 200000
    data1 = pd.read_csv("../../src/opendata/incomeData/2013.csv", usecols=["A02650", "agi_stub"])
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

    data1 = pd.read_csv("../../src/opendata/incomeData/2014.csv", usecols=["a02650"])
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
    data1 = pd.read_csv("../../src/opendata/SFCDATA/sfc.csv", usecols=["Retirement"])
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


sigma = 0.3
def generate_SMN1000(ifsmall=True):
    if ifsmall:
        n = 1000
    else:
        n = 15000
    res = []
    a = np.random.normal(0.7, sigma**2, int(n * 0.5))
    res.extend(a)
    b = np.random.normal(0.212, sigma**2, int(n * 0.5))
    res.extend(b)
    res.extend([-0.5, -0.8, -0.3, -0.2, -0.1, -0.3, -0.1,-0.6])
    new_res = []
    for i in res:
        if i <= 1 and i >= -1:
            new_res.append(i)
    # print("mean:", sum(new_res) / len(new_res), "largest v", max(new_res))
    # ori_theta, _ = np.histogram(new_res, bins=124, range=(-1, 1))
    # plt.hist(ori_theta,bins=100,edgecolor='white')
    # plt.show()
    return new_res


exertime = 5
if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="这是一个示例程序")
    # parser.add_argument("echo", help="显示输入的字符串")
    # parser.add_argument("echo", help="显示输入的字符串")
    # parser.add_argument("echo", help="显示输入的字符串")
    # parser.add_argument("-eps", "--epsilon", type=float, help="一个整数")
    # ori_data = generate_SFCdata()
    # ori_data = generate_Incomedata()
    ori_data = generate_SMN1000(ifsmall=False)

    for epsilon in [0.26]:  # 0.25,0.5,1,2
        acc1, acc2, acc3 = 0, 0, 0
        wd_EM, wd_MR = 0, 0
        for _ in range(exertime):
            noised_samples = perturb_PM(ori_data, epsilon)
            acc_pm, acc_EM, acc_reduct, wd1, wd2 = PM_res(epsilon, ori_data, noised_samples)
            acc1 += acc_pm
            acc2 += acc_EM
            acc3 += acc_reduct
            wd_EM += wd1
            wd_MR += wd2
            print("\033[91m" + "MAE of one estimate:", acc_pm, acc_EM, acc_reduct)
            print("\033[0m")
            # print(res_duchi)
        print("accuracy when epsilon = {0}:".format(epsilon))
        print("MAE of Directly adding:\t", acc1 / exertime)
        print("MAE of EM:\t", acc2 / exertime)
        print("MAE of MR:\t", acc3 / exertime)
        print("WD of EM:\t", wd_EM / exertime)
        print("WD of MR:\t", wd_MR / exertime)
