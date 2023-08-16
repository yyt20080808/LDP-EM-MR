#coding=utf-8
import pandas as pd
import math
import numpy as np
from EM_reduction import perturb_PM
from PM import PM_res
def generate_data():
    data1 = pd.read_csv("./SFCDATA/sfc.csv", usecols=["Retirement"])
    datalist = []
    for index, row in data1.iterrows():
        expendure = row["Retirement"]
        # print(expendure)
        if expendure > 300000:
            datalist.append(300000)
        else:
            datalist.append(expendure)
    print(max(datalist),min(datalist),len(datalist))
    res = []
    Maxvalue,minValue =max(datalist), min(datalist)
    cha = Maxvalue-minValue
    for i in datalist:
        x = 2*(i - minValue) / (cha)-1
        res.append(round(x,4))
    averages = sum(res)/len(res)
    # print(max(res),min(res),averages,(averages+1)/2 *cha)
    return res

def generate_SMN1000():
    res = []
    a = np.random.normal(-0.5, math.sqrt(0.02), 1000)
    res.extend(a)
    b = np.random.normal(-0.9, math.sqrt(0.01), 1230)
    res.extend(b)
    c = np.random.normal(0.5, math.sqrt(0.02), 10)
    res.extend(c)
    # a = np.random.normal(-0.6, math.sqrt(0.07), 1000)
    # res.extend(a)
    # b = np.random.normal(-0.4, math.sqrt(0.05), 1000)
    # res.extend(b)
    # c = np.random.normal(0.2, math.sqrt(0.07), 1000)
    # res.extend(c)
    new_res = []
    for i in res:
        if i <= 1 and i >= -1:
            new_res.append(i)
    # print("mean:", sum(new_res) / len(new_res), "largest v", max(new_res))
    return new_res

if __name__ == "__main__":
    ori_data = generate_data()
    for epsilon in [0.5,1,2,3]:
        acc1,acc2,acc3 =0,0,0
        for _ in range(10):
            noised_samples = perturb_PM(ori_data,epsilon)
            acc_pm, acc_EM, acc_reduct = PM_res(epsilon,ori_data,noised_samples)
            acc1+=acc_pm
            acc2+=acc_EM
            acc3+=acc_reduct
        # print(res_duchi)
        print("accuracy when epsilon = {0}:".format(epsilon))
        print("MAE of Directly adding:\t",acc1/10)
        print("MAE of EM:\t",acc2/10)
        print("MAE of MR:\t",acc3/10)
