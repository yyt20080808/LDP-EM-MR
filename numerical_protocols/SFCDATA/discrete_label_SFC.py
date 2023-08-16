import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def generate_data():
    data1 = pd.read_csv("../SFCDATA/sfc.csv", usecols=["Job Family Code"])
    values = {}

    for index, row in data1.iterrows():
        expendure = row["Job Family Code"]
        if expendure in values:
            values[expendure] += 1
        else:
            values[expendure] = 1
    print(values)
    res = []
    for k, v in values.items():
        res.append(v)
    print(res)
    # print("2013: maxmin-Value:",max(datalist),min(datalist),len(datalist))

    # data1 = pd.read_csv("../incomeData/2014.csv", usecols=["zipcode"])
    # datalist2 = []
    # for index, row in data1.iterrows():
    #     expendure = row["a02650"]
    #     # print(expendure)
    #     if expendure > 0:
    #         if expendure > Max_oo:
    #             datalist2.append(Max_oo*random.random())
    #             count2 += 1
    #         else:
    #             datalist2.append(expendure)
    # print("counts: ",count,count2)
    # print("2014: maxmin-Value:",max(datalist2), min(datalist2))
    # res = []
    # datalist.extend(datalist2)
    # Maxvalue,minValue =max(datalist), min(datalist)
    # cha = Maxvalue-minValue
    # for i in datalist:
    #     x = 2*(i - minValue) / (cha)-1
    #     res.append(round(x,4))
    # averages = sum(res)/len(res)
    # print(max(res),min(res),averages,(averages+1)/2 *cha,len(res))
    # return res
if __name__ == "__main__":
    a = generate_data()