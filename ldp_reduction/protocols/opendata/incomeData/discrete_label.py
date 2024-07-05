import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def generate_data():
    data1 = pd.read_csv("2013.csv", usecols=["STATE", "N02650"])
    dicts = {}

    for index, row in data1.iterrows():
        if row["N02650"] < 1000:
            new_str = "1"
        elif row["N02650"] < 10000:
            new_str = "2"
        elif row["N02650"] < 100000:
            new_str = "3"
        elif row["N02650"] < 1000000:
            new_str = "4"
        else:
            new_str = "5"
        expendure = row["STATE"] + " " + new_str
        if expendure in dicts:
            dicts[expendure] += 1
        else:
            dicts[expendure] = 1
    print(dicts,len(dicts))
    # print("2013: maxmin-Value:",max(datalist),min(datalist),len(datalist))
    data1 = pd.read_csv("2014.csv", usecols=["state", "n02650"])
    for index, row in data1.iterrows():
        if row["n02650"] < 1000:
            new_str = "1"
        elif row["n02650"] < 10000:
            new_str = "2"
        elif row["n02650"] < 100000:
            new_str = "3"
        elif row["n02650"] < 1000000:
            new_str = "4"
        else:
            new_str = "5"
        expendure = row["state"]+" "+new_str
        if expendure in dicts:
            dicts[expendure] += 1
        else:
            dicts[expendure] = 1

    res = dicts.values()
    res = sorted(res,reverse=True)
    print(res,len(res))
    return res
if __name__ == "__main__":
    a = generate_data()
