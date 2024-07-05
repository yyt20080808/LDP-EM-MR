import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def generate_data():
    data1 = pd.read_csv("sfc.csv", usecols=["Union"])
    values = {}

    for index, row in data1.iterrows():
        expendure = row["Union"]
        if expendure in values:
            values[expendure] += 1
        else:
            values[expendure] = 1
    print(values)
    res = []
    for k, v in values.items():
        res.append(v)
    res = sorted(res,reverse=True)
    print(np.array(res)/sum(res))
    print(res,len(res))
    return res
# job family code [3998, 3595, 2706, 2436, 2418, 1881, 1664, 1526, 1363, 1255, 1225, 1189, 1136, 1104, 1052, 1021, 988, 979, 836,
#          810, 784, 731, 633, 497, 461, 442, 410, 371, 367, 362, 274, 265, 254, 247, 237, 200, 194, 187, 180, 180, 147,
#          146, 116, 108, 85, 84, 71, 50, 30, 29, 19, 19, 16, 16, 15, 11, 5])
if __name__ == "__main__":
    a = generate_data()