
import matplotlib.pyplot as plt

import numpy as np
import math
import random
from numpy import linalg as LA
from ldp_reduction.protocols.MR.PCKV_EM import SVEPM
import sys
sys.path.append('../ldp_reduction')
from ldp_reduction.protocols.KeyValueProtocols.PCKVUE import pckv_UE
from ldp_reduction.protocols.KeyValueProtocols.PCKVGRR import pckv_GRR

def generate_Data(alpha,type):
    res = []
    number = 100000
    basic = int(number * alpha)
    if type=="gaussian":
        a = np.random.normal(0.8, 0.35**2, int(basic/2))
        b = np.random.normal(0.3, 0.35 ** 2, int(basic/2))
    else:
        a = np.random.uniform(-1,1,int(basic/2))
        b = np.random.uniform(-1,1,int(basic/2))
    newa = []
    for value in a:
        if 1 >= value >= -1:
            newa.append(value)
        else:
            v = random.random()
            newa.append(v)
    for value in b:
        if 1 >= value >= -1:
            newa.append(value)
        else:
            v = random.random()
            newa.append(v)
    newb = [0 for i in range(number - basic)]
    res.append(newa)
    res.append(newb)
    return res


if __name__ == "__main__":

    epslions = [1.3]
    # epslions = [5]
    m = len(epslions)
    muld = 1
    portion = [0.03,0.05,0.1,0.2]
    n = len(portion)
    MEGRR_f, MEOLH_f, PCKVUE_f,JVEPM_f, SVE_f = np.zeros((n, m)), np.zeros((n, m)), np.zeros((n, m)), np.zeros(
        (n, m)),np.zeros((n, m))
    MEGRR_v, MEOLH_v, PCKVUE_v,JVEPM_v, SVE_v1, SVE_v2 = np.zeros((n, m)), np.zeros((n, m)), np.zeros((n, m)), np.zeros(
        (n, m)), np.zeros((n, m)),np.zeros((n, m))
    for i in range(len(portion)):
        data = generate_Data(portion[i],"gaussian")

        mse_MEGRR_f, mse_MEGRR_v = pckv_GRR(data, epslions, mul_d=muld)
        mse_PCKVUE_f, mse_PCKVUE_v = pckv_UE(data, epslions)
        mse_SVE_f, mse_SVE_v,mse_SVE_EM_v  = SVEPM(0.0, data, epslions, muld)
        # mse_SVE_EM_v
        for j in range(len(epslions)):
            MEGRR_f[i, j] = round(mse_MEGRR_f[j],10)
            MEGRR_v[i, j] = mse_MEGRR_v[j]

            PCKVUE_f[i, j] = mse_PCKVUE_f[j]
            PCKVUE_v[i, j] = mse_PCKVUE_v[j]

            SVE_f[i, j] = round(mse_SVE_f[j],10)
            # SVE_v1[i, j] = round(mse_SVE_v[j],10)
            SVE_v2[i, j] = round(mse_SVE_EM_v[j],10)

    np.set_printoptions(formatter={'all': lambda x: str(x) + ', '})
    for i in range(len(portion)):
        print("\nMSE of frequency : portion", portion[i])
        print("PCKVGRR = ", MEGRR_f[i, :])
        print("PCKVUE = ", PCKVUE_f[i, :])
        print("Ours = ", SVE_f[i, :])
        print("\nMSE of conditional mean values: portion", portion[i])
        print("PCKVGRR = ", MEGRR_v[i, :])
        print("res_PCKV_UE = ", PCKVUE_v[i, :])
        print("Ours = ", SVE_v2[i, :])


