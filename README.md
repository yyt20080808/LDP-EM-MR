# Revisiting EM-based Estimation for Locally Differentially Private Protocols
Equip the existing SOTA LDP protocols with Expectation-maximization algorithm and reduction framework for estimating.

## Dependencies
To run the experiments in this repo, you may need to use python3.8 or later version, you need to install package `numpy`, `xxhash`. 

## File structure
- `./ldp_reduction`: the source directory of all the mechanisms, datasets and models we have experimented with.
  - `./ldp_reduction/MR` implements our main method, the mixture reduction(MR) for different SOTA protocols.
  - `./ldp_reduction/categoricalProtocols` SOTA LDP categorical protocols, we implement its EM-based MLE and unbiased estimation (GRR & OLH).
  - `./ldp_reduction/numericalProtocols`  SOTA LDP numerical protocols, we implement its EM-based MLE and unbiased estimation (PM).
  - `./ldp_reduction/keyvalueProtocols`  SOTA LDP KV protocols, we implement it (PCKV-UE & PCKV-GRR).
  - `./ldp_reduction/SP2023convo` follows the implementation on frequency estimation from SP2023.
  - `./ldp_reduction/opendata` contains the real datasets(SFC & Income), and the scripts t.
- `./simulation` is the directory of Python scripts to run experiments.
  - `./simulation/fre_est_grr.py` runs the baseline GRR with our method for frequency estimation.
  - `./simulation/fre_est_olh.py` runs the baseline OLH with our method for frequency estimation.
  - `./simulation/mean_est` runs baselines (PM & SR) with our method for mean estimation.
  - `./simulation/distribution_est` runs baseline with our method for distribution estimation.
  - `./simulation/output` stores the results.
 
## Running
Inside directory, you can run experiments like:
```
python fre_est_grr.py
```

## Basic Usage
```python
from ldp_reduction.protocols.categoricalProtocols.GRR import GRR, GRR_revision_no, norm_sub, GRR_revision_basecut, IIW
from ldp_reduction.protocols.categoricalProtocols.GRR_EM import EM_GR
from ldp_reduction.protocols.MR.GRR_EM_reduction import EM_GRR_reduct_merge

# Using General Random Response (GRR) for frequency estimation

import matplotlib.pyplot as plt
dataset = "SFC" # Dataset with 60 possible items 
epsilon = 1.5 # Privacy budget of 1.5

acc1, acc11, acc2, acc21, acc3, acc31, acc4, acc41, acc5, acc51, acc6, acc61 = [[] for _ in range(12)]
exper_time = 10
for i in range(exper_time):
    noised_data, number, ori = generate_noised_data(epsilon,dataset) # simulate noised reports 

    weights4 = GRR_revision_no(noised_data, number, epsilon) # unbiased estimation
    acc4.append(calMSE(weights4, ori, length))
    acc41.append(calMSE(weights4, ori, topK=True))
    sum_all4 += np.array(weights4[0:length]) * len(noised_data)

    weights3 = GRR_revision_basecut(noised_data, number, epsilon) # Basecut
    acc3.append(calMSE(weights3, ori))
    acc31.append(calMSE(weights3, ori, topK=True))

    weights5 = norm_sub(np.array(weights4) * len(noised_data), len(noised_data)) # Normsub
    acc5.append(calMSE(weights5 / len(noised_data), ori))
    acc51.append(calMSE(weights5 / len(noised_data), ori, topK=True))
    sum_all3 += np.array(weights5[0:length])
    
    weights = EM_GR(ori, noised_data, number, math.e ** epsilon)     # EM
    acc1.append(calMSE(weights, ori))
    acc11.append(calMSE(weights, ori, topK=True))

    weights2 = EM_GRR_reduct_merge( noised_data, number, math.e ** epsilon)  # Ours, MR, merging strategy
    acc2.append(calMSE(weights2, ori))
    acc21.append(calMSE(weights2, ori, topK=True))

    weights6 = IIW(noised_data, number, epsilon)   # Smoothing based 
    acc6.append(calMSE(weights6, ori ))
    acc61.append(calMSE(weights6, ori,topK=True))

  print("MSE: \t full domain ")
  print("GRR-Nonprocess:", sum(acc4) / exper_time)
  print("GRR-BaseCut:", sum(acc3) / exper_time)
  print("GRR-NormSub:", sum(acc5) / exper_time)
  print("GRR-IIW:", sum(acc6) / exper_time)
  print("GRR-EM:", sum(acc1) / exper_time)
  print("Ours,GRR-MR:", sum(acc2) / exper_time)

```

## Additional comparison

- `./scripts/OLHEMIBU.py`: is the script comparing the our implemeation of OLH-EM  with [1] (require the package 'multi_freq_ldpy') 

[1] Arcolezi, H.H., Cerna, S., Palamidessi, C. "On the Utility Gain of Iterative Bayesian Update for Locally Differentially Private Mechanisms". In: DBSec 2023. 

