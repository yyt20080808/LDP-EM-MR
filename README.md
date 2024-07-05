# Revisiting EM-based Estimation for Locally Differentially Private Protocols
Equip the existing SOTA LDP protocols with Expectation-maximization algorithm and reduction framework for estimating.

## Dependencies
To run the experiments in this repo, you may need to use python3.8 or later version, you need to install package `numpy`, `xxhash`,`matplotlib`. 

## Outline
This includes the implementations of Mixture Reduction (MR) framework for estimating LDP noised reports generate from SOTA LDP protocols.
- ` [GRR & OLH](https://www.usenix.org/system/files/conference/usenixsecurity17/sec17-wang-tianhao.pdf):  Existing FOs for frequency estimation, \grr performs better than \olh when $K<3e^{\varepsilon}+2$.
- ` [PM](https://arxiv.org/abs/1907.00782) & [SR](): Existing numerical protocols for mean estimation, \sr performs better than \pmm when $\varepsilon$ is small.
- ` [SW](https://dl.acm.org/doi/abs/10.1145/3318464.3389700):  Existing numerical protocols for distribution estimation, equiped with EM and smoothing technique. 
- ` [PCKV](https://www.usenix.org/system/files/sec20-gu.pdf): Existing key-value protocols, for frequency estimation on key, and conditional mean estimation on value.

Some FOs code is based on the implementation by [Wang](https://github.com/vvv214/LDP_Protocols) and [Maddock](https://github.com/Samuel-Maddock/pure-LDP/blob/master/README.md). And the estimating methods includes unbiased estimation, post-processing method([Basecut, Normsub](https://github.com/vvv214/LDP_Protocols/tree/master/post-process),[IIW](https://github.com/SEUNICK/LDP)), EM-based MLE and our MR.

## File structure
- `./ldp_reduction`: the source directory of all the mechanisms, datasets and models we have experimented with.
  - `./ldp_reduction/MR` implements our main method, the mixture reduction(MR) framework for different SOTA protocols.
  - `./ldp_reduction/categoricalProtocols` SOTA LDP categorical protocols, we implement its EM-based MLE and unbiased estimation (GRR & OLH).
  - `./ldp_reduction/numericalProtocols`  SOTA LDP numerical protocols, we implement its EM-based MLE and unbiased estimation (PM).
  - `./ldp_reduction/keyvalueProtocols`  SOTA LDP KV protocols, we implement PCKV-UE & PCKV-GRR.
  - `./ldp_reduction/SP2023convo` follows the implementation on frequency estimation from SP'2023.
  - `./ldp_reduction/opendata` contains the real datasets(SFC & Income).
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
    acc4.append(calMSE(weights4, ori)) # calculate mean squared error

    weights3 = GRR_revision_basecut(noised_data, number, epsilon) # Basecut
    acc3.append(calMSE(weights3, ori))

    weights5 = norm_sub(np.array(weights4) * len(noised_data), len(noised_data)) # Normsub
    acc5.append(calMSE(weights5 / len(noised_data), ori))
    
    weights = EM_GR(ori, noised_data, number, math.e ** epsilon)   # EM
    acc1.append(calMSE(weights, ori))

    weights2 = EM_GRR_reduct_merge( noised_data, number, math.e ** epsilon)  # Ours, MR, merging strategy
    acc2.append(calMSE(weights2, ori))

    weights6 = IIW(noised_data, number, epsilon)   # Smoothing based 
    acc6.append(calMSE(weights6, ori ))

print("MSE: \t full domain ")
print("GRR-Nonprocess:", sum(acc4) / exper_time)
print("GRR-BaseCut:", sum(acc3) / exper_time)
print("GRR-NormSub:", sum(acc5) / exper_time)
print("GRR-IIW:", sum(acc6) / exper_time)
print("GRR-EM:", sum(acc1) / exper_time)
print("Ours,GRR-MR:", sum(acc2) / exper_time)

```
And the output is
```
MSE: 	 full domain 
GRR-Nonprocess: 5.916919126602377e-05
GRR-BaseCut: 3.86858407911396e-05
GRR-NormSub: 2.4756176741259298e-05
GRR-IIW: 5.4360581915246e-05
GRR-EM: 2.991674829830207e-05
Ours,GRR-MR: 2.303019160242734e-05
```

<!--## Additional comparison

- `./scripts/OLHEMIBU.py`: is the script comparing the our implemeation of OLH-EM  with [1] (require the package 'multi_freq_ldpy') 

[1] Arcolezi, H.H., Cerna, S., Palamidessi, C. "On the Utility Gain of Iterative Bayesian Update for Locally Differentially Private Mechanisms". In: DBSec 2023. -->

