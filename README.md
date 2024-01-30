# Revisiting EM-based Estimation for Locally Differentially Private Protocols
Equip the existing SOTA LDP protocols with Expectation-maximization algorithm and reduction framework for estimating.

## Dependencies
To run the experiments in this repo, you may need to use python3.8 or later version, you need to install package `numpy`, `xxhash`. 

## File structure
- `./src`: the source directory of all the mechanisms, datasets and models we have experimented with.
  - `./src/MR` implements our main method, the mixture reduction(MR) for different SOTA protocols.
  - `./src/categoricalProtocols` SOTA LDP categorical protocols, we implement its EM-based MLE and unbiased estimation (GRR & OLH).
  - `./src/numericalProtocols`  SOTA LDP numerical protocols, we implement its EM-based MLE and unbiased estimation (PM).
  - `./src/SP2023convo` follows the implementation on frequency estimation from SP2023.
  - `./src/opendata` contains all the real datasets(SFC & Income) .
- `./scripts` is the directory of Python scripts to run experiments.
  - `./scripts/frequencyCompare` runs the baseline with our method for frequency estimation.
  - `./scripts/meanCompare` runs baseline with our method for mean estimation.
  - `./scripts/distributionCompare` runs baseline with our method for distribution estimation.
  - `./scripts/output` stores the results.
 
## Running
Inside directory, you can run experiments like:
```
python3 compareIndifferentDataset_PM.py



