# Revisiting EM-based Estimation for Locally Differentially Private Protocols
Equip the existing SOTA LDP protocols with Expectation-maximization algorithm and reduction framework for estimating.

## Dependencies
To run the experiments in this repo, you may need to use python3.8 or later version, you need to install package `numpy`, `xxhash`. 

## File structure
- `./src`: the source directory of all the mechanisms, datasets and models we have experimented with.
  - `./src/MR` implements our main method, the mixture reduction(MR).
  - `./src/categoricalProtocols` implements our main result, the Blink framework.
  - `./src/numericalProtocols` implements vanilla randomized response as a baseline.
  - `./src/SP2023convo` follows the implementation on frequency estimation from SP2023.
  - `./src/opendata` contains all the code to download, pre-process and load graph datasets including Cora, CiteCeer and LastFM.
- `./scripts` is the directory of Python scripts to run experiments.
  - `./scripts/frequencyCompare` runs the Blink framework with specified settings.
  - `./scripts/meanCompare` runs baseline methods with specified settings.
  - `./scripts/distributionCompare` stores all the log files when running the scripts above.
  - `./scripts/output` stores the results (hyperparameter choices and final accuracy).
 
## Running
Inside directory, you can run experiments with `python3 run_blink.py {variant name} {dataset} {model_name} --eps {epsilon_list}`, like:
```
python3 run_blink.py hybrid cora gcn --eps 1

## Citation
Please cite our paper as follows:

