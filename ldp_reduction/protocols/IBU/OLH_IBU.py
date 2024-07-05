import numpy as np
import matplotlib.pyplot as plt
import xxhash
from sys import maxsize
# Multi-Freq-LDPy functions for L-SUE protocol (a.k.a. Basic RAPPOR[11])
from multi_freq_ldpy.pure_frequency_oracles.LH import  LH_Aggregator_MI, LH_Aggregator_IBU
from utils import calMAE

def GRR_Client(input_data, k, epsilon):
    """
    Generalized Randomized Response (GRR) protocol, a.k.a., direct encoding [1] or k-RR [2].

    :param input_data: user's true value;
    :param k: attribute's domain size;
    :param epsilon: privacy guarantee;
    :return: sanitized value.
    """

    # Validations
    if input_data < 0 or input_data >= k:
        raise ValueError('input_data (integer) should be in the range [0, k-1].')
    if not isinstance(k, int) or k < 2:
        raise ValueError('k needs an integer value >=2.')
    if epsilon > 0:

        # GRR parameters
        p = np.exp(epsilon) / (np.exp(epsilon) + k - 1)

        # Mapping domain size k to the range [0, ..., k-1]
        domain = np.arange(k)

        # GRR perturbation function
        if np.random.binomial(1, p) == 1:
            return input_data

        else:
            return np.random.choice(domain[domain != input_data])


def LH_Client(input_data, k, epsilon, optimal=True):
    """
    Local Hashing (LH) protocol[1], which is logically equivalent to the random matrix projection technique in [2].

    :param input_data: user's true value;
    :param k: attribute's domain size;
    :param epsilon: privacy guarantee;
    :param optimal: if True, it uses the Optimized LH (OLH) protocol from [1];
    :return: tuple of sanitized value and random seed.
    """

    # Validations
    if input_data < 0 or input_data >= k:
        raise ValueError('input_data (integer) should be in the range [0, k-1].')
    if not isinstance(k, int) or k < 2:
        raise ValueError('k needs an integer value >=2.')
    if epsilon > 0:

        # Binary LH (BLH) parameter
        g = 2

        # Optimal LH (OLH) parameter
        if optimal:
            g = int(round(np.exp(epsilon))) + 1

        # Generate random seed and hash the user's value
        rnd_seed = np.random.randint(0, maxsize, dtype=np.int64)
        hashed_input_data = (xxhash.xxh32(str(input_data), seed=rnd_seed).intdigest() % g)

        # LH perturbation function (i.e., GRR-based)
        sanitized_value = GRR_Client(hashed_input_data, g, epsilon)

        return (sanitized_value, rnd_seed)

    else:
        raise ValueError('epsilon (float) needs a numerical value greater than 0.')
# Parameters for simulation
def IBU_est(l_lh_reports,epsilon,k):

    # Simulation of client-side
    # l_lh_reports = [LH_Client(input_data, k, epsilon_perm) for input_data in data]
    # Simulation of server-side aggregation with Matrix Inversion (MI)
    # l_lh_est_freq_MI = LH_Aggregator_MI(l_lh_reports,k, epsilon_perm)
    # Simulation of server-side aggregation with Iterative Bayesian Updates (IBU)[14]
    lh_est_freq_IBU = LH_Aggregator_IBU(l_lh_reports, k, epsilon)
    # Real frequency
    # real_freq = np.unique(data, return_counts=True)[-1] / n
    return lh_est_freq_IBU

# IBU_est(data,epsilon_perm,12)

# Visualizing results
# x = np.arange(k)  # the label locations
# barwidth = 0.3 # the width of the bars
#
# plt.bar(x - barwidth, real_freq, label='Real Freq', width=barwidth)
# plt.bar(x , l_lh_est_freq_MI, label='MI Est Freq: OLH', width=barwidth)
# plt.bar(x + barwidth, l_lh_est_freq_IBU, label='IBU Est Freq: OLH', width=barwidth)
# plt.ylabel('Normalized Frequency')
# plt.xlabel('Domain values')
# plt.legend(ncol=3, loc='upper right', bbox_to_anchor=(1., 1.1))
# plt.show()