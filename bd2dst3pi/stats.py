import numpy as np

def binomial_err(N, k):
    return (1/N)*np.sqrt(k*(1 - k/N))
