import numpy as np
import matplotlib.pyplot as plt

### Q2

def sample_discrete(w):
    cw = np.cumsum(w) # cdf of weights
    u = np.random.uniform(0, 1) # random uniform
    for k in range(len(cw)): # samples the index from the discrete distribution
        if u <= cw[k]:
            return k


def sample_exponential(l):
    u = np.random.uniform(0, 1) # generate uniform number
    inverse = lambda x: (-1/l) * np.log(1-x) # define inverse of exponential pdf
    sample = inverse(u) # samples from exponential distribution using inversion method

    return sample


def sample_chi(p, q, M, nu, l):
    while True:
        q_samp = sample_exponential(l) # sample from the exponential distribution
        acceptance_prob = p(q_samp, nu) / (M * q(q_samp, l)) # calculate acceptance probability
        u = np.random.uniform(0, 1) # sample u
        if u <= acceptance_prob: # acceptance condition
            return q_samp

n = 100000
w_array = np.array([0.2, 0.5, 0.3])
nu_array = np.array([4, 16, 40]) # initialize array of nu values
l_array = 1 / nu_array # initialize array of optimal lambda values
M_array = np.array([(nu ** (nu / 2)) * np.exp(1 - nu / 2) / ((2 ** (nu / 2)) * np.math.factorial(int(nu / 2) - 1)) for nu in nu_array]) # calculate array of optimal M's

p = lambda x, nu: x ** (nu / 2 - 1) * np.exp(- x / 2) / (2 ** (nu / 2) * np.math.factorial(int(nu / 2) - 1)) # chi-squared pdf function
q = lambda x, l : l * np.exp(-l * x) # exponential pdf function
mixture_density = lambda x, w_array, nu_array: sum([w_array[i]*p(x, nu_array[i]) for i in range(len(w_array))])

mixture_sample = np.array([])
for i in range(n):
    idx = sample_discrete(w_array)
    chi = sample_chi(p, q, M_array[idx], nu_array[idx], l_array[idx])
    mixture_sample = np.append(mixture_sample, chi)

### PLOTTING
plt.figure(figsize=(10,6))
xx = np.linspace(0, 80, 1000)
plt.xlim(0, 80)
plt.hist(mixture_sample, bins=100, density=True, rwidth=0.8, color="r", alpha=0.5, label="Sample Histogram")
plt.plot(xx, mixture_density(xx, w_array, nu_array), color="k", linewidth=2, label="$p(x)$")
plt.legend(loc="upper right")
plt.show()
