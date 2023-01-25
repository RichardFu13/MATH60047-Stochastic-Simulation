import numpy as np
import matplotlib.pyplot as plt

### Q1

### DEFINE FUNCTIONS AND PARAMETERS
def sample_exponential(l):
    u = np.random.uniform(0, 1) # generate uniform number
    inverse = lambda x: (-1/l) * np.log(1-x) # define inverse of exponential pdf
    sample = inverse(u) # samples from exponential distribution using inversion method

    return sample


n = 100000 # desired size of sample
nu = 4
l = 1 / nu # optimal lambda to minimise M
M = (nu ** (nu / 2)) * np.exp(1 - nu / 2) / ((2 ** (nu / 2)) * np.math.factorial(int(nu / 2) - 1)) # calculate optimal M using nu and lmbda as above
p = lambda x, nu: x ** (nu / 2 - 1) * np.exp(- x / 2) / (2 ** (nu / 2) * np.math.factorial(int(nu / 2) - 1)) # chi-squared pdf function
q = lambda x, l : l * np.exp(-l * x) # exponential pdf function

### SAMPLING
accepted_sample = np.array([]) # initialize array
for i in range(n):
    q_samp = sample_exponential(l) # sample from the exponential distribution
    acceptance_prob = p(q_samp, nu) / (M * q(q_samp, l)) # calculate acceptance probability
    u = np.random.uniform(0, 1) # sample u
    if u <= acceptance_prob: # acceptance condition
        accepted_sample = np.append(accepted_sample, q_samp) # append to accepted array

### PLOTTING
plt.figure(figsize=(10,6))
xx = np.linspace(0, 20, 1000)
plt.xlim(0, 20)
plt.hist(accepted_sample, bins=100, density=True, rwidth=0.8, color="r", alpha=0.5, label="Sample Histogram")
plt.plot(xx, p(xx, nu), "k-", label="$p_{\\nu}(x)$")
plt.plot(xx, M * q(xx, l), "b-", label="$M_{\lambda^*}q_{\lambda^*}(x)$")
plt.legend(loc="upper right")

### ACCEPTANCE RATE
acceptance_rate = len(accepted_sample) / n # number of accepted samples / total samples considered
theoretical_rate = 1 / M # theoretical acceptance rate
# put these rates and the error in the plot title, to 4 decimal places
plt.title(f"Acceptance rate = {np.around(acceptance_rate, 4)}, Theoretical Acceptance rate = {np.around(theoretical_rate, 4)}, error = {np.around(np.abs(acceptance_rate - theoretical_rate), 4)}")
plt.show()
