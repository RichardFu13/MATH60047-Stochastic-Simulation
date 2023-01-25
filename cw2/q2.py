import numpy as np
import matplotlib.pyplot as plt

### Q2 Part 2
N = 1000000
sigma_y = 1
mu_x = 0
sigma_x = 10
s_arr = np.array([-1, 2, 5])
y_arr = np.array([4.44, 2.51, 0.73])
x_true = 4

def log_acceptance_prob(x, x_prime):
    first_terms = ((x-mu_x)**2 - (x_prime-mu_x)**2) / (2*sigma_x**2)
    sum_term = np.sum([((y_arr[i] - np.abs(x - s_arr[i]))**2 - (y_arr[i] - np.abs(x_prime - s_arr[i]))**2) / (2*sigma_y**2) for i in range(3)])
    return first_terms + sum_term

def sample_MH(x0, sigma_q, N):
    x = x0
    accepted_samples = np.array([])
    for n in range(N):
        x_prime = np.random.normal(x, sigma_q)
        u = np.random.uniform(0, 1)
        prob = log_acceptance_prob(x, x_prime)
        if np.log(u) <= prob:
            accepted_samples = np.append(accepted_samples, x_prime)
            x = x_prime
    return accepted_samples

sample_a = sample_MH(10, 0.1, N)
burnin_a = 1000
burnin_b = 40000
sample_b = sample_MH(10, 0.01, N)
plt.figure(figsize=(20,7))
plt.title("Histogram for $\sigma_q = 0.1, \sigma_y = 1$", fontsize=15)
plt.axvline(x_true, color="k", label="true value", linewidth=2)
plt.hist(sample_a[burnin_a:], bins=50, density=True, label="posterior", alpha=0.5, color=[0.8, 0, 0])
plt.legend(loc="upper right")

plt.figure(figsize=(20,7))
plt.title("Histogram for $\sigma_q = 0.01, \sigma_y = 1$", fontsize=15)
plt.axvline(x_true, color="k", label="true value", linewidth=2)
plt.hist(sample_b[burnin_b:], bins=50, density=True, label="posterior", alpha=0.5, color=[0.8, 0, 0])
plt.legend(loc="upper right")


### Q2 Part 3
sigma_y = 0.1
y_arr = np.array([5.01, 1.97, 1.02])
sample_c = sample_MH(10, 0.1, N)
burnin_c = 100
plt.figure(figsize=(20,7))
plt.title("Histogram for $\sigma_q = 0.1, \sigma_y = 0.1$", fontsize=15)
plt.axvline(x_true, color="k", label="true value", linewidth=2)
plt.hist(sample_c[burnin_c:], bins=50, density=True, label="posterior", alpha=0.5, color=[0.8, 0, 0])
plt.legend(loc="upper right")

### Plotting the samples
plt.figure(figsize=(20,7))
plt.title("Sample plots", fontsize=15)
plt.plot(sample_a, "r-", label = "$\sigma_q = 0.1, \sigma_y = 1$")
plt.plot(sample_b, "b-", label = "$\sigma_q = 0.01, \sigma_y = 1$")
plt.plot(sample_c, "g-", label = "$\sigma_q = 0.1, \sigma_y = 0.1$")
plt.legend(loc="upper right")
plt.show()