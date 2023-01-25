import numpy as np
import matplotlib.pyplot as plt

### Q1 Part 2
phi = lambda x: 1/np.sqrt(2*np.pi) * np.exp(-(9-x)**2 / 2) # phi(x)
true_value = 1/(2* np.sqrt(np.pi)) * np.exp(-9**2 / 4)

N_array = np.array([10, 100, 1000, 10000, 100000])
RAE_array_mc = np.array([])
for N in N_array:
    p_x = np.random.normal(0, 1, N) # sample from p(x) N times
    phi_mc = 1/N * np.sum(phi(p_x))
    RAE_mc = np.abs(phi_mc - true_value) / np.abs(true_value)
    print(phi_mc, true_value)
    RAE_array_mc = np.append(RAE_array_mc, RAE_mc)

plt.figure(figsize=(20,7))
plt.loglog(N_array, RAE_array_mc, "r-", label="RAE for MC")
plt.legend(loc="upper right")
plt.xlabel("N", fontsize=15)
plt.ylabel("RAE", fontsize=15)


### Q1 Part 3
w_phi = lambda x: 1/np.sqrt(2*np.pi) * np.exp(-(x**2 - 6*x + 45)/2) # w(x)phi(x)
RAE_array_is = np.array([])
for N in N_array:
    q_x = np.random.normal(6, 1, N) # sample from q(x) N times
    phi_is = 1/N * np.sum(w_phi(q_x))
    RAE_is = np.abs(phi_is - true_value) / np.abs(true_value)
    RAE_array_is = np.append(RAE_array_is, RAE_is)

plt.figure(figsize=(20,7))
plt.loglog(N_array, RAE_array_is, "r-", label="RAE for IS")
plt.legend(loc="upper right")
plt.xlabel("N", fontsize=15)
plt.ylabel("RAE", fontsize=15)


### Q1 Part 4
plt.figure(figsize=(20,7))
plt.loglog(N_array, RAE_array_mc, "b-", label="RAE for MC")
plt.loglog(N_array, RAE_array_is, "r-", label="RAE for IS")
plt.legend(loc="upper right")
plt.xlabel("N", fontsize=15)
plt.ylabel("RAE", fontsize=15)
plt.show()