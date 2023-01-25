import numpy as np
import matplotlib.pyplot as plt

N = 10000
x0 = np.array([0,0])
w = np.array([0.2993, 0.7007])
A1 = np.array([[0.4, -0.3733], [0.06, 0.6]])
A2 = np.array([[-0.8, -0.1867], [0.1371, 0.8]])
b1 = np.array([0.3533, 0.0])
b2 = np.array([1.1, 0.1])


def f1(x):
    return A1 @ x + b1


def f2(x):
    return A2 @ x + b2


def sample_discrete(w):
    cw = np.cumsum(w) # cdf of weights
    u = np.random.uniform(0, 1) # random uniform
    for k in range(len(cw)): # samples the index from the discrete distribution
        if u <= cw[k]:
            return k


x_array = np.array([x0])
x = x0
for n in range(N):
    i_n = sample_discrete(w)
    if i_n == 0:
        x = f1(x)
    else:
        x = f2(x)
    x_array = np.vstack((x_array, x))

plt.figure(figsize=(10,7))
plt.scatter(x_array[20:, 0], x_array[20:, 1], s=0.1, color=[0.8, 0, 0])
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().set_xticks([])
plt.gca().set_yticks([])
plt.gca().set_xlim(0, 1.05)
plt.gca().set_ylim(0, 1)
plt.show()
