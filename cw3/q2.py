import numpy as np
import matplotlib.pyplot as plt

### Part a
x0 = 1
a = 0.9
sigma_x = 0.01
sigma_y = 0.1
T = 100

x_vals = np.array([x0])
y_vals = np.array([])
x = x0
for t in range(T):
    x = np.random.normal(a*x, sigma_x)
    y = np.random.normal(x, sigma_y)
    x_vals = np.append(x_vals, x)
    y_vals = np.append(y_vals, y)

plt.figure(figsize=(10,7))
plt.plot(x_vals, "k-", label="x")
plt.plot(np.arange(1, T+1), y_vals, "r-", label="y")
plt.xlabel("time")
plt.legend()
plt.show()

### Part b
T = 500

### simulate data for x decaying
x0 = 10
a = 0.99
sigma_x = 0.01
x_decay = np.array([[x0]])
y_decay = np.array([])
x = x0
for t in range(T):
    x = np.abs(np.random.normal(a*x, sigma_x))
    y = np.random.normal(0, x)
    x_decay = np.append(x_decay, x)
    y_decay = np.append(y_decay, y)

### simulate data for x growing
x0 = 0.1
a = 1.01
sigma_x = 0.01
x_grow = np.array([[x0]])
y_grow = np.array([])
x = x0
for t in range(T):
    x = np.abs(np.random.normal(a*x, sigma_x))
    y = np.random.normal(0, x)
    x_grow = np.append(x_grow, x)
    y_grow = np.append(y_grow, y)

### plot x decaying and growing and corresponding ys
fig, axs = plt.subplots(2, 2, figsize=(10,7))
axs[0, 0].plot(x_decay, "k-", label="x")
axs[1, 0].plot(np.arange(1, T+1), y_decay, "r-", label="y")
axs[0, 1].plot(x_grow, "k-", label="x")
axs[1, 1].plot(np.arange(1, T+1), y_grow, "r-", label="y")
for i in range(2):
    for j in range(2):
        axs[i, j].set_xlabel("time")
        axs[i, j].legend()

### simulate other data
x0 = 10
a = 1
sigma_x = 1
x_vals = np.array([x0])
y_vals = np.array([])
x = x0
for t in range(T):
    x = np.abs(np.random.normal(a*x, sigma_x))
    y = np.random.normal(0, x)
    x_vals = np.append(x_vals, x)
    y_vals = np.append(y_vals, y)

### plot simulation data
fig, axs = plt.subplots(2, 1, figsize=(10,7))
axs[0].plot(x_vals, "k-", label="x")
axs[1].plot(np.arange(1, T+1), y_vals, "r-", label="y")
axs[0].set_xlabel("time")
axs[1].set_xlabel("time")
axs[0].legend()
axs[1].legend()
plt.show()
