import matplotlib.pyplot as plt

stride = [1, 2, 4, 8]
gmem_bandwidth = [530.11, 182.489, 91.9925, 46.2869]

plt.plot(stride, gmem_bandwidth, marker='.')
plt.title("Global Memory Bandwidth vs Stride")
plt.xlabel("Stride")
plt.ylabel("Bandwidth")
plt.grid(True)
plt.show()