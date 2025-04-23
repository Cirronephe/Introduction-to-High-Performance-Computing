import matplotlib.pyplot as plt

bitwidths = [2, 4, 8]
strides = [1, 2, 4, 8, 16, 32]
bandwidth = {
    2: [4260.62, 4303.68, 2157.82, 829.42, 426.825, 216.257],
    4: [8594.7, 4313.75, 2026.71, 1008.92, 508.751, 251.018],
    8: [8654.27, 4339.37, 2173.54, 1087.65, 544.07, 544.07]
}

for b in bitwidths:
    plt.plot(strides, bandwidth[b], marker='.', label=f'bitwidth={b}')
    
plt.title("Shared Memory Bandwidth vs Stride")
plt.xlabel("Stride")
plt.ylabel("Bandwidth")
plt.legend()
plt.grid(True)
plt.show()