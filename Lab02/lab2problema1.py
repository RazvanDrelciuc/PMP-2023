import numpy as np
import matplotlib.pyplot as plt

lambda1 = 4  
lambda2 = 6  

prob_primul_mecanic = 0.4

nr_val = 10000

timp_servire = []
for _ in range(nr_val):
    if np.random.rand() < prob_primul_mecanic:
        timp_servire.append(np.random.exponential(1 / lambda1))
    else:
        timp_servire.append(np.random.exponential(1 / lambda2))

media_timp_servire = np.mean(timp_servire)
deviatia_standard = np.std(timp_servire)

print("Media timpului de servire (X):", media_timp_servire)
print("Deviația standard a timpului de servire (X):", deviatia_standard)

plt.hist(timp_servire, bins=50, density=True, alpha=0.6, color='g')
plt.xlabel("Timp de servire (X)")
plt.ylabel("Densitate")
plt.title("Densitatea distribuției lui X")
plt.show()