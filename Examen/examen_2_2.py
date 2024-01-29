import numpy as np
from scipy.stats import geom

# problema 2
# a)
def monte_carlo_approximation(iterations=10000, p_x=0.3, p_y=0.5):
    # generam x y samples cu distributia geometrica
    x_samples = geom.rvs(p_x, size=iterations)
    y_samples = geom.rvs(p_y, size=iterations)
    # comparatia dintre x si y
    comparison = x_samples > (y_samples ** 2)
    # calculul probabilitatii aproximative 
    probability = np.mean(comparison)
    return probability

approximation = monte_carlo_approximation()
print(f"Probabilitatea aproximata P(X > (Y^2)) este: {approximation}")

# b)
k = 30
approximations = [monte_carlo_approximation() for _ in range(30)]
mean_approximation = np.mean(approximations)
std_deviation = np.std(approximations)

print(f"Media pentru {k} aproximari este: {mean_approximation}")
print(f"Deviatia standard pentru {k} aproximari este: {std_deviation}")

# Probabilitatea aproximata P(X > (Y^2)) este: 0.4201
# Media pentru 30 aproximari este: 0.41718999999999995
# Deviatia standard pentru 30 aproximari este: 0.005634024612418135
