import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt

# Student: Drelciuc Razvan 3e4
# Problema 2


# Generam 100 de timpi medii de asteptare folosind distributia norm
np.random.seed(42)
timp_mediu_asteptare_obs = np.random.normal(loc=15, scale=3, size=100)

# Definim modelul PyMC
model = pm.Model()

with model:
    #  Distributii a priori pentru parametrii a și b
    a = pm.Normal('a', mu=15, sd=5)
    b = pm.Normal('b', mu=3, sd=2)

    # Modelam distributia observatiilor folosind distributia normala cu parametrii a și b
    timp_mediu_asteptare_estimat = pm.Normal('timp_mediu_asteptare_estimat', mu=a, sd=b, observed=timp_mediu_asteptare_obs)

# Estimam distributia a posteriori
with model:
    trace = pm.sample(1000, tune=1000)

# Vizualizare distributia a posteriori pentru parametrul a
pm.plot_posterior(trace['a'], kde_plot=True)
plt.title('Distribuția a posteriori pentru parametrul a')
plt.show()
