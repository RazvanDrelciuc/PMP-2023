import pandas as pd
import pymc3 as pm
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('auto-mpg.csv')
df = data[['horsepower', 'mpg']].dropna()

plt.scatter(df['horsepower'], df['mpg'])
plt.xlabel('Cai putere (CP)')
plt.ylabel('Mile per galon (mpg)')
plt.title('Relația dintre CP și mpg')
plt.show()

with pm.Model() as model:
    alpha = pm.Normal('alpha', mu=0, sd=10)
    beta = pm.Normal('beta', mu=0, sd=10)
    sigma = pm.HalfNormal('sigma', sd=1)
    
    mu = alpha + beta * df['horsepower']
    
    likelihood = pm.Normal('mpg', mu=mu, sd=sigma, observed=df['mpg'])
    
with model:
    trace = pm.sample(1000, tune=1000)

pm.traceplot(trace)
plt.show()

plt.scatter(df['horsepower'], df['mpg'], label='Date observate')
pm.plot_posterior_predictive_glm(trace, samples=100, eval=np.linspace(df['horsepower'].min(), df['horsepower'].max(), 100), color='red', alpha=0.2, label='Distribuția predictivă a posteriori', hdi_prob=0.95)
plt.xlabel('Cai putere (CP)')
plt.ylabel('Mile per galon (mpg)')
plt.title('Dreapta de regresie și distribuția predictivă a posteriori')
plt.legend()
plt.show()

# Concluzii:
# Daca intervalul HDI este destul de ingust, modelul poate fi considerat relativ sigur in predictiile sale.
# Daca intervalul HDI este larg, ar putea indica o incertitudine mai mare în predictiile modelului sau ca
# modelul ar putea beneficia de ajustări suplimentare.
