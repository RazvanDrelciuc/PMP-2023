import pymc3 as pm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('trafic.csv')

nr_masini = data['nr. masini']

intervale_creste = [(7, 16), (8, 19)]

rezultate = []

for interval in intervale_creste:
    start, end = interval
    data_subset = nr_masini[(data.index >= start) & (data.index <= end)]
    
    with pm.Model() as model:
        # Definim distributia prior pentru parametrul λ cu o distributie Exponential
        lmbda = pm.Exponential('lmbda', lam=1)
        
        # Modelul observat
        traffic = pm.Poisson('traffic', mu=lmbda, observed=data_subset)
        
        # Realizam inferenta Bayesiana
        trace = pm.sample(1000, tune=1000, cores=1)
        
    # Extragem distributia posteriora pentru λ
    posterior_samples = trace['lmbda']
    
    # Calculam intervalul de credinta 95%
    interval_credinta = np.percentile(posterior_samples, [2.5, 97.5])
    
    # Calculam valoarea medie a lui λ
    medie_lambda = np.mean(posterior_samples)
    
    rezultate.append({
        'Interval': interval,
        'Capete Probabile': (interval_credinta[0], interval_credinta[1]),
        'Valoare Medie Lambda': medie_lambda
    })

for i, rezultat in enumerate(rezultate):
    print(f"Interval {i + 1}:")
    print(f"Capete Probabile: {rezultat['Capete Probabile']}")
    print(f"Valoare Medie Lambda: {rezultat['Valoare Medie Lambda']}\n")


#Bonus Cerinta 1
alpha = 3
timpi_asteptare_medii = np.random.gamma(alpha, scale=1, size=100)

plt.hist(timpi_asteptare_medii, bins=20, density=True, alpha=0.6, color='b', label='Esantion de timpi de asteptare medii')
plt.xlabel('Timp de Asteptare Mediu')
plt.ylabel('Probabilitate')
plt.legend()
plt.show()

#Bonus  Cerinta 2
with pm.Model() as alpha_model:
    alpha_param = pm.Gamma('alpha_param', alpha=2, beta=1) 
    obs_alpha = pm.Gamma('obs_alpha', alpha=alpha_param, beta=1, observed=timpi_asteptare_medii) 

    trace_alpha = pm.sample(1000, tune=1000, cores=1)  

pm.summary(trace_alpha)
pm.plot_posterior(trace_alpha)
plt.xlabel('α')
plt.ylabel('Densitatea de probabilitate')
plt.show()
