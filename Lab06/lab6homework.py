import pymc3 as pm
import numpy as np
import arviz as az

# Datele observate pentru Y
observed_Y = [0, 5, 10]

# Valorile posibile pentru θ
theta_values = [0.2, 0.5]

# Prior pentru n, distribuția Poisson(10)
prior_lambda = 10

num_samples = 1000

with pm.Model() as model:
    n = pm.Poisson('n', mu=prior_lambda)
    for Y in observed_Y:
        for theta in theta_values:
            Y_obs = pm.Binomial('Y_obs', n=n, p=theta, observed=Y)
    
    trace = pm.sample(num_samples, chains=2, tune=500)

az.plot_posterior(trace, var_names='n')
