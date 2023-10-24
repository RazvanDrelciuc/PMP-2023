import numpy as np
from scipy.stats import poisson, norm
from scipy.optimize import minimize_scalar

# Modelul probabilistic:
# 
# a) Numărul de clienți care intră în restaurant într-o oră urmează o distribuție Poisson 
# cu parametrul λ = 20 clienți/oră. Vom nota această variabilă aleatoare ca N (Numărul de clienți).
# 
# b) Timpul de plasare și plată al unei comenzi urmează o distribuție normală cu medie μ = 2 minute 
# și deviație standard σ = 0.5 minute. Vom nota această variabilă aleatoare ca Tp (Timpul de plată).
# 
# c) Timpul necesar pentru a pregăti o comandă la stația de gătit urmează o distribuție exponențială
#  cu o medie α minute. Vom nota această variabilă aleatoare ca Tg (Timpul de gătit).


# Parametrii
lambda_c = 20  # Parametrul Poisson pentru numărul de clienți
mu = 2  # Media timpului de plasare și plată (min)
sigma = 0.5  # Deviația standard a timpului de plasare și plată (min)
target_wait_time = 15  # Timpul țintă de așteptare pentru servirea tuturor clienților (min)
target_probability = 0.95  # Probabilitatea țintă


def ex2(alpha):
    # Calculează E[W] folosind distribuția normală a sumei timpului de așteptare
    mean_wait_time = lambda_c * (mu + alpha)
    
    std_deviation_wait_time = np.sqrt(lambda_c) * alpha
    
    z_score = (target_wait_time - mean_wait_time) / std_deviation_wait_time
    probability = norm.cdf(z_score)
    
    return -probability

# Găsește α maxim
result = minimize_scalar(ex2, bounds=(0, 10))  
alpha_max = result.x

print(f"α maxim: {alpha_max:.2f} minute")

# Timpul mediu de așteptare pentru un client
average_wait_time = (mu + alpha_max) / 2
print(f"Timpul mediu de așteptare pentru un client: {average_wait_time:.2f} minute")