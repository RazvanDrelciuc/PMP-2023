import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

alpha_server1 = 4
lambda_server1 = 3

alpha_server2 = 4
lambda_server2 = 2

alpha_server3 = 5
lambda_server3 = 2

alpha_server4 = 5
lambda_server4 = 3

lambda_latency = 4

prob_server1 = 0.25
prob_server2 = 0.25
prob_server3 = 0.30

dist_server1 = stats.gamma(alpha_server1, scale=1/lambda_server1)
dist_server2 = stats.gamma(alpha_server2, scale=1/lambda_server2)
dist_server3 = stats.gamma(alpha_server3, scale=1/lambda_server3)
dist_server4 = stats.gamma(alpha_server4, scale=1/lambda_server4)

dist_latency = stats.expon(scale=1/lambda_latency)

def compute_X_distribution():
    X_values = []
    X_probs = []

    for latency_sample in range(1, 10001):
        latency = dist_latency.rvs()
        server = np.random.choice([1, 2, 3, 4], p=[prob_server1, prob_server2, prob_server3, 0.2])
        if server == 1:
            X = dist_server1.rvs() + latency
        elif server == 2:
            X = dist_server2.rvs() + latency
        elif server == 3:
            X = dist_server3.rvs() + latency
        else:
            X = dist_server4.rvs() + latency

        X_values.append(X)

    return X_values

X_values = compute_X_distribution()

prob_X_great_3 = np.mean(np.array(X_values) > 3)

print(f"Probabilitatea ca X > 3 milisecunde: {prob_X_great_3}")

plt.hist(X_values, bins=50, density=True, alpha=0.7, color='b', label='Distribuția X')
plt.xlabel('Timp (milisecunde)')
plt.ylabel('Densitate')
plt.title('Densitatea distribuției timpului de servire X')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()