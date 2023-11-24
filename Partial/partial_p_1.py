from pgmpy.models import BayesianModel
from pgmpy.estimators import ParameterEstimator
from pgmpy.inference import VariableElimination
import random

# Student: Drelciuc Razvan 3e4
# Problema 1

# Definirea modelului
model = BayesianModel([('coin_toss', 'j0_toss'), 
                       ('coin_toss', 'j1_toss'),
                       ('j0_toss', 'j0_score'),
                       ('j1_toss', 'j1_score')])

# Generarea de date pentru simulare
data = []
# Date de simulare
for _ in range(10000):
    coin_toss = random.choice(['H', 'T'])
    j0_toss = random.choice(['H', 'T'])
    j1_toss = random.choice(['H', 'T'])
    
    if coin_toss == 'H':
        j0_score = random.choice(['0', '1'])
        j1_score = random.choice(['0', '1'] * (int(j0_score) + 1))
    else:
        j1_score = random.choice(['0', '1'])
        j0_score = random.choice(['0', '1'] * (int(j1_score) + 1))
    
    data.append({'coin_toss': coin_toss,
                 'j0_toss': j0_toss,
                 'j1_toss': j1_toss,
                 'j0_score': j0_score,
                 'j1_score': j1_score})

# Antrenarea modelului cu datele simulate
model.fit(data, estimator=ParameterEstimator)

# Calcularea probabilitatilor marginale pentru castigarea fiecarui player
inference = VariableElimination(model)
probabilities = inference.query(variables=['j0_score', 'j1_score'], evidence={'coin_toss': 'H'})

print(probabilities)
# Calcularea prob conditionate
prob_start_with_player_0 = inference.query(variables=['coin_toss'], evidence={'player_1_score': '1'})

print(prob_start_with_player_0['coin_toss'])
