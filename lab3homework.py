# Importăm bibliotecile necesare
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# Definim modelul probabilist
model = BayesianNetwork([('Cutremur', 'Incendiu'), ('Cutremur', 'Alarma'), ('Incendiu', 'Alarma')])

# Definim distribuțiile de probabilitate condiționată (CPDs)
cpd_cutremur = TabularCPD(variable='Cutremur', variable_card=2, values=[[0.9995], [0.0005]])
cpd_incendiu = TabularCPD(variable='Incendiu', variable_card=2, values=[[0.99], [0.01]])
cpd_alarma = TabularCPD(variable='Alarma', variable_card=2, 
                        values=[[0.9999, 0.02, 0.01, 0.98],
                                [0.0001, 0.98, 0.99, 0.02]],
                        evidence=['Cutremur', 'Incendiu'], evidence_card=[2, 2])

# Adăugăm CPDs la model
model.add_cpds(cpd_cutremur, cpd_incendiu, cpd_alarma)

# Verificăm consistența modelului
assert model.check_model()

# Efectuăm inferența folosind eliminarea variabilei
inference = VariableElimination(model)

# Calculăm probabilitatea că a avut loc un cutremur, dat fiind că alarma de incendiu s-a declanșat
result_cutremur = inference.query(variables=['Cutremur'], evidence={'Alarma': 1})
print("Probabilitatea că a avut loc un cutremur dat fiind că alarma de incendiu s-a declanșat:")
print(result_cutremur)

# Calculăm probabilitatea că a avut loc un incendiu fără ca alarma de incendiu să se activeze
result_incendiu = inference.query(variables=['Incendiu'], evidence={'Alarma': 0})
print("\nProbabilitatea că a avut loc un incendiu fără ca alarma de incendiu să se activeze:")
print(result_incendiu)


# BONUS
#       P(C) reprezintă probabilitatea ca un cutremur să aibă loc (cutremurul să fie adevărat).
#       P(I) reprezintă probabilitatea ca un incendiu să aibă loc (incendiul să fie adevărat
#       P(A) reprezintă probabilitatea ca alarma să se declanșeze (alarma să fie adevărată).
# 1) Probabilitate ca a avut loc un cutremur dat fiind ca alrma de incediu s-a activat.
#       P(C|A) = ( (P(A|C)*P(C)) / P(A) )
#       P(A) = P(A|C)*P(C) + P(A|I)*P(I)
#       P(C) = 0.05/100 (0.0005)
#       P(A|C) =2%
#       P(A|I) = 95%
#       P(I) = 1%