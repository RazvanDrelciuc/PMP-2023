import pymc3 as pm
import pandas as pd

# problema 1
# a)
# incarcam setul de date titanic
df = pd.read_csv('Titanic.csv')
# gestionam val lipsa
df['Age'].fillna(df['Age'].mean(), inplace=True)
df['Pclass'] = df['Pclass'].astype('category')
df['Survived'] = df['Survived'].astype('int')

# b)
with pm.Model() as model:
    # Parametrii modelului
    alpha = pm.Normal('alpha', mu=0, sd=10)
    beta_age = pm.Normal('beta_age', mu=0, sd=10)
    beta_class = pm.Normal('beta_class', mu=0, sd=10)

    # Logitul probabilitatii
    logits = alpha + beta_age * df['Age'] + beta_class * df['Pclass'].cat.codes

    observed = pm.Bernoulli('observed', pm.math.sigmoid(logits), observed=df['Survived'])

    # Efectuarea interfatei
    trace = pm.sample(1000)

# c)
coeficienti_medii = pm.summary(trace)['mean']
influenta_age = abs(coeficienti_medii['beta_age'])
influenta_class = abs(coeficienti_medii['beta_class'])

print(f"influenta 'Age': {influenta_age}")
print(f"influenta 'Pclass': {influenta_class}")
if influenta_age > influenta_class:
    print("Variabila 'Age' are influenta mai mare.")
else:
    print("Variabila 'Pclass' are influenta mai mare.")

# d)
with model:
    # Calculul probabilitatii de supravietuire
    p_survival_30_class2 = pm.math.sigmoid(alpha + beta_age * 30 + beta_class * 1) 
    hdi = pm.hdi(p_survival_30_class2, hdi_prob=0.90)
    print(f"Intervalul HDI de 90% pentru probabilitatea de supravietuire a unui pasager de 30 de ani din clasa a 2-a: {hdi}")
