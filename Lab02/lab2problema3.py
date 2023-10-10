import numpy as np
import matplotlib.pyplot as plt


def aruncare_moneda(p):
    return 's' if np.random.rand() < p else 'b'

def simulare_experiment():
    rezultat = ''
    for _ in range(10):
        moneda1 = aruncare_moneda(0.5)  
        moneda2 = aruncare_moneda(0.3)  
        rezultat += moneda1 + moneda2
    return rezultat

rezultate_experimente = [simulare_experiment() for _ in range(100)]

distributie_ss = [rezultat.count('ss') for rezultat in rezultate_experimente]
distributie_sb = [rezultat.count('sb') for rezultat in rezultate_experimente]
distributie_bs = [rezultat.count('bs') for rezultat in rezultate_experimente]
distributie_bb = [rezultat.count('bb') for rezultat in rezultate_experimente]

plt.figure(figsize=(12, 8))
plt.subplot(221)
plt.hist(distributie_ss, bins=range(0, 11), rwidth=0.8, density=True, align='left')
plt.title('ss')

plt.subplot(222)
plt.hist(distributie_sb, bins=range(0, 11), rwidth=0.8, density=True, align='left')
plt.title('sb')

plt.subplot(223)
plt.hist(distributie_bs, bins=range(0, 11), rwidth=0.8, density=True, align='left')
plt.title('bs')

plt.subplot(224)
plt.hist(distributie_bb, bins=range(0, 11), rwidth=0.8, density=True, align='left')
plt.title('bb')

plt.show()
