import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

x = np.linspace(-6, 6, 200)
pdf1 = norm.pdf(x, 0, 1)
pdf2 = norm.pdf(x, 1, 3)
pdf3 = norm.pdf(x, -2.5, 0.5)

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111)

ax.plot(x, pdf1, color='r', alpha=0.5)
ax.fill_between(x, pdf1, color='r', alpha=0.3)
ax.plot(x, pdf2, color='g', alpha=0.5)
ax.fill_between(x, pdf2, color='g', alpha=0.3)
ax.plot(x, pdf3, color='b', alpha=0.5)
ax.fill_between(x, pdf3, color='b', alpha=0.3)

pdf4 = 0.3 * pdf1 + 0.2 * pdf2 + 0.5 * pdf3
ax.plot(x, pdf4, color='k', alpha=0.8, linewidth=3.0, linestyle='--')
ax.fill_between(x, pdf4, color='k', alpha=0.5)

plt.show()
