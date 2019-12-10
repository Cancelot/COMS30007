import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

xpred = np.load('05tmp1.npy')
varSigma = np.load('05tmp2.npy')

# Generate density
N = 300
x1 = np.linspace(np.min(xpred[:, 0]), np.max(xpred[:, 0]), N)
x2 = np.linspace(np.min(xpred[:, 1]), np.max(xpred[:, 1]), N)
x1p, x2p = np.meshgrid(x1, x2)
pos = np.vstack((x1p.flatten(), x2p.flatten())).T

# Compute posterior
Z = np.zeros((N, N))
for i in range(0, xpred.shape[0]):
    pdf = multivariate_normal(xpred[i, :].flatten(), varSigma)
    Z += pdf.pdf(pos).reshape(N, N)

# Plot figures
fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(121)
ax.scatter(xpred[:, 0], xpred[:, 1])
ax.set_xticks([])
ax.set_yticks([])
ax = fig.add_subplot(122)
ax.imshow(Z, cmap='hot')
ax.set_ylim(ax.get_ylim()[::-1])
ax.set_xticks([])
ax.set_yticks([])

plt.show()
