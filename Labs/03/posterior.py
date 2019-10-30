import numpy as np
from scipy.stats import multivariate_normal


def plot_distribution(ax, mu, sigma):
    x = np.linspace(-1.5, 1.5, 100)
    x1p, x2p = np.meshgrid(x, x)
    pos = np.vstack((x1p.flatten()), x2p.flatten()).T

    pdf = multivariate_normal(mu.flatten(), sigma)
    Z = pdf.pdf(pos)
    Z = Z.reshape(100, 100)

    ax.contour(x1p, x2p, Z, 5, colors='r', lw=5, alpha=0.7)
    ax.set_xlabel('w_0')
    ax.set_xlabel('w_1')

    return

X =

index = np.random.permutation(X.shape[0])