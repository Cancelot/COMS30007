import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from tempfile import TemporaryFile


# Maximum likelihood solution to W
def MLW(Y, q):
    v, W = np.linalg.eig(np.cov(Y.T))
    idx = np.argsort(np.real(v))[::-1][:q]

    return np.real(W[:, idx])


# Posterior distribution of latent variable
def posterior(w, x, mu_x, beta):
    A = np.linalg.inv(w.dot(w.T) + 1 / beta * np.eye(w.shape[0]))
    mu = w.T.dot(A.dot(x - mu_x))
    varSigma = np.eye(w.shape[1]) - w.T.dot(A.dot(w))

    return mu, varSigma


# Generate a spiral
t = np.linspace(0, 3*np.pi, 100)
x = np.zeros((t.shape[0], 2))
x[:, 0] = t * np.sin(t)
x[:, 1] = t * np.cos(t)

# Pick a random matrixx that maps to Y
w = np.random.randn(10, 2)
y = x.dot(w.T)
y += np.random.randn(y.shape[0], y.shape[1])
mu_y = np.mean(y, axis=0)

# Get maximum likelihood solution of W
w = MLW(y, 2)

# Compute predictions for latent space
xpred = np.zeros(x.shape)
varSigma = []
for i in range(0, y.shape[0]):
    xpred[i, :], varSigma = posterior(w, y[i, :], mu_y, 1/2)

np.save('05tmp1.npy', xpred)
np.save('05tmp2.npy', varSigma)
