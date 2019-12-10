import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


def plot_line(axis, w):
    # input data
    X = np.zeros((2, 2))
    X[0, 0] = -5.0
    X[1, 0] = 5.0
    X[:, 1] = 1.0

    # because of the concatenation we have to flip the transpose
    y = w.dot(X.T)
    axis.plot(X[:, 0], y)


def plot_distribution(ax, mu, sigma):
    x = np.linspace(-1.5, 1.5, 100)
    x1p, x2p = np.meshgrid(x, x)
    pos = np.vstack((x1p.flatten(), x2p.flatten())).T

    pdf = multivariate_normal(mu.flatten(), sigma)
    Z = pdf.pdf(pos)
    Z = Z.reshape(100, 100)

    ax.contour(x1p, x2p, Z, colors='r', lw=5, alpha=0.7)
    ax.set_xlabel('w_0')
    ax.set_xlabel('w_1')

    return


# create prior distribution
tau = 1.0 * np.eye(2)
w_0 = np.zeros((2, 1))

# sample from prior
n_samples = 100

w_samp = np.random.multivariate_normal(w_0.flatten(), tau, size=n_samples)

# create plot
fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(131)


for i in range(0, w_samp.shape[0]):
    plot_line(ax, w_samp[i, :])

size = 201
beta = 0.3
error = np.random.normal([0], np.array([beta]), size=size)

X = np.ones((201, 2))
X[:, 0] = np.around(np.arange(-1, 1.01, 0.01), decimals=2)

w = np.array([-1.3, 0.5])

Y = w.dot(X.T) + error

ax1 = fig.add_subplot(132)

plot_distribution(ax1, w_0, tau)


def posterior(tau, beta, X, Y):
    tau_inv = np.linalg.inv(tau)
    index = np.random.permutation(X.shape[0])
    for i in range(0, index.shape[0]):
        X_i = X[index, :]
        Y_i = Y[index].reshape(201, 1)

        mean = np.linalg.inv(tau_inv + beta * X_i.T.dot(X_i)).dot(tau_inv.dot(w_0) + beta * X_i.T.dot(Y_i))
        variance = np.linalg.inv(tau_inv + beta * X_i.T.dot(X_i))
        plot_distribution(ax1, mean, variance)

    return mean, variance


def predictive_posterior(S0, beta, x_star, X, y):
    mN, SN = posterior(S0, beta, X, y)

    m_star = mN.T.dot(x_star)
    S_star = 1.0/beta + x_star.T.dot(SN).dot(x_star)

    return m_star, S_star


x_star = np.linspace(-6, 6, 2)
ax2 = fig.add_subplot(133)
mean, variance = predictive_posterior(tau, beta, x_star, X, Y)
plot_distribution(ax2, mean, variance)

plt.tight_layout()
plt.show()
