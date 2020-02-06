import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.spatial.distance import cdist


def rbf_kernel(x1, x2, varSigma, lengthScale, noise=0):
    if x2 is None:
        d = cdist(x1, x1)
    else:
        d = cdist(x1, x2)

    K = varSigma * np.exp(-np.power(d, 2) / lengthScale) + noise

    return K


def lin_kernel(x1, x2, varSigma):
    if x2 is None:
        return varSigma * x1.dot(x1.T)
    else:
        return varSigma * x1.dot(x2.T)


def white_kernel(x1, x2, varSigma):
    if x2 is None:
        return varSigma * np.eye(x1.shape[0])
    else:
        return np.zeros(x1.shape[0], x2.shape[0])


def periodic_kernel(x1, x2, varSigma, period, lengthScale):
    if x2 is None:
        d = cdist(x1, x1)
    else:
        d = cdist(x1, x2)

    return varSigma * np.exp(-(2 * np.sin((np.pi / period) * np.sqrt(d)) ** 2) / lengthScale ** 2)


# choose index set for the marginal
x = np.linspace(-6, 6, 200).reshape(-1, 1)

# compute covariance matrix
RBF = rbf_kernel(x, None, 1.0, 2.0)
LIN = lin_kernel(x, None, 1.0)
WHI = white_kernel(x, None, 1.0)
PER = periodic_kernel(x, None, 1.0, 2.0, 2.0)

# create mean vector
mu = np.zeros(x.shape[0])

# draw 20 samples from the Gaussian distribution
RBF_f = np.random.multivariate_normal(mu, RBF, 20)
LIN_f = np.random.multivariate_normal(mu, LIN, 20)
WHI_f = np.random.multivariate_normal(mu, WHI, 20)
PER_f = np.random.multivariate_normal(mu, PER, 20)


def plot_figs():
    fig = plt.figure(num=None, figsize=(18, 5), dpi=80, facecolor='w', edgecolor='k')

    ax = fig.add_subplot(141)
    ax.plot(x, RBF_f.T)

    ax1 = fig.add_subplot(142)
    ax1.plot(x, LIN_f.T)

    ax2 = fig.add_subplot(143)
    ax2.plot(x, WHI_f.T)

    ax3 = fig.add_subplot(144)
    ax3.plot(x, PER_f.T)

    plt.show()


plot_figs()


N = 5
x = np.linspace(-3.1, 3, N)
y = np.sin(2 * np.pi / x) + x * 0.1 + 0.3 * np.random.randn((x.shape[0]))
x = np.reshape(x, (-1, 1))
y = np.reshape(y, (-1, 1))
x_star = np.linspace(-6, 6, 500).reshape(-1, 1)


def gp_prediction(x1, y1, x_star, lengthScale, varSigma, noise):
    k_starX = rbf_kernel(x_star, x1, lengthScale, varSigma, noise)
    k_xx = rbf_kernel(x1, None, lengthScale, varSigma, noise)
    k_star_star = rbf_kernel(x_star, None, lengthScale, varSigma, noise)

    mu = k_starX.dot(np.linalg.inv(k_xx)).dot(y1)
    var = k_star_star - k_starX.dot(np.linalg.inv(k_xx)).dot(k_starX.T)

    return mu, var, x_star


Nsamp = 100
mu_star, var_star, x_star = gp_prediction(x, y, x_star, 2.0, 1.0, 0.05)
f_star = np.random.multivariate_normal(mu_star.flatten(), var_star, Nsamp)
fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111)
ax.plot(x_star, f_star.T)
ax.scatter(x, y, 200, 'k', '*', zorder=3)
plt.show()
