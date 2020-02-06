import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.spatial.distance import cdist
from scipy.stats import norm


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


def rbf_kernel(x1, x2, varSigma, lengthscale):
    if x2 is None:
        d = cdist(x1, x1)
    else:
        d = cdist(x1, x2)

    K = varSigma * np.exp(-np.power(d, 2) / lengthscale)
    return K


def gp_prediction(x1, y1, xstar, noise, kernel, *args):
    k_starX = kernel(xstar, x1, *args)
    k_xx = kernel(x1, None, *args)
    if noise is not None:
        k_xx += (1 / noise) * np.eye(k_xx.shape[0])
    k_starstar = kernel(xstar, None, *args)
    mu = k_starX.dot(np.linalg.inv(k_xx)).dot(y1)
    var = k_starstar - (k_starX).dot(np.linalg.inv(k_xx)).dot(k_starX.T)
    return mu, var, xstar


def surrogate_belief(x, f, x_star, theta):
    mu_star, varSigma_star, xstar = gp_prediction(x.reshape((-1, 1)), f.reshape((-1, 1)), x_star.reshape((-1, 1)), None,
                                                  rbf_kernel, theta, 2.0)
    return mu_star, varSigma_star


def expected_improvement(f_star, mu, varSigma, x):
    # print(f"f_star: {f_star}, mu: {mu}, varSigma: {varSigma}, x: {x}")
    x_2d = np.array([x]).reshape((-1, 1))
    k_xx = rbf_kernel(x_2d, x_2d, 1.0, 2.0)
    exploit = (f_star - mu) * norm.cdf(x, mu, k_xx)
    # print(x)
    # print(f"mu: {mu}")
    exploration = k_xx * multivariate_normal.pdf(x, mean=mu, cov=k_xx)

    # norm.cdf(x, loc, scale) evaluates the cdf of the normal distribution
    return exploit + exploration


def f(x, beta=0, alpha1=1.0, alpha2=1.0):
    return np.sin(3.0 * x) - alpha1 * x + alpha2 * x ** 2 + beta * np.random.randn(x.shape[0])


def BO(f, a, X):
    fig = plt.figure(figsize=(50, 10))

    subset_index = np.random.permutation(X.shape[0])[0:3]
    print(subset_index)
    x_1 = X[subset_index]
    x_remaining = np.delete(X, subset_index)

    fs = f(x_1)
    f_prime = np.min(fs)

    iter_max = 8
    draw_skip = 0
    for ite in range(0, iter_max):
        ax = None
        if ite >= draw_skip:
            ax = fig.add_subplot(1, iter_max - draw_skip, (ite - draw_skip) + 1)
            ax.scatter(x_1, fs)

        mu_star, varSigma_star = surrogate_belief(x_1, fs, x_remaining, 1.0)
        mu_star_flat = mu_star.reshape((-1,))
        varSigma_star_flat = varSigma_star.reshape((-1,))
        alpha = np.array([a(f_prime, mu_star_flat[i], varSigma_star_flat[i], x_remaining[i]) for i in
                          range(0, x_remaining.shape[0])]).flatten()
        x_prime_index = np.argmax(alpha)
        x_prime = x_remaining[x_prime_index]
        x_remaining_old = x_remaining
        x_remaining = np.delete(x_remaining, x_prime_index)
        x_1 = np.append(x_1, x_prime)
        f_star = f(np.array([x_prime]))
        fs = np.append(fs, f_star)

        if ite >= draw_skip:
            ax.plot(X, f(X), 'r--')
            ax.plot(x_remaining_old, alpha, 'g--')

            deviation = np.diagonal(varSigma_star)
            mu_star_flat = mu_star.reshape((-1,))
            y_upper = mu_star_flat + deviation
            y_lower = mu_star_flat - deviation

            # ax.plot(x_remaining_old, y_upper, 'r--')
            # ax.plot(x_remaining_old, y_lower, 'b--')
            ax.plot(x_remaining_old, mu_star_flat, 'b--')

            ax.fill_between(x_remaining_old, y_upper, y_lower, alpha=0.5)

        if f_star < f_prime:
            f_prime = f_star

    plt.show()
    return f_prime


# choose index set for the marginal
x = np.linspace(-6, 6, 200)
# # print(x)
# # compute covariance matrix
# K = rbf_kernel(x, None, 1, 2.0)
# # K = lin_kernel(x, None, 5)
# # K = white_kernel(x, None, 5)
# # K = periodic_kernel(x, None, 1, 1.5, 5.0)
# # K = periodic_kernel(x, None, 1, 1.5, 5.0) + rbf_kernel(x, None, 1, 25.0)
# # print(K)
# # create mean vector
# mu = np.zeros(x.shape).reshape(-1,)
# # draw samples 20 from Gaussian distribution
# # f = np.random.multivariate_normal(mu, K, 20)
print(f"f' = {BO(f, expected_improvement, x)}")