import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt


def posterior(a, b, X):
    a_n = a + X.sum()
    b_n = b + (X.shape[0] - X.sum())

    return beta.pdf(mu_test, a_n, b_n), a_n, b_n

# parameters to generate data
mu = 0.2
N = 100

# generate some data
X = np.random.binomial(1, mu, N)
mu_test = np.linspace(0, 1, 100)

# now lets define our prior
a = 10
b = 10

# p(mu) = Beta(alpha, beta)
prior_mean = a / (a + b)
prior_mu = beta.pdf(mu_test, a, b)

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(211)

ax.plot(mu_test, prior_mu, 'g')
ax.fill_between(mu_test, prior_mu, color='green', alpha=0.3)

ax.set_xlabel('$\mu$')
ax.set_ylabel('$p(\mu|\mathbf{x})$')

dist_ax = fig.add_subplot(212)
distances = np.zeros(X.shape[0])

# lets pick a random (uniform) point from the data
# and update our assumption with this
index = np.random.permutation(X.shape[0])
for i in range(0, X.shape[0]):
    y, a_n, b_n = posterior(a, b, X[:index[i]])
    ax.plot(mu_test, y, 'r', alpha=0.3)
    posterior_mean = a_n / (a_n + b_n)
    distances[index[i]] = abs(posterior_mean - prior_mean)

y, a_n, b_n = posterior(a, b, X)
ax.plot(mu_test, y, 'b', linewidth=3.0)

xs = list(range(X.shape[0]))
dist_ax.plot(xs, distances, 'r')

plt.tight_layout()
plt.show()
