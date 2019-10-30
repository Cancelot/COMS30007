import numpy as np

mu = np.ones(2).reshape(-1, 1)
x = np.random.multivariate_normal(mu.flatten(), np.eye(2))

# generate a random matrix
M = np.random.randn(100).reshape(10, 10)

# compute the inverse of M
A = np.linalg.inv(M)

# print the diagonal elements
print(np.diag(A.dot(M)))

# print the sum of the absolute values of the off-diagonal elements
print(np.abs(A.dot(M)).sum() - np.abs(A.dot(M))[np.diag_indices(M.shape[0])].sum())oo