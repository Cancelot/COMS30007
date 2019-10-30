import numpy as np
import scipy.linalg

# generate a "hopefully" PSD matrix
A = np.random.randn(100).reshape(10, 10)
A = A.dot(A.T) + np.eye(A.shape[0])

# Ax=b
b = np.ones(A.shape[0]).reshape(-1, 1)

# Cholesky factor
L = scipy.linalg.cholesky(A, lower=True)

# solve system of equations using cholesky
x = scipy.linalg.cho_solve((L, True), b)

# print the "difference" between the two
print(np.abs(b - A.dot(x)).sum())

# create a hopefully PSD matrix
M = np.random.randn(100).reshape(10, 10)
M = M.dot(M.T) + np.eye(M.shape[0])

# compute eigen decomposition
U, V = np.linalg.eig(M)

# print element-wise error
print(np.abs(V.dot(np.diag(U).dot(V.T)) - M).sum())