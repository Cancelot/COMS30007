import numpy as np

x = np.arange(3)
y = np.arange(3).reshape(3, 1)
A = np.arange(9).reshape(3, 3)

print(x - y)
print(y - x)
print(A - x)
print(A - y)
print(x * y)
print(A * x)
print(x * A)
print(A * y)

x = x.reshape(-1, 1)
print(x.shape)
A.dot(x)
A.dot(x.T)
A.dot(y)
A.dot(y.T)

A.dot(x)  # scalar product
A.dot(y.T)  # scalar product with transpose of y
np.outer(x , y.T)  # outer product
