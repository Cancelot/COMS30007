import numpy as np

A = np.random.randn(10, 10)
B = A > 0  # Boolean matrix

A = np.random.randn(10, 10)
B = A[A > 0]  # All values of A that are positive
