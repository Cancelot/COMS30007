import numpy as np

# create a 10x10 matrix
A = np.arange(100).reshape(10 , 10)
# different direct index modes
print(A[0 , 0])  # first row and column
print(A[: , 0])  # all rows of first column
print(A[4 , :])  # all columns of fourth row
print(A[:])
# all elements

print(A[1:3, 2])  # access elements at row 1,2 and column 2
print(A[:3, 2])  # access element at row 0,1,2 and column 2
print(A[:2, 4:6])  # access element at row 0,1 and column 4,5

print(A[-1,-1])  # access the element at the last row and column
print(A[:-1, 0])  # access all the rows except for the last of the first column
print(A[0:-1, 0])  # same as the above
print(A[:10, 0])  # access all the rows of the first column

print(A.mean())  # compute the mean value of the matrix
print(A.mean(axis=0))  # compute the mean value of each column of the matrix
print(A.mean(axis=1))  # compute the mean value of each row of the matrix
