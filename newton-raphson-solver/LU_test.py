import numpy as np
from solver import lu_decomposition

# Define matrix A and vector b as specified
A = np.array([
    [2,  0, -1,  0],
    [0,  4, -1, -1],
    [-1, -1,  5,  0],
    [0, -1,  0,  4]
], dtype=float)

b = np.array([1, 2, 3, 2], dtype=float)

# Solve Ax = b using our LU decomposition solver
x = lu_decomposition(A, b)

print("Solution x:", np.round(x, 6))

# Verify residual r = b - A x
r = b - A @ x
print("Residual r:", np.round(r, 12))
print("Residual norm ||r||:", np.linalg.norm(r))
