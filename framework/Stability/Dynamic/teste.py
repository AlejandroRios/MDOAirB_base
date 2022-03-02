
import numpy as np
A = np.zeros((5, 6), dtype=int)
print(A)
B = np.r_[np.arange(2, 10), 3].reshape(3, 3)
print(B)
r, c = 0, 2
A[r:r+B.shape[0], c:c+B.shape[1]] += B

print(A)
