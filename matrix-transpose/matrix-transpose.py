import numpy as np

def matrix_transpose(A: list) -> np.ndarray:
    """
    Returns the transposed matrix as a NumPy array.
    """
    # Write code here
    A = np.asarray(A)
    n, m = A.shape
    out = np.empty((m, n), dtype=A.dtype)
    for i in range(n):
        out[:, i] = A[i]
    return out
    pass
