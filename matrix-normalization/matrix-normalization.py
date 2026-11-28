import numpy as np

def matrix_normalization(matrix: list, axis=None, norm_type: str = "l2") -> np.ndarray:
    """
    Returns a NumPy array with the same shape as matrix.
    """
    # Write code here
    M = np.asarray(matrix, dtype=float)
    absM = np.abs(M)

    if norm_type == "l1":
        norm = np.sum(absM, axis=axis, keepdims=True)
    elif norm_type == "l2":
        norm = np.sqrt(np.sum(M ** 2, axis=axis, keepdims=True))
    else:
        norm = np.max(absM, axis=axis, keepdims=True)

    norm = np.where(norm == 0, 1.0, norm)
    return M / norm
    pass