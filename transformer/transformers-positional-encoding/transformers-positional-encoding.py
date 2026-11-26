import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Returns the sinusoidal position matrix.
    """
    pe = np.zeros((seq_length, d_model), dtype=np.float64)
    pos = np.arange(seq_length, dtype=np.float64).reshape(-1, 1)
    two_i = np.arange(0, d_model, 2, dtype=np.float64)
    inv_freq = 1.0 / np.power(10000.0, two_i / d_model)
    angles = pos * inv_freq
    pe[:, 0::2] = np.sin(angles)
    pe[:, 1::2] = np.cos(angles)
    return pe
    pass