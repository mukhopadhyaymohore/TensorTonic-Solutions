import numpy as np

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Returns projected multi-head attention outputs.
    """
    B, Lq, D = Q.shape
    Lk = K.shape[1]
    d_k = D // num_heads
    q = Q @ W_q
    k = K @ W_k
    v = V @ W_v

    def split(x, L):
        return x.reshape(B, L, num_heads, d_k).transpose(0, 2, 1, 3)

    q, k, v = split(q, Lq), split(k, Lk), split(v, Lk)
    scores = q @ k.transpose(0, 1, 3, 2) / np.sqrt(d_k)
    scores -= scores.max(axis=-1, keepdims=True)
    w = np.exp(scores)
    w /= w.sum(axis=-1, keepdims=True)

    heads = w @ v
    concat = heads.transpose(0, 2, 1, 3).reshape(B, Lq, D)
    return concat @ W_o
    pass