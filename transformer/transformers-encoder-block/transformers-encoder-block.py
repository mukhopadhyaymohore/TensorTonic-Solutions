import numpy as np

def softmax(x, axis=-1):
    """Provided: numerically stable softmax."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Returns x normalized over the last axis, then scaled by gamma and shifted by beta.
    """
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return gamma * (x - mean) / np.sqrt(var + eps) + beta
    pass

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Returns multi-head attention of Q, K, V, projecting them with W_q, W_k, W_v and the merged heads with W_o.
    """
    B, Lq, D = Q.shape
    Lk = K.shape[1]
    d_k = D // num_heads
    q, k, v = Q @ W_q, K @ W_k, V @ W_v

    def split(t, L):
        return t.reshape(B, L, num_heads, d_k).transpose(0, 2, 1, 3)

    q, k, v = split(q, Lq), split(k, Lk), split(v, Lk)
    scores = q @ k.transpose(0, 1, 3, 2) / np.sqrt(d_k)
    scores -= scores.max(axis=-1, keepdims=True)
    w = np.exp(scores)
    w /= w.sum(axis=-1, keepdims=True)
    heads = w @ v
    concat = heads.transpose(0, 2, 1, 3).reshape(B, Lq, D)
    return concat @ W_o
    pass

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Returns the position-wise feed-forward network output with a ReLU hidden layer.
    """
    return np.maximum(0, x @ W1 + b1) @ W2 + b2
    pass

def encoder_block(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                  W_o: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                  W2: np.ndarray, b2: np.ndarray, gamma1: np.ndarray,
                  beta1: np.ndarray, gamma2: np.ndarray, beta2: np.ndarray,
                  num_heads: int) -> np.ndarray:
    """
    Returns the post-normalized Transformer encoder states.
    """
    attn = multi_head_attention(x, x, x, W_q, W_k, W_v, W_o, num_heads)
    z = layer_norm(x + attn, gamma1, beta1)
    ff = feed_forward(z, W1, b1, W2, b2)
    return layer_norm(z + ff, gamma2, beta2)
    pass
