from collections import Counter
import numpy as np

def mean_median_mode(x: list) -> dict:
    """
    Returns a dictionary with mean, median, and mode.
    """
    # Write code here
    arr = np.asarray(x, dtype=float)
    mean = float(np.mean(arr))
    median = float(np.median(arr))
    counts = Counter(arr.tolist())
    max_freq = max(counts.values())
    mode = min(v for v,c in counts.items() if c == max_freq)

    return {"mean": mean, "median": median, "mode": float(mode)}
    pass