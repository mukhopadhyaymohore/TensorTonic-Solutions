import math

def binomial_pmf_cdf(n: int, p: float, k: int) -> dict:
    """
    Returns a dictionary with pmf and cdf.
    """
    # Write code here
    probs = [math.comb(n, i) * (p ** i) * ((1 - p) ** (n - i)) for i in range(k + 1)]
    return {"pmf": float(probs[-1]), "cdf": float(sum(probs))}
    pass