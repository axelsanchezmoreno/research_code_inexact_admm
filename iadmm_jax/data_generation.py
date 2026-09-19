import numpy as np

def generate_data(delta, m, n, s):
    if m <= 0 or n <= 0:
        raise ValueError("m and n must be positive")
    if s < 1 or s > n:
        raise ValueError("s must satisfy 1 <= s <= n")
    if delta < 0:
        raise ValueError("delta must be non-negative")

    ## Method 1 - Gaussian matrix
    A = np.random.normal(loc=0.0, scale=1.0, size=(m,n))
    A /= np.linalg.norm(A, axis=0, keepdims=True)
    
    ## x_bar - sparse vector
    x_bar = np.zeros(n)
    s_bar = np.random.randint(1,s+1)
    x_values = np.random.normal(loc=0.0, scale=1.0, size=s_bar)
    x_bar[:s_bar] = x_values
    np.random.shuffle(x_bar)

    ## eps - 'noise' vector
    eps = np.random.normal(loc=0.0, scale=1.0, size=m)
    
    ## b - inexact output 
    b = A @ x_bar + delta * eps

    return A, x_bar, b