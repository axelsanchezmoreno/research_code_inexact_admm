import numpy as np

def gen_data(m, n, s, matrix_init):

    ## Defining A matrix - Normalized Gaussian, Aij ~ N(0,1) and i.i.d
    A = np.random.normal(loc=0.0, scale=1.0, size=(m, n))
    A /= np.linalg.norm(A, axis=0, keepdims=True)

    if matrix_init == 'gauss': 
        B = np.random.normal(loc=0.0, scale=1.0, size=(s, n))
        D, _ = np.linalg.qr(B)
    elif matrix_init == 'dct':
        indices = np.arange(n)
        dct = np.cos(np.pi * (indices[:, None] + 0.5) * indices / n)
        dct *= np.sqrt(2.0 / n)
        dct[0] /= np.sqrt(2.0)
        D = dct[np.random.permutation(n)[:s]]
    else: raise ValueError("matrix-definition argument must be 'gauss' or 'dct' ")

    ## y_bar - sparse vector
    y = np.zeros(s)
    s_bar = np.random.randint(1,s+1)
    y_values = np.random.normal(loc=0.0, scale=1.0, size=s_bar)
    y[:s_bar] = y_values
    np.random.shuffle(y)

    ## Creating x_sol = D.T @ y
    x = D.T @ y

    ## 'Noisy' component
    delta = 1e-3
    e = np.random.normal(loc=0.0, scale=1.0, size=m)

    ## vector output to 'noisy' system
    c = A @ x + delta * e

    return A, D, x, c