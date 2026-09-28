import numpy as np
from scipy.linalg import cholesky

class chol_class:
    def __init__(self, A, D, c, y_p, l_d, mu, beta):
        self.C = A.T @ A + mu * beta * D.T @ D
        self.b = mu * D.T @ (l_d * y_p) + A.T @ c

    def execute(self, x_p):
        G = cholesky(self.C, lower=True)
        x = np.linalg.solve(G.T, np.linalg.solve(G, self.b))
        return x
    