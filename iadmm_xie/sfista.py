import numpy as np

class sfista_class:
    def __init__(self, A, D, c, y_p, l_d, mu, beta):
        self.C = A.T @ A + mu * beta * D.T @ D
        self.b = mu * D.T @ (l_d * y_p) + A.T @ c

    def execute():
        pass