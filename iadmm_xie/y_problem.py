import numpy as np

def y_subproblem(D, x_p, l_d, beta):
    return np.sign(D @ x_p - (l_d / beta)) * np.maximum(np.abs(D @ x_p - (l_d / beta)) - (1/beta), 0)
    