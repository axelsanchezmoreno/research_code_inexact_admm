import numpy as np

from cg import cg_class as cg
from cholesky import chol_class as chol
from sfista import sfista_class as sfista

def iadmm_algorithm(A, D, c, mu, beta, sigma, m, n, s, method_init):
    ## Define x_p and y_p (primal) | l_d (dual)
    x_p, y_p, l_d = np.zeros(n), np.zeros(s), np.zeros(s)

    ## Choosing the technique used to solve the x-subproblem.
    method_obj = None

    if method_init == 'cg': method_obj = cg(A, D, c, y_p, l_d, mu, beta)
    elif method_init == 'chol': method_obj = chol(A, D, c, y_p, l_d, mu, beta)
    elif method_init == 'sfista': method_obj = sfista(A, D, c, y_p, l_d, mu, beta)
    else: raise ValueError("Method must be 'cg', 'chol', 'sfista' ....")

    ## solving x-subproblem
    x_p = method_obj.execute(x_p)

    ## solving y-subproblem
    y_p = np.sign(D @ x_p - (l_d / beta)) * np.maximum(np.abs(D @ x_p - (l_d / beta)) - (1/beta), 0)
    