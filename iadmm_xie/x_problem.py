import numpy as np

from cg import cg_class as cg
from cholesky import chol_class as chol
from sfista import sfista_class as sfista

from conditions import x_cond
from subgradients import df_subgrad

# method_init, A, D, x_p, y_p, l_d, w1, c, beta, tau, sigma, mu

def x_subproblem(method_init, A, D, x_p, y_p, l_d, w1, c, beta, tau, sigma, mu):

    ## Choosing the technique used to solve the x-subproblem.
    method_obj = None

    if method_init == 'cg': method_obj = cg(A, D, c, y_p, l_d, mu, beta)
    elif method_init == 'chol': method_obj = chol(A, D, c, y_p, l_d, mu, beta)
    elif method_init == 'sfista': method_obj = sfista(A, D, c, y_p, l_d, mu, beta)
    else: raise ValueError("Method must be 'cg', 'chol', 'sfista' ....")

    ## solving x-subproblem
    x_p = method_obj.execute(x_p) # Reminder, we implicitly satisfy condition (11). 

    ## Computing d^{k_1}
    d = df_subgrad(A, D, x_p, y_p, l_d, c, beta, mu)

    result = x_cond(D, x_p, y_p, w1, d, tau, sigma, beta)
    print(f"Condition 12 result: {result}")

    return x_p, d
