import numpy as np
import sys

## Comments:
## This code implements the inexact ADMM with relative error 
## criterion algorithm. The main loop is performing the minimization problem. 
## Author(s): Jiaxin Xie

from data_generation import gen_data
from algorithm import iadmm_algorithm

def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ("1", "2"):
        print(f"Usage: python3 {sys.argv[0]} 1|2")
        sys.exit(1)

    algorithm_number = int(sys.argv[1])

    # Dimensions of the problems, sparsity of vector definition
    m, n = 2048, 256
    s = int(np.round(0.1 * m))

    matrix_init = 'dct'               ## Specifying how the D matrix is defined
    method_def = 'chol'                   ## Specify how we solve the x-subproblem ('cg' or 'chol')

    ## Constants to algorithm 1 and algorithm 2 in the paper 'On inexact ADMMs with relative error criteria.'
    mu = 0.125
    beta = 30
    sigma = 0.99
    eta = 0.7

    if algorithm_number == 1:
        alpha = 0.4
        gamma = 1.0 - alpha
        zeta = (2.0 * gamma - 1.0 + np.sqrt(4.0 * gamma**2.0 + 1.0)) / (2.0 * gamma)
        is_frac = np.floor(1000 * zeta) / 1000 % 1 != 0
        if is_frac: tau = np.floor(1000 * zeta) / 1000
        else: tau = (np.floor(1000 * zeta) - 1) / 1000
    else:
        alpha = 1.0
        tau = 1.468

    print(f"Problem: m, n, s = {m}, {n}, {s}\n")

    b = np.zeros(s)
    w1 = np.zeros(n)
    for trial in range(1, 11):  
        print("*" * 60)
        A, D, x_sol, c = gen_data(m, n, s, matrix_init)
        iadmm_algorithm(A, D, w1, b, c, alpha, mu, beta, tau, sigma, eta, m, n, s, method_def)

    print("*" * 60, "\n")

    # A, D = np.random.randn(m, n), np.random.randn(s, n)
    # c = np.random.randn(m)
    # # Defining the unknown variables
    # x_p, y_p, l_d = np.zeros(n), np.random.randn(s), np.random.randn(s)

    # cg_dict = {
    #     "A" : A,
    #     "D" : D,
    #     "y" : y_p,
    #     "l" : l_d,
    #     "c" : c,
    #     "mu" : mu,
    #     "beta" : beta, 
    #     "sigma" : sigma
    # }

    # # Executing the CG method
    # conjugate_class = cg_class(**cg_dict)
    # x_p = conjugate_class.cg_method(x_p)

if __name__ == '__main__':
	main()
